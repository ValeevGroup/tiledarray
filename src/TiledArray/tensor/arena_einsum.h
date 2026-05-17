/// Arena-aware ToT einsum: plans, fused kernels, and dispatch.

#ifndef TILEDARRAY_TENSOR_ARENA_EINSUM_H__INCLUDED
#define TILEDARRAY_TENSOR_ARENA_EINSUM_H__INCLUDED

#include "TiledArray/error.h"
#include "TiledArray/math/gemm_helper.h"
#include "TiledArray/permutation.h"
#include "TiledArray/tensor/arena.h"
#include "TiledArray/tensor/arena_kernels.h"
#include "TiledArray/tensor/kernels.h"
#include "TiledArray/tensor/type_traits.h"

#include <optional>
#include <type_traits>
#include <utility>
#include <variant>

#if defined(_MSC_VER) && _MSC_VER < 1937  // VS 2022 < 17.7
#define TA_NO_UNIQUE_ADDRESS [[msvc::no_unique_address]]
#else
#define TA_NO_UNIQUE_ADDRESS [[no_unique_address]]
#endif

namespace TiledArray::detail {

/// Specifies how an inner-cell range is derived from operand inner cells.
enum class ArenaInnerShapeKind {
  left_range,        // Hadamard inner; Scale tot_x_t
  right_range,       // Scale t_x_tot
  gemm_result_range  // inner Contraction (uses inner_gh)
};

/// Inner-shape derivation plan: kind + (optional) inner GemmHelper.
struct ArenaInnerShapePlan {
  ArenaInnerShapeKind kind;
  std::optional<math::GemmHelper> inner_gh;  // only for gemm_result_range

  /// Derives one result inner range from operand inner cells.
  template <typename ResultInnerRange, typename LInner, typename RInner>
  ResultInnerRange make(const LInner& l, const RInner& r) const {
    switch (kind) {
      case ArenaInnerShapeKind::left_range:
        return l.range();
      case ArenaInnerShapeKind::right_range:
        return r.range();
      case ArenaInnerShapeKind::gemm_result_range:
        TA_ASSERT(inner_gh.has_value());
        return inner_gh->template make_result_range<ResultInnerRange>(
            l.range(), r.range());
    }
    TA_ASSERT(false);
    return ResultInnerRange{};
  }
};

/// Derives result ranges and constructs non-empty inner cells in one arena
/// slab.
template <typename Result, typename Left, typename Right>
class ContractionArenaPlan {
 public:
  /// Stores the inner shape plan used to construct result cells.
  explicit ContractionArenaPlan(ArenaInnerShapePlan p)
      : inner_plan_(std::move(p)) {}

  /// Constructs a result tile whose non-empty inner cells alias arena storage.
  Result reserve_and_construct(const Left& left, const Right& right,
                               const math::GemmHelper& outer_gh) const;

 private:
  ArenaInnerShapePlan inner_plan_{};
};

/// True when `T` is a `TA::Tensor` outer whose inner cells the arena
/// machinery knows how to allocate (legacy `TA::Tensor` ToT inner or the
/// pinned-view `ArenaTensor`). Doesn't require `is_tensor_of_tensor_v` --
/// `ArenaTensor` is deliberately not registered as `is_tensor_helper`, so
/// trait propagation can't reach it that way.
template <typename T>
inline constexpr bool is_arena_eligible_outer_v =
    is_ta_tensor_v<T> &&
    (is_ta_tensor_v<typename T::value_type> ||
     ::TiledArray::is_arena_tensor_v<typename T::value_type>);

/// True when `T` is an inner-cell type that the arena machinery treats as
/// tensor-shaped (as opposed to a scalar in mixed Scale ops). Covers the
/// legacy `TA::Tensor` inner and the pinned `ArenaTensor`. Used by the
/// regime-A `accumulate` dispatch to distinguish the tensor-inner branches
/// from the scalar-inner ones in `scale_left`/`scale_right` cases.
template <typename T>
inline constexpr bool is_arena_inner_cell_v =
    is_ta_tensor_v<T> || ::TiledArray::is_arena_tensor_v<T>;

/// True when the result is an arena-eligible outer; gates the arena
/// allocation path in cont_engine.
template <typename Result, typename Left, typename Right>
inline constexpr bool is_contraction_arena_tot_v =
    is_arena_eligible_outer_v<Result>;

/// Stores an arena plan for ToT results and std::monostate otherwise.
template <typename Result, typename Left, typename Right>
using arena_plan_storage_t =
    std::conditional_t<is_contraction_arena_tot_v<Result, Left, Right>,
                       std::optional<ContractionArenaPlan<Result, Left, Right>>,
                       std::monostate>;

/// Builds a contraction arena plan when the result and inner permutation allow
/// it.
template <typename Result, typename Left, typename Right>
auto make_contraction_arena_plan(ArenaInnerShapeKind inner_kind,
                                 std::optional<math::GemmHelper> inner_gh,
                                 const Permutation& inner_perm)
    -> std::optional<ContractionArenaPlan<Result, Left, Right>> {
  if (arena_disabled()) return std::nullopt;
  if constexpr (!is_contraction_arena_tot_v<Result, Left, Right>) {
    return std::nullopt;
  } else {
    if (bool(inner_perm) && !inner_perm.is_identity()) return std::nullopt;
    if (inner_kind != ArenaInnerShapeKind::gemm_result_range)
      inner_gh.reset();
    else if (!inner_gh.has_value())
      return std::nullopt;
    return std::optional<ContractionArenaPlan<Result, Left, Right>>(
        std::in_place, ArenaInnerShapePlan{inner_kind, std::move(inner_gh)});
  }
}

/// Reserves arena storage and constructs the result tensor-of-tensor tile.
template <typename Result, typename Left, typename Right>
Result ContractionArenaPlan<Result, Left, Right>::reserve_and_construct(
    const Left& left, const Right& right,
    const math::GemmHelper& outer_gh) const {
  using inner_t = typename Result::value_type;
  using inner_range_t = typename inner_t::range_type;
  using integer = math::blas::integer;

  auto outer_range =
      outer_gh.template make_result_range<typename Result::range_type>(
          left.range(), right.range());

  integer M, N, K;
  outer_gh.compute_matrix_sizes(M, N, K, left.range(), right.range());
  const integer lda = (outer_gh.left_op() == math::blas::NoTranspose) ? K : M;
  const integer ldb = (outer_gh.right_op() == math::blas::NoTranspose) ? N : K;
  TA_ASSERT(left.nbatch() == right.nbatch());
  const std::size_t batch_sz = static_cast<std::size_t>(left.nbatch());
  const std::size_t mn =
      static_cast<std::size_t>(M) * static_cast<std::size_t>(N);

  auto range_for = [&](std::size_t ord) -> inner_range_t {
    if (mn == 0) return inner_range_t{};
    const integer b = static_cast<integer>(ord / mn);
    const integer rem = static_cast<integer>(ord % mn);
    const integer m = rem / N;
    const integer n = rem % N;

    if (inner_plan_.kind == ArenaInnerShapeKind::left_range) {
      if constexpr (is_arena_eligible_outer_v<Left>) {
        const auto* lbase = left.batch_data(static_cast<std::size_t>(b));
        for (integer k = 0; k != K; ++k) {
          const auto aoff = (outer_gh.left_op() == math::blas::NoTranspose)
                                ? m * lda + k
                                : k * lda + m;
          const auto& lc = *(lbase + aoff);
          if (!lc.empty()) return lc.range();
        }
      }
      return inner_range_t{};
    }
    if (inner_plan_.kind == ArenaInnerShapeKind::right_range) {
      if constexpr (is_arena_eligible_outer_v<Right>) {
        const auto* rbase = right.batch_data(static_cast<std::size_t>(b));
        for (integer k = 0; k != K; ++k) {
          const auto boff = (outer_gh.right_op() == math::blas::NoTranspose)
                                ? k * ldb + n
                                : n * ldb + k;
          const auto& rc = *(rbase + boff);
          if (!rc.empty()) return rc.range();
        }
      }
      return inner_range_t{};
    }
    // gemm_result_range needs both operands to be ToT.
    if constexpr (is_arena_eligible_outer_v<Left> &&
                  is_arena_eligible_outer_v<Right>) {
      const auto* lbase = left.batch_data(static_cast<std::size_t>(b));
      const auto* rbase = right.batch_data(static_cast<std::size_t>(b));
      for (integer k = 0; k != K; ++k) {
        const auto aoff = (outer_gh.left_op() == math::blas::NoTranspose)
                              ? m * lda + k
                              : k * lda + m;
        const auto boff = (outer_gh.right_op() == math::blas::NoTranspose)
                              ? k * ldb + n
                              : n * ldb + k;
        const auto& lc = *(lbase + aoff);
        const auto& rc = *(rbase + boff);
        if (lc.empty() || rc.empty()) continue;
        return inner_plan_.template make<inner_range_t>(lc, rc);
      }
    }
    return inner_range_t{};
  };

  // arena_outer_init dispatches internally on the inner-cell type.
  return detail::arena_outer_init<Result>(outer_range, batch_sz, range_for);
}

/// Accumulates a contraction into an already-allocated result cell.
template <typename Result, typename Left, typename Right, typename Scalar>
void fused_contraction_inplace(Result& result, const Left& left,
                               const Right& right, Scalar alpha,
                               const math::GemmHelper& gh) {
  if (left.empty() || right.empty()) return;
  TA_ASSERT(!result.empty());
  // Free `gemm` CPO, not the member: `ArenaTensor` (a view) provides only the
  // free in-place overload, while `TA::Tensor` is reached via the
  // `tile_interface.h` CPO that forwards to its member.
  gemm(result, left, right, alpha, gh);
}

/// Accumulates an elementwise product into an already-allocated result cell.
template <typename Result, typename Left, typename Right>
void fused_hadamard_inplace(Result& result, const Left& left,
                            const Right& right) {
  if (left.empty() || right.empty()) return;
  TA_ASSERT(!result.empty());
  inplace_tensor_op(
      [](typename Result::value_type& MADNESS_RESTRICT r,
         const typename Left::value_type& MADNESS_RESTRICT l,
         const typename Right::value_type& MADNESS_RESTRICT rr) {
        r += l * rr;
      },
      result, left, right);
}

/// Accumulates a scaled elementwise product into an allocated result cell.
template <typename Result, typename Left, typename Right, typename Scalar>
void fused_hadamard_scaled_inplace(Result& result, const Left& left,
                                   const Right& right, Scalar factor) {
  if (left.empty() || right.empty()) return;
  TA_ASSERT(!result.empty());
  // Preserve historical grouping: r += (l * rr) * factor.
  inplace_tensor_op(
      [factor](typename Result::value_type& MADNESS_RESTRICT r,
               const typename Left::value_type& MADNESS_RESTRICT l,
               const typename Right::value_type& MADNESS_RESTRICT rr) {
        r += (l * rr) * factor;
      },
      result, left, right);
}

/// Accumulates a ToT cell scaled by a scalar right operand.
template <typename Result, typename Left, typename Scalar>
void fused_scale_tot_x_t_inplace(Result& result, const Left& left,
                                 const Scalar& s) {
  if (left.empty()) return;
  TA_ASSERT(!result.empty());
  inplace_tensor_op(
      [s](typename Result::value_type& MADNESS_RESTRICT r,
          const typename Left::value_type& MADNESS_RESTRICT l) { r += l * s; },
      result, left);
}

/// Accumulates a ToT right operand scaled by a scalar left operand.
template <typename Result, typename Scalar, typename Right>
void fused_scale_t_x_tot_inplace(Result& result, const Scalar& s,
                                 const Right& right) {
  if (right.empty()) return;
  TA_ASSERT(!result.empty());
  inplace_tensor_op(
      [s](typename Result::value_type& MADNESS_RESTRICT r,
          const typename Right::value_type& MADNESS_RESTRICT rr) {
        r += rr * s;
      },
      result, right);
}

/// Creates a fused contraction callback.
template <typename Result, typename Left, typename Right, typename Op>
auto make_fused_contraction_lambda(Op contrreduce_op) {
  return
      [contrreduce_op](Result& result, const Left& left, const Right& right) {
        TA_ASSERT(!contrreduce_op.perm());
        fused_contraction_inplace(result, left, right, contrreduce_op.factor(),
                                  contrreduce_op.gemm_helper());
      };
}

/// Creates a fused Hadamard callback.
template <typename Result, typename Left, typename Right>
auto make_fused_hadamard_lambda() {
  return [](Result& result, const Left& left, const Right& right) {
    fused_hadamard_inplace(result, left, right);
  };
}

/// Creates a fused scaled-Hadamard callback.
template <typename Result, typename Left, typename Right, typename Scalar>
auto make_fused_hadamard_scaled_lambda(Scalar factor) {
  return [factor](Result& result, const Left& left, const Right& right) {
    fused_hadamard_scaled_inplace(result, left, right, factor);
  };
}

/// Creates a fused ToT-times-scalar callback.
template <typename Result, typename Left, typename Right>
auto make_fused_scale_tot_x_t_lambda() {
  return [](Result& result, const Left& left, const Right& right) {
    fused_scale_tot_x_t_inplace(result, left, right);
  };
}

/// Creates a fused scalar-times-ToT callback.
template <typename Result, typename Left, typename Right>
auto make_fused_scale_t_x_tot_lambda() {
  return [](Result& result, const Left& left, const Right& right) {
    fused_scale_t_x_tot_inplace(result, left, right);
  };
}

/// Discriminates the per-cell operation used by the arena regime-A path.
enum class RegimeAInnerKind {
  hadamard,
  contraction,
  scale_left,  // ToT × plain T → ToT (right operand contributes scalars)
  scale_right  // plain T × ToT → ToT (left operand contributes scalars)
};

/// Permute the extents of `src` by `perm` and materialize a range of type
/// `RangeT`. Generic over the inner-cell range types regime-A einsum sees:
/// `TA::Range` (legacy `Tensor<Tensor>` inners) and `btas::zb::RangeNd`
/// (`Tensor<ArenaTensor>` inners). `Permutation * Range` only exists for
/// `TA::Range`, so the permutation is applied to a plain extent vector and
/// the target range is rebuilt from the result.
template <typename RangeT, typename SrcRange>
RangeT arena_make_permuted_range(const TiledArray::Permutation& perm,
                                 const SrcRange& src) {
  const std::size_t rank = src.rank();
  const auto& src_ext = src.extent();
  container::svector<std::size_t> ext(rank);
  for (std::size_t d = 0; d < rank; ++d)
    ext[d] = static_cast<std::size_t>(src_ext[d]);
  if (perm && !perm.is_identity()) {
    TA_ASSERT(perm.size() == rank);
    return RangeT(perm * ext);
  }
  return RangeT(ext);
}

/// Holds the inner operation plan for arena regime-A dispatch.
template <typename Result, typename A, typename B, typename Inner>
struct RegimeAArenaPlan {
  using Annot = ::Einsum::Index<std::string>;

  bool active = false;
  RegimeAInnerKind kind = RegimeAInnerKind::hadamard;

  // Exactly one plan optional is engaged; optionals avoid default construction.
  std::optional<TensorHadamardPlan<Annot>> h_plan{};
  std::optional<TensorContractionPlan<Annot>> c_plan{};

  /// Derives the result inner range from a non-empty input-cell pair.
  template <typename InnerRange, typename LRange, typename RRange>
  InnerRange derive_inner_range(const LRange& l_range,
                                const RRange& r_range) const {
    switch (kind) {
      case RegimeAInnerKind::hadamard:
        TA_ASSERT(h_plan.has_value());
        return arena_make_permuted_range<InnerRange>(h_plan->perm.AC, l_range);
      case RegimeAInnerKind::contraction: {
        TA_ASSERT(c_plan.has_value());
        const auto& p = *c_plan;
        using PlanIndices = std::remove_cvref_t<decltype(p.A)>;
        using PlanIndex = typename PlanIndices::value_type;
        using Extent =
            std::remove_cv_t<typename decltype(std::declval<TiledArray::Range>()
                                                   .extent())::value_type>;
        using ExtentMap = ::Einsum::index::IndexMap<PlanIndex, Extent>;
        ExtentMap extent = (ExtentMap{p.A, l_range.extent()} |
                            ExtentMap{p.B, r_range.extent()});
        container::vector<Extent> rng;
        rng.reserve(p.e.size());
        for (auto&& ix : p.e) rng.emplace_back(extent[ix]);
        return InnerRange(rng);
      }
      case RegimeAInnerKind::scale_left:
        // Scale-left preserves the ToT operand's inner range.
        return InnerRange(l_range);
      case RegimeAInnerKind::scale_right:
        return InnerRange(r_range);
    }
    TA_ASSERT(false && "RegimeAInnerKind: unhandled kind");
    return InnerRange{};
  }

  /// Accumulates one input-cell pair into the result cell.
  template <typename ResultCell, typename LCell, typename RCell>
  void accumulate(ResultCell& r, const LCell& l, const RCell& rr) const {
    switch (kind) {
      case RegimeAInnerKind::hadamard: {
        if constexpr (is_arena_inner_cell_v<LCell> &&
                      is_arena_inner_cell_v<RCell>) {
          if (l.empty() || rr.empty()) return;
          TA_ASSERT(h_plan.has_value());
          const auto& hp = *h_plan;
          TA_ASSERT((hp.no_perm || hp.perm_b) &&
                    "regime-A arena plan should be inactive for unsupported "
                    "Hadamard perm branches (perm_to_c/perm_a/else)");
          fused_hadamard_inplace(r, l, rr);
        }
        return;
      }
      case RegimeAInnerKind::contraction: {
        if constexpr (is_arena_inner_cell_v<LCell> &&
                      is_arena_inner_cell_v<RCell>) {
          if (l.empty() || rr.empty()) return;
          TA_ASSERT(c_plan.has_value());
          // make_regime_a_arena_plan bails out unless do_perm.{A,B,C}
          // are all false -- so tensor_contract_to's fast path fires: one
          // GEMM into r with beta=1. Uniform for TA::Tensor and ArenaTensor
          // inner cells (free `gemm` CPO dispatches by type).
          using Scalar = typename std::remove_cv_t<ResultCell>::numeric_type;
          tensor_contract_to(r, l, rr, Scalar{1}, *c_plan);
        }
        return;
      }
      case RegimeAInnerKind::scale_left: {
        // Scale-left receives a ToT inner cell and a scalar.
        if constexpr (is_arena_inner_cell_v<LCell> &&
                      !is_arena_inner_cell_v<RCell>) {
          if (l.empty()) return;
          fused_scale_tot_x_t_inplace(r, l, rr);
        }
        return;
      }
      case RegimeAInnerKind::scale_right: {
        if constexpr (!is_arena_inner_cell_v<LCell> &&
                      is_arena_inner_cell_v<RCell>) {
          if (rr.empty()) return;
          fused_scale_t_x_tot_inplace(r, l, rr);
        }
        return;
      }
    }
  }
};

/// Builds an arena regime-A plan when result and permutation constraints allow
/// it.
template <typename Result, typename A, typename B, typename Inner,
          typename PermT>
auto make_regime_a_arena_plan(const A& a, const B& b, const Inner& inner,
                              const PermT& inner_perm)
    -> RegimeAArenaPlan<Result, A, B, Inner> {
  using Plan = RegimeAArenaPlan<Result, A, B, Inner>;
  Plan plan;
  if (arena_disabled()) return plan;
  if constexpr (!is_arena_eligible_outer_v<Result>) {
    return plan;
  } else {
    if (bool(inner_perm) && !inner_perm.is_identity()) return plan;

    using ArrayA_t = std::remove_cvref_t<decltype(a.array)>;
    using ArrayB_t = std::remove_cvref_t<decltype(b.array)>;
    // "Tot" here means "tile is a ToT-like thing whose inner cell is the
    // tensor we want to operate on"; covers both legacy TA::Tensor inners
    // and pinned ArenaTensor inners.
    constexpr bool a_is_tot =
        is_arena_eligible_outer_v<typename ArrayA_t::value_type>;
    constexpr bool b_is_tot =
        is_arena_eligible_outer_v<typename ArrayB_t::value_type>;

    if constexpr (a_is_tot && b_is_tot) {
      if (static_cast<bool>(inner.h)) {
        plan.kind = RegimeAInnerKind::hadamard;
        plan.h_plan.emplace(inner.A, inner.B, inner.C);
        const auto& hp = *plan.h_plan;
        if (!(hp.no_perm || hp.perm_b)) return plan;
      } else {
        plan.kind = RegimeAInnerKind::contraction;
        plan.c_plan.emplace(inner.A, inner.B, inner.C);
        const auto& cp = *plan.c_plan;
        // Inner contraction must be canonically aligned -- the accumulate
        // path collapses to a single GEMM (one tensor_contract_to call,
        // fast path). If any operand or the result needs permuting, leave
        // the plan inactive and let the legacy einsum path handle it.
        //
        // TODO(arena-einsum-perm): the legacy fallback value-returns inner
        // tensors and cannot run for ArenaTensor inner cells. To cover the
        // general case, handle do_perm here instead of bailing: GEMM into an
        // owning `TA::Tensor` temp, permute the temp, then fold it into the
        // pre-allocated arena result cell in place. Note einsum-emitted
        // contractions can carry *real* permutations (M/K- or M/N-interleaved
        // inner annotations), not just GEMM-absorbable transposes -- the temp
        // path must handle both. See the einsum/gemm/perm discussion.
        if (cp.do_perm.A || cp.do_perm.B || cp.do_perm.C) return plan;
      }
    } else if constexpr (a_is_tot && !b_is_tot) {
      plan.kind = RegimeAInnerKind::scale_left;
    } else if constexpr (!a_is_tot && b_is_tot) {
      plan.kind = RegimeAInnerKind::scale_right;
    } else {
      return plan;
    }
    plan.active = true;
    (void)a;
    (void)b;
    return plan;
  }
}

/// Runs the arena regime-A path for one H-slice when the plan is active.
template <typename Plan, typename HIndex, typename TermA, typename TermB,
          typename TermC, typename LocalTiles, typename Tiles, typename Trange>
bool run_regime_a_arena(const Plan& plan, const HIndex& h, std::size_t batch,
                        const TermA& A, const TermB& B, const TermC& C,
                        LocalTiles& C_local_tiles, const Tiles& tiles,
                        const Trange& trange) {
  if (!plan.active) return false;

  using ResultTensor = typename LocalTiles::value_type::second_type;
  // Guard avoids naming inner-cell APIs for non-ToT instantiations.
  using ArrayA_t = std::remove_cvref_t<decltype(A.array)>;
  using ArrayB_t = std::remove_cvref_t<decltype(B.array)>;
  // ToT-like in the regime-A sense: tile is an arena-eligible outer
  // (legacy TA::Tensor inner or pinned ArenaTensor inner).
  constexpr bool a_is_tot =
      is_arena_eligible_outer_v<typename ArrayA_t::value_type>;
  constexpr bool b_is_tot =
      is_arena_eligible_outer_v<typename ArrayB_t::value_type>;
  if constexpr (!is_arena_eligible_outer_v<ResultTensor> ||
                (!a_is_tot && !b_is_tot)) {
    (void)h;
    (void)batch;
    (void)A;
    (void)B;
    (void)C;
    (void)C_local_tiles;
    (void)tiles;
    (void)trange;
    return false;
  } else {
    using InnerT = typename ResultTensor::value_type;
    using InnerRange = typename InnerT::range_type;

    const auto& pa = A.permutation;
    const auto& pb = B.permutation;
    const auto& pc = C.permutation;
    auto const c = apply(pc, h);

    if constexpr (a_is_tot && b_is_tot) {
      using IIndex = ::Einsum::index::Index<std::size_t>;
      auto range_for = [&](std::size_t k) -> InnerRange {
        if (k >= batch) return InnerRange{};
        for (IIndex i : tiles) {
          const auto pahi_inv = apply_inverse(pa, h + i);
          const auto pbhi_inv = apply_inverse(pb, h + i);
          if (A.array.is_zero(pahi_inv) || B.array.is_zero(pbhi_inv)) continue;
          auto ai = A.array.find(pahi_inv).get();
          auto bi = B.array.find(pbhi_inv).get();
          if (pa) ai = ai.permute(pa);
          if (pb) bi = bi.permute(pb);
          auto shape = trange.tile(i);
          ai = ai.reshape(shape, batch);
          bi = bi.reshape(shape, batch);
          auto aik = ai.batch(k);
          auto bik = bi.batch(k);
          auto vol = aik.total_size();
          TA_ASSERT(vol == bik.total_size());
          for (decltype(vol) j = 0; j < vol; ++j) {
            const auto& l_inner = aik.data()[j];
            const auto& r_inner = bik.data()[j];
            if (l_inner.empty() || r_inner.empty()) continue;
            return plan.template derive_inner_range<InnerRange>(
                l_inner.range(), r_inner.range());
          }
        }
        return InnerRange{};
      };

      ResultTensor tile = arena_outer_init<ResultTensor>(
          TiledArray::Range{batch}, /*batch_sz=*/1, range_for);

      for (IIndex i : tiles) {
        const auto pahi_inv = apply_inverse(pa, h + i);
        const auto pbhi_inv = apply_inverse(pb, h + i);
        if (A.array.is_zero(pahi_inv) || B.array.is_zero(pbhi_inv)) continue;
        auto ai = A.array.find(pahi_inv).get();
        auto bi = B.array.find(pbhi_inv).get();
        if (pa) ai = ai.permute(pa);
        if (pb) bi = bi.permute(pb);
        auto shape = trange.tile(i);
        ai = ai.reshape(shape, batch);
        bi = bi.reshape(shape, batch);
        for (std::size_t k = 0; k < batch; ++k) {
          auto& cell = tile({k});
          if (cell.empty()) continue;
          auto aik = ai.batch(k);
          auto bik = bi.batch(k);
          auto vol = aik.total_size();
          TA_ASSERT(vol == bik.total_size());
          for (decltype(vol) j = 0; j < vol; ++j) {
            const auto& l_inner = aik.data()[j];
            const auto& r_inner = bik.data()[j];
            plan.accumulate(cell, l_inner, r_inner);
          }
        }
      }

      auto shape = apply_inverse(pc, C.array.trange().tile(c));
      tile = tile.reshape(shape);
      if (pc) tile = tile.permute(pc);
      C_local_tiles.emplace_back(std::move(c), std::move(tile));
      return true;
    } else {
      // Scale path has exactly one ToT operand and one scalar-cell operand.
      using IIndex = ::Einsum::index::Index<std::size_t>;
      auto range_for = [&](std::size_t k) -> InnerRange {
        if (k >= batch) return InnerRange{};
        for (IIndex i : tiles) {
          const auto pahi_inv = apply_inverse(pa, h + i);
          const auto pbhi_inv = apply_inverse(pb, h + i);
          if (A.array.is_zero(pahi_inv) || B.array.is_zero(pbhi_inv)) continue;
          auto ai = A.array.find(pahi_inv).get();
          auto bi = B.array.find(pbhi_inv).get();
          if (pa) ai = ai.permute(pa);
          if (pb) bi = bi.permute(pb);
          auto shape = trange.tile(i);
          ai = ai.reshape(shape, batch);
          bi = bi.reshape(shape, batch);
          auto aik = ai.batch(k);
          auto bik = bi.batch(k);
          if constexpr (a_is_tot) {
            auto vol = aik.total_size();
            for (decltype(vol) j = 0; j < vol; ++j) {
              const auto& l_inner = aik.data()[j];
              if (l_inner.empty()) continue;
              return InnerRange(l_inner.range());
            }
          } else {
            auto vol = bik.total_size();
            for (decltype(vol) j = 0; j < vol; ++j) {
              const auto& r_inner = bik.data()[j];
              if (r_inner.empty()) continue;
              return InnerRange(r_inner.range());
            }
          }
        }
        return InnerRange{};
      };

      ResultTensor tile = arena_outer_init<ResultTensor>(
          TiledArray::Range{batch}, /*batch_sz=*/1, range_for);

      for (IIndex i : tiles) {
        const auto pahi_inv = apply_inverse(pa, h + i);
        const auto pbhi_inv = apply_inverse(pb, h + i);
        if (A.array.is_zero(pahi_inv) || B.array.is_zero(pbhi_inv)) continue;
        auto ai = A.array.find(pahi_inv).get();
        auto bi = B.array.find(pbhi_inv).get();
        if (pa) ai = ai.permute(pa);
        if (pb) bi = bi.permute(pb);
        auto shape = trange.tile(i);
        ai = ai.reshape(shape, batch);
        bi = bi.reshape(shape, batch);
        for (std::size_t k = 0; k < batch; ++k) {
          auto& cell = tile({k});
          if (cell.empty()) continue;
          auto aik = ai.batch(k);
          auto bik = bi.batch(k);
          auto vol = aik.total_size();
          TA_ASSERT(vol == bik.total_size());
          for (decltype(vol) j = 0; j < vol; ++j) {
            const auto& l_elem = aik.data()[j];
            const auto& r_elem = bik.data()[j];
            plan.accumulate(cell, l_elem, r_elem);
          }
        }
      }

      auto shape = apply_inverse(pc, C.array.trange().tile(c));
      tile = tile.reshape(shape);
      if (pc) tile = tile.permute(pc);
      C_local_tiles.emplace_back(std::move(c), std::move(tile));
      return true;
    }
  }
}

}  // namespace TiledArray::detail

#endif  // TILEDARRAY_TENSOR_ARENA_EINSUM_H__INCLUDED
