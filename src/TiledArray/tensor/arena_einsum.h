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

  /// Grows an already-constructed result tile in place so it covers every
  /// inner cell implied by this `left`/`right` K-panel. A SUMMA reduction
  /// shapes the result from its first K-panel only; a later panel of a
  /// contracted-dimension-sparse ToT operand can touch inner cells the first
  /// panel left null, so each subsequent panel must extend the result.
  void grow_to_cover(Result& result, const Left& left, const Right& right,
                     const math::GemmHelper& outer_gh) const;

 private:
  /// Per-output-cell inner ranges implied by one `left`/`right` K-panel.
  std::vector<typename Result::value_type::range_type> operand_inner_ranges(
      const Left& left, const Right& right,
      const math::GemmHelper& outer_gh) const;

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

/// Per-output-cell inner ranges implied by one `left`/`right` K-panel.
template <typename Result, typename Left, typename Right>
std::vector<typename Result::value_type::range_type>
ContractionArenaPlan<Result, Left, Right>::operand_inner_ranges(
    const Left& left, const Right& right,
    const math::GemmHelper& outer_gh) const {
  using inner_t = typename Result::value_type;
  using inner_range_t = typename inner_t::range_type;
  using integer = math::blas::integer;

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

  std::vector<inner_range_t> ranges;
  const std::size_t N_cells = mn * batch_sz;
  ranges.reserve(N_cells);
  for (std::size_t ord = 0; ord < N_cells; ++ord)
    ranges.emplace_back(range_for(ord));
  return ranges;
}

/// Reserves arena storage and constructs the result tensor-of-tensor tile.
template <typename Result, typename Left, typename Right>
Result ContractionArenaPlan<Result, Left, Right>::reserve_and_construct(
    const Left& left, const Right& right,
    const math::GemmHelper& outer_gh) const {
  using inner_range_t = typename Result::value_type::range_type;
  auto outer_range =
      outer_gh.template make_result_range<typename Result::range_type>(
          left.range(), right.range());
  TA_ASSERT(left.nbatch() == right.nbatch());
  const std::size_t batch_sz = static_cast<std::size_t>(left.nbatch());
  const auto ranges = operand_inner_ranges(left, right, outer_gh);
  // arena_outer_init dispatches internally on the inner-cell type.
  return detail::arena_outer_init<Result>(
      outer_range, batch_sz,
      [&ranges](std::size_t ord) -> inner_range_t { return ranges[ord]; });
}

/// Grows an already-constructed result tile to cover this K-panel's cells.
template <typename Result, typename Left, typename Right>
void ContractionArenaPlan<Result, Left, Right>::grow_to_cover(
    Result& result, const Left& left, const Right& right,
    const math::GemmHelper& outer_gh) const {
  using inner_range_t = typename Result::value_type::range_type;
  const auto ranges = operand_inner_ranges(left, right, outer_gh);
  detail::arena_tot_grow_inplace(
      result,
      [&ranges](std::size_t ord) -> inner_range_t { return ranges[ord]; });
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

/// Hadamard-outer, contraction-inner ToT x ToT product into a fresh arena
/// tile. `left` and `right` share the (Hadamard) outer layout; each result
/// outer cell is the inner GEMM of the corresponding left/right inner cells,
/// shaped by `inner_gh`. `cell_op(result_cell, left_cell, right_cell)` runs
/// the per-cell in-place contraction (e.g. the make_fused_contraction_lambda
/// callback). The per-cell op is perm-free; a non-identity `inner_perm`
/// permutes the result cells' inner modes as a slab-level post-pass.
template <typename Result, typename Left, typename Right, typename CellOp>
Result arena_hadamard_inner_contract(const Left& left, const Right& right,
                                     const math::GemmHelper& inner_gh,
                                     const CellOp& cell_op,
                                     const Permutation& inner_perm) {
  using inner_range_t = typename Result::value_type::range_type;
  TA_ASSERT(left.range().volume() == right.range().volume());
  TA_ASSERT(left.nbatch() == right.nbatch());
  const std::size_t N_cells = left.range().volume() * left.nbatch();
  auto range_fn = [&left, &right, &inner_gh](std::size_t ord) -> inner_range_t {
    const auto& lc = left.data()[ord];
    const auto& rc = right.data()[ord];
    if (lc.empty() || rc.empty()) return inner_range_t{};
    return inner_gh.template make_result_range<inner_range_t>(lc.range(),
                                                              rc.range());
  };
  Result result =
      arena_outer_init<Result>(left.range(), left.nbatch(), range_fn);
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    if (result.data()[ord].empty()) continue;
    cell_op(result.data()[ord], left.data()[ord], right.data()[ord]);
  }
  if (inner_perm && !inner_perm.is_identity())
    result = arena_inner_permute<Result>(result, inner_perm);
  return result;
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
          // run_regime_a_arena has already hoisted any operand inner
          // permutation, so l and rr are both in C-layout: the per-cell op
          // is a flat r += l * rr on congruent cells.
          fused_hadamard_inplace(r, l, rr);
        }
        return;
      }
      case RegimeAInnerKind::contraction: {
        if constexpr (is_arena_inner_cell_v<LCell> &&
                      is_arena_inner_cell_v<RCell>) {
          if (l.empty() || rr.empty()) return;
          TA_ASSERT(c_plan.has_value());
          // run_regime_a_arena has already hoisted any operand inner
          // permutation, so l and rr are in canonical (blas_layout) order:
          // the per-cell op is a single canonical GEMM into r with beta=1.
          // Uniform for TA::Tensor and ArenaTensor cells (free `gemm` CPO).
          using Scalar = typename std::remove_cv_t<ResultCell>::numeric_type;
          fused_contraction_inplace(r, l, rr, Scalar{1}, c_plan->gemm_helper);
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
    // `inner_perm` (== C.permutation at the call site) is the result *outer*
    // permutation. run_regime_a_arena applies it itself via tile.permute(pc)
    // -- byte-identical to the legacy non-arena path, and supported for an
    // arena ToT via arena_permute_shallow -- so it does not gate the plan.
    // Inner-operand and inner-result permutations are likewise handled, by
    // hoisting them to slab-level arena_inner_permute rewrites (see below).
    (void)inner_perm;

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
        // A non-canonical inner Hadamard (h_plan.perm.{AC,BC} non-identity)
        // is handled the same way as a non-canonical inner contraction:
        // run_regime_a_arena hoists each operand inner permutation to a
        // slab-level rewrite (arena_inner_permute) so both operands reach
        // C-layout before the per-cell flat r += l * rr. No need to bail.
      } else {
        plan.kind = RegimeAInnerKind::contraction;
        plan.c_plan.emplace(inner.A, inner.B, inner.C);
        // A non-canonical inner contraction (c_plan.do_perm.{A,B,C} set --
        // e.g. M/K- or M/N-interleaved inner annotations that are not
        // GEMM-absorbable transposes) is still handled: run_regime_a_arena
        // hoists each operand inner permutation, and the result inner
        // permutation, to slab-level rewrites (arena_inner_permute), leaving
        // the per-cell op a single canonical GEMM. No need to bail here.
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
        // Hoist a non-canonical inner op's operand inner permutations to
        // slab-level rewrites, so the per-cell op below stays canonical:
        // contraction -> a single canonical GEMM; Hadamard -> a flat
        // r += l * rr on congruent C-layout cells. No per-cell view permute.
        if (plan.kind == RegimeAInnerKind::contraction) {
          const auto& cp = *plan.c_plan;
          if (cp.do_perm.A)
            ai = arena_inner_permute<decltype(ai)>(ai, cp.perm.A);
          if (cp.do_perm.B)
            bi = arena_inner_permute<decltype(bi)>(bi, cp.perm.B);
        } else if (plan.kind == RegimeAInnerKind::hadamard) {
          const auto& hp = *plan.h_plan;
          if (!hp.perm.AC.is_identity())
            ai = arena_inner_permute<decltype(ai)>(ai, hp.perm.AC);
          if (!hp.perm.BC.is_identity())
            bi = arena_inner_permute<decltype(bi)>(bi, hp.perm.BC);
        }
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

      // Hoist the result inner permutation: cells were accumulated in
      // blas_layout (e) order; rewrite the slab to the C inner order.
      if (plan.kind == RegimeAInnerKind::contraction && plan.c_plan->do_perm.C)
        tile =
            arena_inner_permute<ResultTensor>(tile, plan.c_plan->perm.C.inv());
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
