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
#include "TiledArray/util/annotation.h"

#include <atomic>
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
  left_range,         // Hadamard inner; Scale tot_x_t
  right_range,        // Scale t_x_tot
  gemm_result_range,  // inner Contraction (uses inner_gh)
  unit_range  // phantom-unit denest: a unit-extent [1]^phantom_rank cell
              // independent of operand inner ranges (the inner product is a
              // flat dot; see RegimeAInnerKind::phantom_dot)
};

/// Inner-shape derivation plan: kind + (optional) inner GemmHelper.
struct ArenaInnerShapePlan {
  ArenaInnerShapeKind kind;
  std::optional<math::GemmHelper> inner_gh;  // only for gemm_result_range
  std::size_t phantom_rank = 0;              // only for unit_range

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
      case ArenaInnerShapeKind::unit_range: {
        container::vector<std::size_t> ext(phantom_rank, std::size_t{1});
        return ResultInnerRange(ext);
      }
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
  /// Deduced return type: spelling `Result::value_type::range_type` in the
  /// declaration would make the whole class ill-formed for a non-ToT
  /// `Result`, but `make_contraction_arena_plan` names this class in its
  /// return type unconditionally (and returns nullopt for non-ToT).
  auto operand_inner_ranges(const Left& left, const Right& right,
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
                                 const Permutation& inner_perm,
                                 std::size_t phantom_rank = 0)
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
        std::in_place,
        ArenaInnerShapePlan{inner_kind, std::move(inner_gh), phantom_rank});
  }
}

/// Per-output-cell inner ranges implied by one `left`/`right` K-panel.
template <typename Result, typename Left, typename Right>
auto ContractionArenaPlan<Result, Left, Right>::operand_inner_ranges(
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

#ifdef TA_STRIDED_DGEMM_COUNT
inline std::atomic<std::size_t> g_strided_dgemm_ce_e_calls{0};
#endif

/// ce+e strided-DGEMM core (inner OUTER-PRODUCT), looped over the Hadamard-
/// folded nbatch. For each batch b and result cell (m,n):
///   C[m,n](p,q) += factor * sum_k L[m,k](p) * R[k,n](q)
/// as ONE P x Q DGEMM riding the outer-contracted k into BLAS K via the
/// inter-cell slab stride (zero-copy) when the k-run is "clean" (all cells
/// present, uniform inner size, single constant stride); else an inline per-k
/// rank-1 fallback for THAT cell only. Orientation-aware (left_op/right_op pick
/// per-(m,n,k) offsets). M=left-external, N=right-external, K=outer-contracted.
template <typename ResultOuter, typename LeftOuter, typename RightOuter>
void arena_strided_dgemm_ce_e(ResultOuter& C, const LeftOuter& L,
                              const RightOuter& R, std::size_t M, std::size_t N,
                              std::size_t K, math::blas::Op left_op,
                              math::blas::Op right_op, double factor) {
  namespace blas = TiledArray::math::blas;
  using integer = blas::integer;
  static_assert(is_tensor_view_v<typename ResultOuter::value_type> &&
                    is_tensor_view_v<typename LeftOuter::value_type> &&
                    is_tensor_view_v<typename RightOuter::value_type>,
                "arena_strided_dgemm_ce_e: arena (view) inner cells only");
  static_assert(
      std::is_same_v<typename ResultOuter::value_type::numeric_type, double> &&
          std::is_same_v<typename LeftOuter::value_type::numeric_type, double> &&
          std::is_same_v<typename RightOuter::value_type::numeric_type, double>,
      "arena_strided_dgemm_ce_e: double inner storage only");
  if (M == 0 || N == 0 || K == 0) return;
  const std::size_t nbatch = static_cast<std::size_t>(C.nbatch());
  if (nbatch == 0) return;
  const bool shape_ok =
      (C.range().volume() == M * N && L.range().volume() == M * K &&
       R.range().volume() == K * N &&
       static_cast<std::size_t>(L.nbatch()) == nbatch &&
       static_cast<std::size_t>(R.nbatch()) == nbatch);
  TA_ASSERT(shape_ok);
  if (!shape_ok) return;
  const std::size_t lda = (left_op == blas::NoTranspose) ? K : M;
  const std::size_t ldb = (right_op == blas::NoTranspose) ? N : K;
  auto a_off = [&](std::size_t m, std::size_t k) {
    return (left_op == blas::NoTranspose) ? m * lda + k : k * lda + m;
  };
  auto b_off = [&](std::size_t k, std::size_t n) {
    return (right_op == blas::NoTranspose) ? k * ldb + n : n * ldb + k;
  };
  const auto* lc = L.data();
  const auto* rc = R.data();
  auto* cc = C.data();
  for (std::size_t b = 0; b < nbatch; ++b) {
    const std::size_t cbase = b * M * N;
    const std::size_t lbase = b * M * K;
    const std::size_t rbase = b * K * N;
    for (std::size_t m = 0; m < M; ++m) {
      for (std::size_t n = 0; n < N; ++n) {
        auto& Cc = cc[cbase + m * N + n];
        if (!Cc) continue;
        const auto& l0 = lc[lbase + a_off(m, 0)];
        const auto& r0 = rc[rbase + b_off(0, n)];
        long P = l0 ? static_cast<long>(l0.size()) : -1;
        long Q = r0 ? static_cast<long>(r0.size()) : -1;
        bool clean = (P > 0 && Q > 0 && static_cast<long>(Cc.size()) == P * Q);
        // presence-first: verify every k-cell present + uniform size BEFORE
        // any .data() pointer subtraction.
        for (std::size_t k = 0; clean && k < K; ++k) {
          const auto& lk = lc[lbase + a_off(m, k)];
          const auto& rk = rc[rbase + b_off(k, n)];
          if (!lk || static_cast<long>(lk.size()) != P) clean = false;
          else if (!rk || static_cast<long>(rk.size()) != Q) clean = false;
        }
        long ldA = P, ldB = Q;
        if (clean && K > 1) {
          ldA = static_cast<long>(lc[lbase + a_off(m, 1)].data() - l0.data());
          ldB = static_cast<long>(rc[rbase + b_off(1, n)].data() - r0.data());
          if (ldA < P || ldB < Q) clean = false;
          for (std::size_t k = 0; clean && k < K; ++k) {
            if (lc[lbase + a_off(m, k)].data() !=
                l0.data() + static_cast<std::ptrdiff_t>(k) * ldA)
              clean = false;
            else if (rc[rbase + b_off(k, n)].data() !=
                     r0.data() + static_cast<std::ptrdiff_t>(k) * ldB)
              clean = false;
          }
        }
        if (clean) {
          // C(P x Q) += factor * Lmat(P x K) . Rmat^T... realized as
          // gemm(Transpose, NoTranspose): A=K x P slab, B=K x Q slab.
          blas::gemm(blas::Transpose, blas::NoTranspose,
                     /*M=*/static_cast<integer>(P),
                     /*N=*/static_cast<integer>(Q),
                     /*K=*/static_cast<integer>(K), factor,
                     /*A=*/l0.data(), /*lda=*/static_cast<integer>(ldA),
                     /*B=*/r0.data(), /*ldb=*/static_cast<integer>(ldB),
                     /*beta=*/1.0,
                     /*C=*/Cc.data(), /*ldc=*/static_cast<integer>(Q));
#ifdef TA_STRIDED_DGEMM_COUNT
          g_strided_dgemm_ce_e_calls.fetch_add(1, std::memory_order_relaxed);
#endif
        } else {
          // inline per-k rank-1 fallback for THIS cell (computed once)
          double* c = Cc.data();
          for (std::size_t k = 0; k < K; ++k) {
            const auto& lk = lc[lbase + a_off(m, k)];
            const auto& rk = rc[rbase + b_off(k, n)];
            if (!lk || !rk) continue;
            const std::size_t pp = lk.size(), qq = rk.size();
            if (static_cast<long>(Cc.size()) != static_cast<long>(pp * qq))
              continue;
            const double* lp = lk.data();
            const double* rp = rk.data();
            for (std::size_t p = 0; p < pp; ++p)
              for (std::size_t q = 0; q < qq; ++q)
                c[p * qq + q] += factor * lp[p] * rp[q];
          }
        }
      }
    }
  }
}

#ifdef TA_STRIDED_DGEMM_COUNT
inline std::atomic<std::size_t> g_strided_dgemm_ce_ce_calls{0};
#endif

/// ce+ce strided-DGEMM core (inner CONTRACTION; ride right-external μ̃ into BLAS
/// M). ORIENTATION-AWARE (offsets derived from left_op/right_op of the OUTER
/// GemmHelper, exactly as arena_strided_dgemm_ce_e) and Hadamard-agnostic:
/// Hadamard is carried as nbatch by the einsum driver, so a thin nbatch loop
/// wraps a fixed-Hadamard ce+ce core and serves both hce+ce (nbatch>1) and the
/// no-Hadamard ce+ce (nbatch==1). Do NOT assert nbatch==1.
///
/// Outer GemmHelper mapping: Mo = left outer-external, No = right outer-external
/// = Mμ, Ko = outer-contracted = nK. The left-external `m` (Mo) is an OUTER loop
/// around the per-(b) body; R (a function of k,μ̃ only) is reused across m.
/// For each batch b, each left-external m, and each outer-contraction cell Κ=k:
///   C̃[m,μ̃, a_1] += factor * Σ_{a_4} R[k,μ̃](a_4) · L[m,k](a_1,a_4)
/// realized as ONE M=μ̃ × N=a_1 × K=a_4 DGEMM riding μ̃ into BLAS M via the
/// (empirically measured) inter-μ̃-cell slab stride (zero-copy), looping k with
/// beta=1. Mo==1 reduces to the original ce+ce kernel exactly. If a per-(b,m)
/// run is not clean, an inline per-cell GEMV fallback handles THAT (b,m) only
/// (each cell once -> no double-count). C must be pre-shaped (a_1-major); the
/// result outer is (m, μ̃) row-major (left-then-right concatenation, matching
/// make_result_range). Accumulates into C (beta=1).
template <typename ResultOuter, typename LeftOuter, typename RightOuter>
void arena_strided_dgemm_ce_ce(ResultOuter& C, const LeftOuter& L,
                               const RightOuter& R, std::size_t Mo,
                               std::size_t No, std::size_t Ko,
                               math::blas::Op left_op, math::blas::Op right_op,
                               double factor) {
  namespace blas = TiledArray::math::blas;
  using integer = blas::integer;
  static_assert(is_tensor_view_v<typename ResultOuter::value_type> &&
                    is_tensor_view_v<typename LeftOuter::value_type> &&
                    is_tensor_view_v<typename RightOuter::value_type>,
                "arena_strided_dgemm_ce_ce: arena (view) inner cells only");
  static_assert(
      std::is_same_v<typename ResultOuter::value_type::numeric_type, double> &&
          std::is_same_v<typename LeftOuter::value_type::numeric_type, double> &&
          std::is_same_v<typename RightOuter::value_type::numeric_type, double>,
      "arena_strided_dgemm_ce_ce: double inner storage only");
  const std::size_t Mmu = No;  // right outer-external rides BLAS M
  const std::size_t nK = Ko;   // outer-contracted is looped with beta=1
  const std::size_t nbatch = static_cast<std::size_t>(C.nbatch());
  if (nbatch == 0 || Mmu == 0 || nK == 0 || Mo == 0) return;
  // structural + self-defense (a mis-gated shape falls back / no-ops, never
  // miscomputes). The left-external Mo>=1 is supported (rides as an outer loop).
  const bool shape_ok =
      (C.range().volume() == Mo * Mmu && L.range().volume() == Mo * nK &&
       R.range().volume() == Mmu * nK &&
       static_cast<std::size_t>(L.nbatch()) == nbatch &&
       static_cast<std::size_t>(R.nbatch()) == nbatch);
  // If the structural invariant is violated, do nothing rather than form
  // out-of-bounds cell offsets below. This is only reachable via a mis-gate.
  TA_ASSERT(shape_ok);
  if (!shape_ok) return;
  // orientation-aware outer offsets (mirror arena_strided_dgemm_ce_e a_off/b_off)
  const std::size_t ldb_o = (right_op == blas::NoTranspose) ? No : Ko;
  auto r_off = [&](std::size_t k, std::size_t mu) {
    return (right_op == blas::NoTranspose) ? k * ldb_o + mu : mu * ldb_o + k;
  };
  // L: 2-D (m,k) offset (orientation-aware). l_off(k)==k only held for Mo==1.
  const std::size_t lda_o = (left_op == blas::NoTranspose) ? Ko : Mo;
  auto l_off = [&](std::size_t m, std::size_t k) {
    return (left_op == blas::NoTranspose) ? m * lda_o + k : k * lda_o + m;
  };
  // result outer (Mo x Mμ) row-major: (m, μ̃) = m*Mmu + mu.
  auto c_off = [&](std::size_t m, std::size_t mu) { return m * Mmu + mu; };
  const auto* lc = L.data();
  const auto* rc = R.data();
  auto* cc = C.data();
  for (std::size_t b = 0; b < nbatch; ++b) {
    const std::size_t cbase = b * Mo * Mmu;
    const std::size_t rbase = b * Mmu * nK;
    const std::size_t lbase = b * Mo * nK;
    for (std::size_t m = 0; m < Mo; ++m) {
      const auto& c0 = cc[cbase + c_off(m, 0)];
      long P = c0 ? static_cast<long>(c0.size()) : -1;
      bool clean = (P > 0);
      // C μ̃-run: uniform size P, constant stride sC>=P (page-jump guard).
      // Presence-first: verify every μ̃ cell present + uniform size BEFORE
      // probing the cell-0->1 stride via pointer subtraction.
      long sC = P;
      if (clean && Mmu > 1) {
        for (std::size_t mu = 0; clean && mu < Mmu; ++mu) {
          const auto& cmu = cc[cbase + c_off(m, mu)];
          if (!cmu || static_cast<long>(cmu.size()) != P) clean = false;
        }
        if (clean) {
          sC = static_cast<long>(cc[cbase + c_off(m, 1)].data() - c0.data());
          if (sC < P) clean = false;
          for (std::size_t mu = 0; clean && mu < Mmu; ++mu) {
            if (cc[cbase + c_off(m, mu)].data() !=
                c0.data() + static_cast<std::ptrdiff_t>(mu) * sC)
              clean = false;
          }
        }
      }
      // Q from R[k=0, μ̃=0]; L_{m,k} size P*Q; R μ̃-run per k uniform Q at
      // constant stride sR>=Q (uniform across k) -- page-jump guard on R too.
      const auto* r00 = clean ? &rc[rbase + r_off(0, 0)] : nullptr;
      long Q = (r00 && *r00) ? static_cast<long>(r00->size()) : -1;
      if (Q <= 0) clean = false;
      long sR = Q;
      for (std::size_t k = 0; clean && k < nK; ++k) {
        const auto& lk = lc[lbase + l_off(m, k)];
        if (!lk || static_cast<long>(lk.size()) != P * Q) {
          clean = false;
          break;
        }
        const auto& rk0 = rc[rbase + r_off(k, 0)];
        if (!rk0 || static_cast<long>(rk0.size()) != Q) {
          clean = false;
          break;
        }
        if (Mmu > 1) {
          for (std::size_t mu = 0; clean && mu < Mmu; ++mu) {
            const auto& rmu = rc[rbase + r_off(k, mu)];
            if (!rmu || static_cast<long>(rmu.size()) != Q) clean = false;
          }
          if (!clean) break;
          const long sRk =
              static_cast<long>(rc[rbase + r_off(k, 1)].data() - rk0.data());
          if (sRk < Q) {
            clean = false;
            break;
          }
          if (k == 0) sR = sRk;
          else if (sRk != sR) {
            clean = false;
            break;
          }
          for (std::size_t mu = 0; clean && mu < Mmu; ++mu) {
            if (rc[rbase + r_off(k, mu)].data() !=
                rk0.data() + static_cast<std::ptrdiff_t>(mu) * sR)
              clean = false;
          }
        }
      }
      if (clean) {
        double* Cd = cc[cbase + c_off(m, 0)].data();  // μ̃-run base, stride sC
        for (std::size_t k = 0; k < nK; ++k) {
          const double* Rk =
              rc[rbase + r_off(k, 0)].data();  // μ̃-run base for k, stride sR
          const double* Lk = lc[lbase + l_off(m, k)].data();  // P x Q row-major
          // C(Mμ x P) += factor * R̃(Mμ x Q) · L(P x Q)^T ; contract a_4(=Q)
          blas::gemm(blas::NoTranspose, blas::Transpose,
                     /*M=*/static_cast<integer>(Mmu),
                     /*N=*/static_cast<integer>(P),
                     /*K=*/static_cast<integer>(Q), factor,
                     /*A=*/Rk, /*lda=*/static_cast<integer>(sR),
                     /*B=*/Lk, /*ldb=*/static_cast<integer>(Q),
                     /*beta=*/1.0,
                     /*C=*/Cd, /*ldc=*/static_cast<integer>(sC));
#ifdef TA_STRIDED_DGEMM_COUNT
          g_strided_dgemm_ce_ce_calls.fetch_add(1, std::memory_order_relaxed);
#endif
        }
      } else {
        // inline per-cell GEMV fallback for THIS (b,m) (each cell once)
        for (std::size_t mu = 0; mu < Mmu; ++mu) {
          auto& Cc = cc[cbase + c_off(m, mu)];
          if (!Cc) continue;
          const long Pl = static_cast<long>(Cc.size());
          double* c = Cc.data();
          for (std::size_t k = 0; k < nK; ++k) {
            const auto& lk = lc[lbase + l_off(m, k)];
            const auto& rk = rc[rbase + r_off(k, mu)];
            if (!lk || !rk) continue;
            const long Ql = static_cast<long>(rk.size());
            if (Ql == 0 || static_cast<long>(lk.size()) != Pl * Ql) continue;
            const double* l = lk.data();   // Pl x Ql row-major
            const double* rr = rk.data();  // Ql
            for (long a1 = 0; a1 < Pl; ++a1) {
              double acc = 0;
              const double* lr = l + a1 * Ql;
              for (long a4 = 0; a4 < Ql; ++a4) acc += lr[a4] * rr[a4];
              c[a1] += factor * acc;
            }
          }
        }
      }
    }
  }
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

/// Hadamard-outer, phantom-unit-denest-inner ToT x ToT product into a fresh
/// arena tile. Like arena_hadamard_inner_contract, but each result outer cell
/// is a unit-extent [1]^phantom_rank cell (the inner product is a full
/// contraction = a flat dot; there are no real result inner modes). `cell_op`
/// (the phantom-dot per-cell op) fills the pre-shaped unit cell. No inner
/// permutation: phantom modes are all unit-extent.
template <typename Result, typename Left, typename Right, typename CellOp>
Result arena_hadamard_phantom_dot(const Left& left, const Right& right,
                                  std::size_t phantom_rank,
                                  const CellOp& cell_op) {
  using inner_range_t = typename Result::value_type::range_type;
  TA_ASSERT(left.range().volume() == right.range().volume());
  TA_ASSERT(left.nbatch() == right.nbatch());
  const std::size_t N_cells = left.range().volume() * left.nbatch();
  const container::vector<std::size_t> unit_ext(phantom_rank, std::size_t{1});
  auto range_fn = [&left, &right, &unit_ext](std::size_t ord) -> inner_range_t {
    const auto& lc = left.data()[ord];
    const auto& rc = right.data()[ord];
    if (lc.empty() || rc.empty()) return inner_range_t{};
    return inner_range_t(unit_ext);
  };
  Result result =
      arena_outer_init<Result>(left.range(), left.nbatch(), range_fn);
  for (std::size_t ord = 0; ord < N_cells; ++ord) {
    if (result.data()[ord].empty()) continue;
    cell_op(result.data()[ord], left.data()[ord], right.data()[ord]);
  }
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
  scale_left,   // ToT × plain T → ToT (right operand contributes scalars)
  scale_right,  // plain T × ToT → ToT (left operand contributes scalars)
  phantom_dot   // full inner contraction (dot) into a unit-extent result cell;
                // the result keeps only phantom-unit inner modes (see
                // is_phantom_unit_label). Operand cells are read flat, so no
                // operand carries the phantom mode and no GEMM rank match is
  // required. Realizes the ToT×ToT→plain-T (DeNest) inner product.
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

  // For kind == phantom_dot: the number of phantom-unit result modes (the rank
  // of the unit-extent result inner cell, e.g. 1 for `⊗₁`).
  std::size_t phantom_rank = 0;

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
      case RegimeAInnerKind::phantom_dot: {
        // The result keeps only phantom-unit modes: a rank-`phantom_rank`,
        // all-unit-extent cell (e.g. [1] for `⊗₁`).
        container::vector<std::size_t> ext(phantom_rank, std::size_t{1});
        return InnerRange(ext);
      }
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
      case RegimeAInnerKind::phantom_dot: {
        if constexpr (is_arena_inner_cell_v<LCell> &&
                      is_arena_inner_cell_v<RCell>) {
          if (l.empty() || rr.empty()) return;
          // Full inner contraction with only phantom-unit modes surviving: a
          // flat (non-conjugating) dot of the operand cells -- the same value a
          // GEMM with M=N=1,K=vol would compute -- accumulated into the lone
          // element of the unit-extent result cell. Reads operands flat, so no
          // operand need carry the phantom mode and no rank match is required;
          // uniform for TA::Tensor and ArenaTensor cells.
          using Numeric = typename std::remove_cv_t<ResultCell>::numeric_type;
          const std::size_t n = l.range().volume();
          TA_ASSERT(n == rr.range().volume());
          const auto* MADNESS_RESTRICT lp = l.data();
          const auto* MADNESS_RESTRICT rp = rr.data();
          Numeric acc{0};
          for (std::size_t j = 0; j < n; ++j) acc += lp[j] * rp[j];
          r.data()[0] += acc;
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
      } else if (bool(inner.C) && [&] {
                   for (const auto& lbl : inner.C)
                     if (!::TiledArray::detail::is_phantom_unit_label(lbl))
                       return false;
                   return true;
                 }()) {
        // Phantom-unit denest: every surviving result inner mode is a phantom
        // unit (⊗ₙ), i.e. the real inner modes are fully contracted. Realize
        // the inner product as a flat dot into a unit-extent result cell --
        // operands are read flat, so neither needs to carry the phantom mode
        // (no GEMM, no TensorContractionPlan rank match). See accumulate().
        plan.kind = RegimeAInnerKind::phantom_dot;
        plan.phantom_rank = inner.C.size();
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

/// Kill switch for the regime-A hc+e strided-DGEMM reuse path: when true,
/// run_regime_a_arena keeps the legacy per-cell accumulate. Test/bench hook
/// for the strided-vs-per-cell differential (correctness) and the perf
/// measurement; production default is false (strided on). Mirrors
/// arena_disabled().
inline bool& regime_a_strided_disabled() {
  static bool flag = false;
  return flag;
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
      // hc+e reuse gate: the result/operand inner cells must be the kernel's
      // (view + double) inner type; mirror arena_strided_dgemm_ce_e's
      // static_assert so non-view / non-double ToT keep the per-cell path.
      using LInnerT = typename ArrayA_t::value_type::value_type;
      using RInnerT = typename ArrayB_t::value_type::value_type;
      constexpr bool ce_e_kernel_ok =
          is_tensor_view_v<InnerT> && is_tensor_view_v<LInnerT> &&
          is_tensor_view_v<RInnerT> &&
          std::is_same_v<typename InnerT::numeric_type, double> &&
          std::is_same_v<typename LInnerT::numeric_type, double> &&
          std::is_same_v<typename RInnerT::numeric_type, double>;
      // Inner OUTER-PRODUCT (K_inner==0) is the strided-reusable shape; any
      // inner contraction (hc+ce) stays per-cell (two-level stride). The
      // runtime toggle lets tests/benches force the per-cell path.
      const bool hce_e_strided =
          plan.kind == RegimeAInnerKind::contraction && plan.c_plan &&
          plan.c_plan->gemm_helper.num_contract_ranks() == 0 &&
          !regime_a_strided_disabled();
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
        if constexpr (ce_e_kernel_ok) {
          if (hce_e_strided) {
            // hc+e: ride the within-tile contraction cells into BLAS K via the
            // landed ce+e core. M=N=1, K=vol; kernel nbatch == Hadamard batch.
            // cview shares tile's data_ (storage-aliasing reshape), so the
            // kernel's beta=1 writes accumulate into tile across the i-loop.
            namespace blas = TiledArray::math::blas;
            const std::size_t Kvol =
                static_cast<std::size_t>(trange.tile(i).volume());
            auto cview = tile.reshape(TiledArray::Range{1}, batch);
            arena_strided_dgemm_ce_e(cview, ai, bi, /*M=*/std::size_t{1},
                                     /*N=*/std::size_t{1}, /*K=*/Kvol,
                                     blas::NoTranspose, blas::NoTranspose,
                                     /*factor=*/1.0);
            continue;  // tile-i contribution complete
          }
        }
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
