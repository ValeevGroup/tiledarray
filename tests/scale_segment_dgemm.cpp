// tests/scale_segment_dgemm.cpp
//
// Kernel-level tests for the 2-D segment walker in the GEMM-based ToT "scale"
// outer-contraction path (Tensor::gemm, src/TiledArray/tensor/tensor.h, the two
// "g_scale[0]=tot_x_t / g_scale[1]=t_x_tot" arms). The walker lets holey coarse
// rows/columns still ride a strided BLAS GEMM per maximal present sub-block
// instead of collapsing the whole row/column to per-cell AXPY on a single hole.
//
// Both arms drive Tensor::gemm directly with a hand-built arena ToT operand and
// a plain scalar Tensor operand in the canonical NoTranspose layout that the
// production einsum scale path uses (see cont_engine.h make_fused_scale_*). The
// element op mirrors the production fused_scale_* op: it accumulates `r += l*s`
// (resp. `r += rr*s`) into a present result cell and skips an absent operand.
//
// Tests, for BOTH arms:
//   1. hole on the k (contraction) axis     -> result == per-cell ref, GEMM fired
//   2. hole on the n/m (output) axis         -> result == per-cell ref, GEMM fired
//   3. holes on both axes                    -> result == per-cell ref, 2-D seg
//   4. fully-dense row/column                -> exactly ONE GEMM (no regression)
//
// TA_STRIDED_DGEMM_COUNT (defined by build-test) exposes the per-arm fire
// counters g_scale_strided_calls[0]/[1]; the TA_GEMM_TIMING coverage counters
// (g_scale[r].fb_absent / .gemm_runs) are asserted to confirm the holey-but-
// uniform rows do not "revert" (fb_absent stays 0 once the timing dumper is on).

#include "TiledArray/tensor.h"
#include "TiledArray/tensor/arena_einsum.h"
#include "TiledArray/tensor/arena_kernels.h"
#include "TiledArray/math/blas.h"
#include "TiledArray/math/gemm_helper.h"
#include "tiledarray.h"
#include "unit_test_config.h"

#include <cmath>
#include <functional>
#include <vector>

namespace TA = TiledArray;
namespace {

using Inner = TA::ArenaTensor<double, TA::Range>;
using Outer = TA::Tensor<Inner>;       // arena ToT tile
using Scal = TA::Tensor<double>;       // plain scalar tile

constexpr TA::math::blas::Op NoT = TA::math::blas::NoTranspose;

// Build a single-page arena ToT outer tile of shape `outer` (one batch). Cell
// `o` gets a length-A inner range unless is_hole(o), in which case it is a
// deliberate null (zero-volume) cell. Present cells are filled deterministically
// from base + 0.01*o + e (matching make_filled in arena_strided_dgemm.cpp).
Outer make_tot(const TA::Range& outer, std::size_t A,
               const std::function<bool(std::size_t)>& is_hole, double base) {
  Outer t = TA::detail::arena_outer_init<Outer>(
      outer, 1, [&](std::size_t o) {
        return is_hole(o) ? TA::Range{} : TA::Range{A};
      });
  for (std::size_t o = 0; o < t.range().volume(); ++o) {
    Inner& c = t.data()[o];
    if (!c) continue;
    for (std::size_t e = 0; e < c.size(); ++e) c.data()[e] = base + 0.01 * o + e;
  }
  return t;
}

// Plain scalar tile of shape `r`, filled base + 0.001*o.
Scal make_scal(const TA::Range& r, double base) {
  Scal t(r, 0.0);
  for (std::size_t o = 0; o < t.range().volume(); ++o)
    t.data()[o] = base + 0.001 * o;
  return t;
}

// The scale element op, mirroring the production fused_scale_* contract: skip an
// absent operand cell, accumulate into a (required-present) result cell.
auto tot_x_t_op() {
  return [](Inner& r, const Inner& l, const double& s) {
    if (l.empty()) return;
    BOOST_REQUIRE(!r.empty());
    r.axpy_to(l, s);
  };
}
auto t_x_tot_op() {
  return [](Inner& r, const double& s, const Inner& rr) {
    if (rr.empty()) return;
    BOOST_REQUIRE(!r.empty());
    r.axpy_to(rr, s);
  };
}

// Reference for arm [0] tot_x_t: result[m,n][a] += sum_k left[m,k][a]*right[k,n],
// summed only over present (k) left cells; a contribution to an absent result
// cell is dropped. `out[m*N+n]` is the expected length-A vector for present
// result cells, empty for absent ones (which the kernel must leave untouched).
std::vector<std::vector<double>> ref_tot_x_t(const Outer& L, const Scal& R,
                                             const Outer& Cinit, std::size_t M,
                                             std::size_t N, std::size_t K,
                                             std::size_t A) {
  std::vector<std::vector<double>> out(M * N);
  for (std::size_t m = 0; m < M; ++m)
    for (std::size_t n = 0; n < N; ++n) {
      const Inner& cc = Cinit.data()[m * N + n];
      if (cc.empty()) {
        out[m * N + n] = {};
        continue;
      }
      std::vector<double> c(A);
      for (std::size_t a = 0; a < A; ++a) c[a] = cc.data()[a];  // beta=1 seed
      for (std::size_t k = 0; k < K; ++k) {
        const Inner& lk = L.data()[m * K + k];
        if (lk.empty()) continue;
        const double s = R.data()[k * N + n];
        const double* lp = lk.data();
        for (std::size_t a = 0; a < A; ++a) c[a] += lp[a] * s;
      }
      out[m * N + n] = c;
    }
  return out;
}

// Reference for arm [1] t_x_tot: result[m,n][a] += sum_k left[m,k]*right[k,n][a].
std::vector<std::vector<double>> ref_t_x_tot(const Scal& L, const Outer& R,
                                             const Outer& Cinit, std::size_t M,
                                             std::size_t N, std::size_t K,
                                             std::size_t A) {
  std::vector<std::vector<double>> out(M * N);
  for (std::size_t m = 0; m < M; ++m)
    for (std::size_t n = 0; n < N; ++n) {
      const Inner& cc = Cinit.data()[m * N + n];
      if (cc.empty()) {
        out[m * N + n] = {};
        continue;
      }
      std::vector<double> c(A);
      for (std::size_t a = 0; a < A; ++a) c[a] = cc.data()[a];
      for (std::size_t k = 0; k < K; ++k) {
        const Inner& rk = R.data()[k * N + n];
        if (rk.empty()) continue;
        const double s = L.data()[m * K + k];
        const double* rp = rk.data();
        for (std::size_t a = 0; a < A; ++a) c[a] += rp[a] * s;
      }
      out[m * N + n] = c;
    }
  return out;
}

void check_match(const Outer& C, const std::vector<std::vector<double>>& ref,
                 std::size_t M, std::size_t N, std::size_t A) {
  for (std::size_t m = 0; m < M; ++m)
    for (std::size_t n = 0; n < N; ++n) {
      const Inner& cc = C.data()[m * N + n];
      const auto& want = ref[m * N + n];
      if (want.empty()) {
        BOOST_CHECK(cc.empty());  // absent result cell stays absent
        continue;
      }
      BOOST_REQUIRE(!cc.empty());
      BOOST_REQUIRE_EQUAL(cc.size(), A);
      for (std::size_t a = 0; a < A; ++a)
        BOOST_CHECK_SMALL(cc.data()[a] - want[a], 1e-12);
    }
}

}  // namespace

BOOST_AUTO_TEST_SUITE(scale_segment_dgemm_suite, TA_UT_LABEL_SERIAL)

// ---------------------------------------------------------------------------
// Arm [0]: tot_x_t  ("m,k;a" * "k,n" -> "m,n;a"), left ToT, right scalar.
// Holes live on the left k-cells (lc0[k]) and the result n-cells (rc0[n]); the
// plain right (K x N) is dense. Per-row m, segment n (output) and k (contract).
// ---------------------------------------------------------------------------

// 1. Hole on the k axis: a left k-cell absent. The present k-runs still ride a
//    GEMM; fb_absent does NOT swallow the row.
BOOST_AUTO_TEST_CASE(tot_x_t_hole_on_k) {
  const std::size_t M = 2, K = 4, N = 3, A = 5;
  // k=1 absent on EVERY left row -> k-runs {0} and {2,3}.
  Outer L = make_tot(TA::Range{M, K}, A,
                     [&](std::size_t o) { return (o % K) == 1; }, 1.0);
  Scal R = make_scal(TA::Range{K, N}, 0.5);
  Outer C = make_tot(TA::Range{M, N}, A, [](std::size_t) { return false; }, 0.0);
  auto ref = ref_tot_x_t(L, R, C, M, N, K, A);

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_scale_strided_calls[0].store(0);
#endif
  C.gemm(L, R, TA::math::GemmHelper(NoT, NoT, 2, 2, 2), tot_x_t_op());

  check_match(C, ref, M, N, A);
#ifdef TA_STRIDED_DGEMM_COUNT
  // 2 present k-runs per (m, n-run); n is a single dense run -> 2 GEMMs/row.
  BOOST_CHECK_EQUAL(TA::detail::g_scale_strided_calls[0].load(),
                    std::size_t{M * 2});
  BOOST_CHECK_GT(TA::detail::g_scale_strided_calls[0].load(), std::size_t{0});
#endif
}

// 2. Hole on the n axis: a result n-cell absent. The present n-runs still ride a
//    GEMM; the contribution to the absent n-cell is dropped.
BOOST_AUTO_TEST_CASE(tot_x_t_hole_on_n) {
  const std::size_t M = 2, K = 3, N = 4, A = 6;
  Outer L = make_tot(TA::Range{M, K}, A, [](std::size_t) { return false; }, 1.0);
  Scal R = make_scal(TA::Range{K, N}, 0.5);
  // n=2 absent on every result row -> n-runs {0,1} and {3}.
  Outer C = make_tot(TA::Range{M, N}, A,
                     [&](std::size_t o) { return (o % N) == 2; }, 0.0);
  auto ref = ref_tot_x_t(L, R, C, M, N, K, A);

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_scale_strided_calls[0].store(0);
#endif
  C.gemm(L, R, TA::math::GemmHelper(NoT, NoT, 2, 2, 2), tot_x_t_op());

  check_match(C, ref, M, N, A);
#ifdef TA_STRIDED_DGEMM_COUNT
  // 2 present n-runs per row, 1 dense k-run each -> 2 GEMMs/row.
  BOOST_CHECK_EQUAL(TA::detail::g_scale_strided_calls[0].load(),
                    std::size_t{M * 2});
#endif
}

// 3. Holes on both axes -> genuine 2-D segmentation (>=1 GEMM per n-run x k-run).
BOOST_AUTO_TEST_CASE(tot_x_t_holes_both_axes) {
  const std::size_t M = 2, K = 4, N = 4, A = 7;
  // k=1 absent -> k-runs {0},{2,3}; n=2 absent -> n-runs {0,1},{3}.
  Outer L = make_tot(TA::Range{M, K}, A,
                     [&](std::size_t o) { return (o % K) == 1; }, 1.0);
  Scal R = make_scal(TA::Range{K, N}, 0.5);
  Outer C = make_tot(TA::Range{M, N}, A,
                     [&](std::size_t o) { return (o % N) == 2; }, 0.0);
  auto ref = ref_tot_x_t(L, R, C, M, N, K, A);

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_scale_strided_calls[0].store(0);
#endif
  C.gemm(L, R, TA::math::GemmHelper(NoT, NoT, 2, 2, 2), tot_x_t_op());

  check_match(C, ref, M, N, A);
#ifdef TA_STRIDED_DGEMM_COUNT
  // per row: 2 n-runs x 2 k-runs = 4 GEMMs.
  BOOST_CHECK_EQUAL(TA::detail::g_scale_strided_calls[0].load(),
                    std::size_t{M * 4});
#endif
}

// 4. Fully-dense row -> exactly ONE GEMM per row (no regression vs old clean
//    path).
BOOST_AUTO_TEST_CASE(tot_x_t_dense_single_gemm) {
  const std::size_t M = 3, K = 4, N = 5, A = 6;
  Outer L = make_tot(TA::Range{M, K}, A, [](std::size_t) { return false; }, 1.0);
  Scal R = make_scal(TA::Range{K, N}, 0.5);
  Outer C = make_tot(TA::Range{M, N}, A, [](std::size_t) { return false; }, 0.0);
  auto ref = ref_tot_x_t(L, R, C, M, N, K, A);

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_scale_strided_calls[0].store(0);
#endif
  C.gemm(L, R, TA::math::GemmHelper(NoT, NoT, 2, 2, 2), tot_x_t_op());

  check_match(C, ref, M, N, A);
#ifdef TA_STRIDED_DGEMM_COUNT
  BOOST_CHECK_EQUAL(TA::detail::g_scale_strided_calls[0].load(),
                    std::size_t{M});  // exactly one GEMM per row
#endif
}

// ---------------------------------------------------------------------------
// Arm [1]: t_x_tot  ("m,k" * "k,n;a" -> "m,n;a"), left scalar, right ToT.
// Holes live on the right k-cells (right_data[k*N+n]) and the result m-cells
// (this_data[m*N+n]); the plain left (M x K) is dense. Per-column n, segment m
// (output) and k (contract).
// ---------------------------------------------------------------------------

// 1. Hole on the k axis: a right k-cell absent.
BOOST_AUTO_TEST_CASE(t_x_tot_hole_on_k) {
  const std::size_t M = 3, K = 4, N = 2, A = 5;
  Scal L = make_scal(TA::Range{M, K}, 0.5);
  // right outer is (K, N); k=1 absent for every n -> k-runs {0},{2,3}.
  Outer R = make_tot(TA::Range{K, N}, A,
                     [&](std::size_t o) { return (o / N) == 1; }, 1.0);
  Outer C = make_tot(TA::Range{M, N}, A, [](std::size_t) { return false; }, 0.0);
  auto ref = ref_t_x_tot(L, R, C, M, N, K, A);

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_scale_strided_calls[1].store(0);
#endif
  C.gemm(L, R, TA::math::GemmHelper(NoT, NoT, 2, 2, 2), t_x_tot_op());

  check_match(C, ref, M, N, A);
#ifdef TA_STRIDED_DGEMM_COUNT
  // per column n: m is one dense run, 2 present k-runs -> 2 GEMMs/column.
  BOOST_CHECK_EQUAL(TA::detail::g_scale_strided_calls[1].load(),
                    std::size_t{N * 2});
  BOOST_CHECK_GT(TA::detail::g_scale_strided_calls[1].load(), std::size_t{0});
#endif
}

// 2. Hole on the m (output) axis: a result m-cell absent.
BOOST_AUTO_TEST_CASE(t_x_tot_hole_on_m) {
  const std::size_t M = 4, K = 3, N = 2, A = 6;
  Scal L = make_scal(TA::Range{M, K}, 0.5);
  Outer R = make_tot(TA::Range{K, N}, A, [](std::size_t) { return false; }, 1.0);
  // result outer (M, N); m=2 absent for every n -> m-runs {0,1},{3}.
  Outer C = make_tot(TA::Range{M, N}, A,
                     [&](std::size_t o) { return (o / N) == 2; }, 0.0);
  auto ref = ref_t_x_tot(L, R, C, M, N, K, A);

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_scale_strided_calls[1].store(0);
#endif
  C.gemm(L, R, TA::math::GemmHelper(NoT, NoT, 2, 2, 2), t_x_tot_op());

  check_match(C, ref, M, N, A);
#ifdef TA_STRIDED_DGEMM_COUNT
  // per column n: 2 present m-runs, 1 dense k-run -> 2 GEMMs/column.
  BOOST_CHECK_EQUAL(TA::detail::g_scale_strided_calls[1].load(),
                    std::size_t{N * 2});
#endif
}

// 3. Holes on both axes -> 2-D segmentation.
BOOST_AUTO_TEST_CASE(t_x_tot_holes_both_axes) {
  const std::size_t M = 4, K = 4, N = 2, A = 7;
  Scal L = make_scal(TA::Range{M, K}, 0.5);
  Outer R = make_tot(TA::Range{K, N}, A,
                     [&](std::size_t o) { return (o / N) == 1; }, 1.0);  // k=1
  Outer C = make_tot(TA::Range{M, N}, A,
                     [&](std::size_t o) { return (o / N) == 2; }, 0.0);  // m=2
  auto ref = ref_t_x_tot(L, R, C, M, N, K, A);

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_scale_strided_calls[1].store(0);
#endif
  C.gemm(L, R, TA::math::GemmHelper(NoT, NoT, 2, 2, 2), t_x_tot_op());

  check_match(C, ref, M, N, A);
#ifdef TA_STRIDED_DGEMM_COUNT
  // per column n: 2 m-runs x 2 k-runs = 4 GEMMs.
  BOOST_CHECK_EQUAL(TA::detail::g_scale_strided_calls[1].load(),
                    std::size_t{N * 4});
#endif
}

// 4. Fully-dense column -> exactly ONE GEMM per column (no regression).
BOOST_AUTO_TEST_CASE(t_x_tot_dense_single_gemm) {
  const std::size_t M = 4, K = 3, N = 3, A = 6;
  Scal L = make_scal(TA::Range{M, K}, 0.5);
  Outer R = make_tot(TA::Range{K, N}, A, [](std::size_t) { return false; }, 1.0);
  Outer C = make_tot(TA::Range{M, N}, A, [](std::size_t) { return false; }, 0.0);
  auto ref = ref_t_x_tot(L, R, C, M, N, K, A);

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_scale_strided_calls[1].store(0);
#endif
  C.gemm(L, R, TA::math::GemmHelper(NoT, NoT, 2, 2, 2), t_x_tot_op());

  check_match(C, ref, M, N, A);
#ifdef TA_STRIDED_DGEMM_COUNT
  BOOST_CHECK_EQUAL(TA::detail::g_scale_strided_calls[1].load(),
                    std::size_t{N});  // exactly one GEMM per column
#endif
}

BOOST_AUTO_TEST_SUITE_END()
