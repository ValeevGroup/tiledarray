// tests/arena_strided_dgemm.cpp
#include "TiledArray/tensor/arena_einsum.h"
#include "TiledArray/tensor/arena_kernels.h"
#include "TiledArray/tensor.h"
#include "TiledArray/math/blas.h"
#include "tiledarray.h"
#include "unit_test_config.h"
#include <functional>
#include <memory>
#include <vector>

namespace TA = TiledArray;
using Inner = TA::ArenaTensor<double, TA::Range>;
using Outer = TA::Tensor<Inner>;

namespace {
// Fabricate an arena ToT tile: outer range r, one batch, inner shape from
// shape_fn(ordinal); fill cell e of ordinal o with base + 0.01*o + e.
Outer make_filled(const TA::Range& r,
                  const std::function<TA::Range(std::size_t)>& shape_fn,
                  double base) {
  Outer t = TA::detail::arena_outer_init<Outer>(r, 1, shape_fn);
  for (std::size_t o = 0; o < t.range().volume(); ++o) {
    Inner& c = t.data()[o];
    if (!c) continue;
    for (std::size_t e = 0; e < c.size(); ++e) c.data()[e] = base + 0.01 * o + e;
  }
  return t;
}

std::vector<double> ref_ce_e(const Outer& L, const Outer& R, std::size_t m,
                             std::size_t n, std::size_t K, std::size_t P,
                             std::size_t Q, double factor) {
  std::vector<double> c(P * Q, 0.0);
  for (std::size_t k = 0; k < K; ++k) {
    const double* lp = L.data()[m * K + k].data();
    const double* rp = R.data()[n * K + k].data();
    for (std::size_t p = 0; p < P; ++p)
      for (std::size_t q = 0; q < Q; ++q) c[p * Q + q] += factor * lp[p] * rp[q];
  }
  return c;
}
}  // namespace

BOOST_AUTO_TEST_SUITE(arena_strided_dgemm_suite, TA_UT_LABEL_SERIAL)

// FACT A: uniform cells -> single constant inter-cell stride (>= cell size).
BOOST_AUTO_TEST_CASE(fact_uniform_constant_stride) {
  const std::size_t A = 8;
  Outer t = make_filled(TA::Range{5}, [&](std::size_t) { return TA::Range{A}; }, 1.0);
  const std::ptrdiff_t s = t.data()[1].data() - t.data()[0].data();
  BOOST_CHECK_GE(s, static_cast<std::ptrdiff_t>(A));
  for (std::size_t k = 0; k < 5; ++k)
    BOOST_CHECK_EQUAL(t.data()[k].data(), t.data()[0].data() + k * s);
}

// FACT B: ragged sizes whose padded cell allocations land in the SAME stride
// bucket alias to the SAME inter-cell stride, so a stride-only fusability check
// is unsafe -> the kernel guard must ALSO check size(). Each cell consumes
// arena_align_up(cell_size(vol), kArenaCachelineAlign) bytes; the per-element
// step (sizeof(double)=8B) is far smaller than the 128B bucket, so most
// adjacent volumes share a bucket -- we search for one such pair rather than
// hardcoding sizes (a hardcoded pair like 8/9 can straddle a bucket boundary
// and NOT alias, which says nothing about the hazard the kernel must guard).
BOOST_AUTO_TEST_CASE(fact_padding_aliases_stride) {
  auto padded = [](std::size_t n) {
    return TA::detail::arena_align_up(Inner::cell_size(n),
                                      TA::detail::kArenaCachelineAlign);
  };
  // smallest n>0 whose padded allocation equals that of n+1 (same bucket)
  std::size_t n = 1;
  while (n < 4096 && padded(n) != padded(n + 1)) ++n;
  BOOST_REQUIRE_LT(n, 4096u);  // such an aliasing pair must exist
  // Cells [n, n, n+1]: padded(n)==padded(n+1) makes all three sit at one
  // constant stride, yet cell 2's size() differs -> stride-only check fooled.
  Outer t = make_filled(TA::Range{3}, [&](std::size_t o) {
    return TA::Range{o < 2 ? n : n + 1};
  }, 1.0);
  const std::ptrdiff_t s01 = t.data()[1].data() - t.data()[0].data();
  const std::ptrdiff_t s12 = t.data()[2].data() - t.data()[1].data();
  BOOST_CHECK_EQUAL(s01, s12);                               // strides alias...
  BOOST_CHECK_NE(t.data()[1].size(), t.data()[2].size());    // ...but sizes differ
}

BOOST_AUTO_TEST_CASE(ce_e_matches_reference) {
  namespace blas = TA::math::blas;
  const std::size_t M = 2, N = 2, K = 3, P = 4, Q = 5;
  Outer L = make_filled(TA::Range{M, K}, [&](std::size_t){return TA::Range{P};}, 1.0);
  Outer R = make_filled(TA::Range{N, K}, [&](std::size_t){return TA::Range{Q};}, 2.0);
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{M, N}, 1, [&](std::size_t){return TA::Range{P, Q};});  // zero-init
  TA::detail::arena_strided_dgemm_ce_e(C, L, R, M, N, K, blas::NoTranspose,
                                       blas::Transpose, /*factor=*/1.0);
  for (std::size_t m = 0; m < M; ++m)
    for (std::size_t n = 0; n < N; ++n) {
      auto ref = ref_ce_e(L, R, m, n, K, P, Q, 1.0);
      const double* got = C.data()[m * N + n].data();
      for (std::size_t e = 0; e < P * Q; ++e) BOOST_CHECK_CLOSE(got[e], ref[e], 1e-12);
    }
}

BOOST_AUTO_TEST_CASE(ce_e_ragged_cell_still_correct_inline) {
  namespace blas = TA::math::blas;
  const std::size_t M = 2, N = 1, K = 2, Q = 3;
  // m=0: clean P=3 across k; m=1: ragged P (3 then 4) -> cell (1,0) falls back.
  Outer L = TA::detail::arena_outer_init<Outer>(
      TA::Range{M, K}, 1, [&](std::size_t o){
        return (o < K) ? TA::Range{3} : TA::Range{3 + (o % 2)}; });
  for (std::size_t o = 0; o < L.range().volume(); ++o)
    for (std::size_t e = 0; e < L.data()[o].size(); ++e) L.data()[o].data()[e] = 1.0 + e;
  Outer R = make_filled(TA::Range{N, K}, [&](std::size_t){return TA::Range{Q};}, 2.0);
  // result cell sizes: (0,0) -> 3*Q ; (1,0) -> only well-defined when P uniform;
  // for the ragged row use P from k=0 (=3) so the fallback's pp*qq guard holds.
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{M, N}, 1, [&](std::size_t o){ return TA::Range{3, Q}; });
  TA::detail::arena_strided_dgemm_ce_e(C, L, R, M, N, K, blas::NoTranspose,
                                       blas::Transpose, 1.0);
  // (0,0): clean GEMM path
  {
    auto ref = ref_ce_e(L, R, 0, 0, K, 3, Q, 1.0);
    const double* got = C.data()[0].data();
    for (std::size_t e = 0; e < 3 * Q; ++e) BOOST_CHECK_CLOSE(got[e], ref[e], 1e-12);
  }
  // (1,0): fallback path; only the k whose P==3 contributes under the guard.
  // Reference: same formula, restricted to k with L cell size == 3.
  {
    std::vector<double> ref(3 * Q, 0.0);
    for (std::size_t k = 0; k < K; ++k) {
      const auto& lk = L.data()[1 * K + k];
      if (lk.size() != 3) continue;
      const double* lp = lk.data();
      const double* rp = R.data()[0 * K + k].data();
      for (std::size_t p = 0; p < 3; ++p)
        for (std::size_t q = 0; q < Q; ++q) ref[p * Q + q] += lp[p] * rp[q];
    }
    const double* got = C.data()[1].data();
    for (std::size_t e = 0; e < 3 * Q; ++e) BOOST_CHECK_CLOSE(got[e], ref[e], 1e-12);
  }
}

BOOST_AUTO_TEST_CASE(ce_e_applies_factor) {
  namespace blas = TA::math::blas;
  const std::size_t M = 1, N = 1, K = 4, P = 3, Q = 3;
  Outer L = make_filled(TA::Range{M, K}, [&](std::size_t){return TA::Range{P};}, 1.0);
  Outer R = make_filled(TA::Range{N, K}, [&](std::size_t){return TA::Range{Q};}, 2.0);
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{M, N}, 1, [&](std::size_t){return TA::Range{P, Q};});
  TA::detail::arena_strided_dgemm_ce_e(C, L, R, M, N, K, blas::NoTranspose,
                                       blas::Transpose, 0.5);
  auto ref = ref_ce_e(L, R, 0, 0, K, P, Q, 0.5);
  const double* got = C.data()[0].data();
  for (std::size_t e = 0; e < P * Q; ++e) BOOST_CHECK_CLOSE(got[e], ref[e], 1e-12);
}

// ce+e core looped over Hadamard-folded nbatch: each batch b independently
// computes C_b[m,n](p,q) += sum_k L_b[m,k](p) * R_b[k,n](q).
BOOST_AUTO_TEST_CASE(ce_e_multi_batch) {
  namespace blas = TiledArray::math::blas;
  const std::size_t M = 2, N = 2, K = 3, P = 3, Q = 4, NB = 2;
  // L outer (M,K) inner {P}; R outer (K,N) inner {Q}; C outer (M,N) inner {P,Q}.
  Outer L = TA::detail::arena_outer_init<Outer>(
      TA::Range{M, K}, NB, [&](std::size_t){ return TA::Range{P}; });
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{K, N}, NB, [&](std::size_t){ return TA::Range{Q}; });
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{M, N}, NB, [&](std::size_t){ return TA::Range{P, Q}; });
  for (std::size_t o = 0; o < NB * M * K; ++o)
    for (std::size_t e = 0; e < L.data()[o].size(); ++e)
      L.data()[o].data()[e] = 1.0 + 0.01 * o + e;
  for (std::size_t o = 0; o < NB * K * N; ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  // M=left-ext, N=right-ext, K=contracted; canonical orientation.
  TA::detail::arena_strided_dgemm_ce_e(C, L, R, M, N, K,
                                       blas::NoTranspose, blas::NoTranspose, 1.0);
  for (std::size_t b = 0; b < NB; ++b)
    for (std::size_t m = 0; m < M; ++m)
      for (std::size_t n = 0; n < N; ++n) {
        std::vector<double> ref(P * Q, 0.0);
        for (std::size_t k = 0; k < K; ++k) {
          const double* l = L.data()[b * M * K + m * K + k].data();  // P
          const double* r = R.data()[b * K * N + k * N + n].data();  // Q
          for (std::size_t p = 0; p < P; ++p)
            for (std::size_t q = 0; q < Q; ++q) ref[p * Q + q] += l[p] * r[q];
        }
        const double* got = C.data()[b * M * N + m * N + n].data();
        for (std::size_t e = 0; e < P * Q; ++e)
          BOOST_CHECK_CLOSE(got[e], ref[e], 1e-12);
      }
}

// Addition A: multi-external inner indices on BOTH operands flatten into the
// inner outer-product. Left inner {a1,a2} (P=a1*a2), right inner {a3,a4}
// (Q=a3*a4), result inner {a1,a2,a3,a4} (P*Q). Single batch. Independent
// outer-product reference; under TA_STRIDED_DGEMM_COUNT the per-cell DGEMM
// fires once per result cell (= M*N).
BOOST_AUTO_TEST_CASE(ce_e_multi_external_inner) {
  namespace blas = TiledArray::math::blas;
  const std::size_t M = 2, N = 2, K = 3;
  const std::size_t a1 = 2, a2 = 3, a3 = 2, a4 = 2;
  const std::size_t P = a1 * a2, Q = a3 * a4;
  Outer L = make_filled(TA::Range{M, K},
                        [&](std::size_t){ return TA::Range{a1, a2}; }, 1.0);
  Outer R = make_filled(TA::Range{N, K},
                        [&](std::size_t){ return TA::Range{a3, a4}; }, 2.0);
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{M, N}, 1, [&](std::size_t){ return TA::Range{a1, a2, a3, a4}; });
#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_e_calls.store(0);
#endif
  TA::detail::arena_strided_dgemm_ce_e(C, L, R, M, N, K, blas::NoTranspose,
                                       blas::Transpose, 1.0);
  for (std::size_t m = 0; m < M; ++m)
    for (std::size_t n = 0; n < N; ++n) {
      auto ref = ref_ce_e(L, R, m, n, K, P, Q, 1.0);  // flat P*Q
      const double* got = C.data()[m * N + n].data();
      for (std::size_t e = 0; e < P * Q; ++e)
        BOOST_CHECK_CLOSE(got[e], ref[e], 1e-12);
    }
#ifdef TA_STRIDED_DGEMM_COUNT
  BOOST_CHECK_EQUAL(TA::detail::g_strided_dgemm_ce_e_calls.load(), M * N);
#endif
}

#ifdef TA_STRIDED_DGEMM_COUNT
BOOST_AUTO_TEST_CASE(ce_e_fires_clean_path) {
  namespace blas = TiledArray::math::blas;
  const std::size_t M = 2, N = 2, K = 3, P = 3, Q = 4, NB = 2;
  // NB batches, all uniform sizes => every result cell clean => one DGEMM each.
  Outer L = TA::detail::arena_outer_init<Outer>(
      TA::Range{M, K}, NB, [&](std::size_t){ return TA::Range{P}; });
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{K, N}, NB, [&](std::size_t){ return TA::Range{Q}; });
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{M, N}, NB, [&](std::size_t){ return TA::Range{P, Q}; });
  for (std::size_t o = 0; o < NB * M * K; ++o)
    for (std::size_t e = 0; e < L.data()[o].size(); ++e)
      L.data()[o].data()[e] = 1.0 + 0.01 * o + e;
  for (std::size_t o = 0; o < NB * K * N; ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  TA::detail::g_strided_dgemm_ce_e_calls.store(0);
  TA::detail::arena_strided_dgemm_ce_e(C, L, R, M, N, K, blas::NoTranspose,
                                       blas::NoTranspose, 1.0);
  // one DGEMM per result cell per batch
  BOOST_CHECK_EQUAL(TA::detail::g_strided_dgemm_ce_e_calls.load(), NB * M * N);
}
#endif

BOOST_AUTO_TEST_SUITE_END()
