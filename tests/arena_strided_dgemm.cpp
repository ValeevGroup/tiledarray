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

// C[mu](a1) = factor * sum_k sum_{a4} L[k](a1,a4) * R[mu,k](a4)
// L outer {nK} inner {P,Q}; R outer {Mmu,nK} (mu slow, k fast) inner {Q};
// C outer {Mmu} inner {P}.  (Mo==1 reference.)
std::vector<double> ref_ce_ce(const Outer& L, const Outer& R, std::size_t Mmu,
                              std::size_t nK, std::size_t P, std::size_t Q,
                              double factor) {
  std::vector<double> c(Mmu * P, 0.0);
  for (std::size_t k = 0; k < nK; ++k) {
    const double* l = L.data()[k].data();              // P x Q row-major
    for (std::size_t mu = 0; mu < Mmu; ++mu) {
      const double* r = R.data()[mu * nK + k].data();  // Q  (mu slow, k fast)
      for (std::size_t a1 = 0; a1 < P; ++a1) {
        double acc = 0;
        for (std::size_t a4 = 0; a4 < Q; ++a4) acc += l[a1 * Q + a4] * r[a4];
        c[mu * P + a1] += factor * acc;
      }
    }
  }
  return c;
}

// ---------------------------------------------------------------------------
// Sparsity-aware helpers for the per-k segmented strided-DGEMM tests (T1-T15).
//
// make_sparse: like make_filled, but a cell whose dense_shape(o) is selected by
// is_hole(o) is built from a zero-volume TA::Range{}, which arena_outer_init
// leaves NULL (a hole). Present cells get deterministic data. nbatch>=1.
Outer make_sparse(const TA::Range& outer_range, std::size_t nbatch,
                  const std::function<TA::Range(std::size_t)>& dense_shape,
                  const std::function<bool(std::size_t)>& is_hole, double base) {
  Outer t = TA::detail::arena_outer_init<Outer>(
      outer_range, nbatch,
      [&](std::size_t o) { return is_hole(o) ? TA::Range{} : dense_shape(o); });
  for (std::size_t o = 0; o < t.range().volume() * nbatch; ++o) {
    Inner& c = t.data()[o];
    if (!c) continue;
    for (std::size_t e = 0; e < c.size(); ++e) c.data()[e] = base + 0.01 * o + e;
  }
  return t;
}

// Sparsity-aware reference for arena_strided_dgemm_ce_ce_right in the SAME
// canonical convention as ref_ce_ce (L=strided P x Q matrix, R=single Q-vector),
// but generalized to Mo>=1 outer rows, NB batches, holes, and the left-inner
// transpose. Per the kernel: for result cell C[m,mu] (length P),
//   C[m,mu](a1) = factor * sum_k present L[m,k] (P x Q) * present R[mu,k] (Q),
// skipping any k where L[m,k] or R[mu,k] is absent or size-mismatched.
//   L outer (Mo x nK), canonical index b*Mo*nK + m*nK + k, inner {P,Q}
//   R outer (Mmu x nK), canonical (mu slow, k fast) b*Mmu*nK + mu*nK + k, inner {Q}
//   C outer (Mo x Mmu), index b*Mo*Mmu + m*Mmu + mu, inner {P}
// out[(b*Mo+m)*Mmu+mu] is the expected length-P vector (empty == expect absent).
// lt mirrors left_inner_transposed: lt=false L stored P x Q (l[a1*Q+a4]),
// lt=true L stored Q x P (l[a4*P+a1]).
std::vector<std::vector<double>> ref_ce_ce_right_sparse(
    const Outer& L, const Outer& R, std::size_t Mo, std::size_t Mmu,
    std::size_t nK, std::size_t P, double factor, std::size_t nbatch = 1,
    bool lt = false) {
  std::vector<std::vector<double>> out(nbatch * Mo * Mmu);
  for (std::size_t b = 0; b < nbatch; ++b)
    for (std::size_t m = 0; m < Mo; ++m)
      for (std::size_t mu = 0; mu < Mmu; ++mu) {
        std::vector<double> c(P, 0.0);
        bool any = false;
        for (std::size_t k = 0; k < nK; ++k) {
          const Inner& lk = L.data()[b * Mo * nK + m * nK + k];
          const Inner& rk = R.data()[b * Mmu * nK + mu * nK + k];
          if (!lk || !rk) continue;
          const std::size_t Q = rk.size();
          if (Q == 0 || lk.size() != P * Q) continue;
          const double* l = lk.data();
          const double* r = rk.data();
          for (std::size_t a1 = 0; a1 < P; ++a1) {
            double acc = 0.0;
            for (std::size_t a4 = 0; a4 < Q; ++a4)
              acc += (lt ? l[a4 * P + a1] : l[a1 * Q + a4]) * r[a4];
            c[a1] += factor * acc;
          }
          any = true;
        }
        out[(b * Mo + m) * Mmu + mu] = any ? c : std::vector<double>{};
      }
  return out;
}

// Sparsity-aware reference for arena_strided_dgemm_ce_ce_left. Here m (Mo) is
// the strided axis, n (No) the fixed result column; the RIGHT operand cell
// R[k,n] (the P x Q matrix) is the single non-strided operand and L[m,k] (the
// length-Q contraction vector) is the strided run. Per the kernel:
//   C[m,n](b1) = factor * sum_k present L[m,k] (Q) * present R[k,n] (Q x P),
// where result inner length P = b1. R canonical (a4,b1)=Q x P row-major
// (rt=false: r[a4*P+b1]); rt mirrors right_inner_transposed (P x Q, r[b1*Q+a4]).
//   L outer (Mo x nK), index b*Mo*nK + m*nK + k, inner {Q}
//   R outer (nK x No), canonical (k slow, n fast) b*nK*No + k*No + n, inner {Q,P}
//   C outer (Mo x No), index b*Mo*No + m*No + n, inner {P}
std::vector<std::vector<double>> ref_ce_ce_left_sparse(
    const Outer& L, const Outer& R, std::size_t Mo, std::size_t No,
    std::size_t nK, std::size_t P, double factor, std::size_t nbatch = 1,
    bool rt = false) {
  std::vector<std::vector<double>> out(nbatch * Mo * No);
  for (std::size_t b = 0; b < nbatch; ++b)
    for (std::size_t m = 0; m < Mo; ++m)
      for (std::size_t n = 0; n < No; ++n) {
        std::vector<double> c(P, 0.0);
        bool any = false;
        for (std::size_t k = 0; k < nK; ++k) {
          const Inner& lk = L.data()[b * Mo * nK + m * nK + k];
          const Inner& rk = R.data()[b * nK * No + k * No + n];
          if (!lk || !rk) continue;
          const std::size_t Q = lk.size();
          if (Q == 0 || rk.size() != P * Q) continue;
          const double* l = lk.data();
          const double* r = rk.data();
          for (std::size_t b1 = 0; b1 < P; ++b1) {
            double acc = 0.0;
            for (std::size_t a4 = 0; a4 < Q; ++a4)
              acc += l[a4] * (rt ? r[b1 * Q + a4] : r[a4 * P + b1]);
            c[b1] += factor * acc;
          }
          any = true;
        }
        out[(b * Mo + m) * No + n] = any ? c : std::vector<double>{};
      }
  return out;
}

// Assemble an Outer whose outer-cell views point at the cells of `src` in the
// order given by `phys[ord]` (assembled.data()[ord] aliases the SAME Cell as
// src.data()[phys[ord]]). Because ArenaTensor is a non-owning view, the
// assembled tile shares `src`'s arena slab without copying element data; the
// deleter keeps `src` alive. This is the only in-harness lever that yields a
// UNIFORM-SIZE run with a NON-CONSTANT inter-cell .data() stride.
Outer assemble_aliased(const Outer& src, const TA::Range& outer_range,
                       const std::vector<std::size_t>& phys) {
  const std::size_t n = outer_range.volume();
  TA_ASSERT(phys.size() == n);
  std::allocator<Inner> alloc;
  Inner* raw = alloc.allocate(n);
  for (std::size_t ord = 0; ord < n; ++ord)
    ::new (raw + ord) Inner(src.data()[phys[ord]]);  // shallow rebind
  auto deleter = [alloc, src, n](Inner* p) mutable {
    for (std::size_t i = 0; i < n; ++i) (p + i)->~Inner();
    alloc.deallocate(p, n);
    (void)src;
  };
  std::shared_ptr<Inner[]> data(raw, std::move(deleter));
  return Outer(outer_range, /*nbatch=*/1, std::move(data));
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

// Regime-A hc+e calling convention: M=N=1, K=vol, kernel-nbatch == Hadamard
// batch. Result tile is (Range{batch}, nbatch=1); presented to the kernel as
// (Range{1}, nbatch=batch) via a storage-aliasing reshape. Two operand
// "tiles" accumulate into the SAME result with beta=1 (cross-tile reduction).
BOOST_AUTO_TEST_CASE(regime_a_oprod_mapping_two_tiles) {
  namespace blas = TiledArray::math::blas;
  const std::size_t NB = 2;            // Hadamard-folded batch
  const std::size_t P = 3, Q = 4;      // inner outer-product extents (left, right)
  const std::size_t VOL0 = 3, VOL1 = 2;  // within-tile contraction cells per tile

  // Result: outer Range{NB}, single batch; each cell inner {P,Q}, zero-filled.
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{NB}, /*batch=*/1, [&](std::size_t) { return TA::Range{P, Q}; });
  // Operand tile t: outer Range{VOL_t}, nbatch=NB; left cells {P}, right {Q}.
  auto make_L = [&](std::size_t VOL, double seed) {
    Outer L = TA::detail::arena_outer_init<Outer>(
        TA::Range{VOL}, NB, [&](std::size_t) { return TA::Range{P}; });
    for (std::size_t o = 0; o < NB * VOL; ++o)
      for (std::size_t e = 0; e < L.data()[o].size(); ++e)
        L.data()[o].data()[e] = seed + 0.01 * o + e;
    return L;
  };
  auto make_R = [&](std::size_t VOL, double seed) {
    Outer R = TA::detail::arena_outer_init<Outer>(
        TA::Range{VOL}, NB, [&](std::size_t) { return TA::Range{Q}; });
    for (std::size_t o = 0; o < NB * VOL; ++o)
      for (std::size_t e = 0; e < R.data()[o].size(); ++e)
        R.data()[o].data()[e] = seed + 0.02 * o + e;
    return R;
  };
  Outer L0 = make_L(VOL0, 1.0), R0 = make_R(VOL0, 2.0);
  Outer L1 = make_L(VOL1, 5.0), R1 = make_R(VOL1, 7.0);

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_e_calls.store(0);
#endif
  // Regime-A call: present C as (Range{1}, nbatch=NB), M=N=1, K=vol, per tile.
  auto cview = C.reshape(TA::Range{1}, NB);
  TA::detail::arena_strided_dgemm_ce_e(cview, L0, R0, /*M=*/std::size_t{1},
                                       /*N=*/std::size_t{1}, /*K=*/VOL0,
                                       blas::NoTranspose, blas::NoTranspose, 1.0);
  auto cview2 = C.reshape(TA::Range{1}, NB);
  TA::detail::arena_strided_dgemm_ce_e(cview2, L1, R1, /*M=*/std::size_t{1},
                                       /*N=*/std::size_t{1}, /*K=*/VOL1,
                                       blas::NoTranspose, blas::NoTranspose, 1.0);

  // Reference: for each batch b, sum the rank-1 outer products over BOTH tiles.
  for (std::size_t b = 0; b < NB; ++b) {
    std::vector<double> ref(P * Q, 0.0);
    auto add_tile = [&](const Outer& L, const Outer& R, std::size_t VOL) {
      for (std::size_t k = 0; k < VOL; ++k) {
        const double* l = L.data()[b * VOL + k].data();  // P
        const double* r = R.data()[b * VOL + k].data();  // Q
        for (std::size_t p = 0; p < P; ++p)
          for (std::size_t q = 0; q < Q; ++q) ref[p * Q + q] += l[p] * r[q];
      }
    };
    add_tile(L0, R0, VOL0);
    add_tile(L1, R1, VOL1);
    const double* got = C.data()[b].data();  // C still (Range{NB}, nbatch=1)
    for (std::size_t e = 0; e < P * Q; ++e)
      BOOST_CHECK_CLOSE(got[e], ref[e], 1e-12);
  }
#ifdef TA_STRIDED_DGEMM_COUNT
  // clean operands => one DGEMM per (tile, batch); both tiles clean.
  BOOST_CHECK_EQUAL(TA::detail::g_strided_dgemm_ce_e_calls.load(),
                    std::size_t{2} * NB);
#endif
}

// Task 3.2 (tile-level fallback): regime-A hc+e convention (M=N=1, K=vol>1)
// with a RAGGED left operand so the kernel's clean-check rejects and the
// inline per-k fallback runs for the (single) result cell.
//
// WHY TILE-LEVEL: the einsum driver lays arena cells out as clean contiguous
// slabs (uniform-size, constant-stride), so e2e the kernel's clean-check
// always passes and the inline fallback is not reliably reachable from the
// einsum entry point. We therefore exercise the fallback directly at the
// kernel call, modeled on ce_e_ragged_cell_still_correct_inline, but in the
// regime-A reshape(Range{1}, NB) / M=N=1 / K=vol presentation from R1.
//
// The reference uses the SAME guard the kernel uses: only k-cells whose LEFT
// inner size equals the result-cell P (==P0, the k=0 size) contribute; the
// ragged k-cell is skipped. (See ce_e_ragged_cell_still_correct_inline.)
BOOST_AUTO_TEST_CASE(regime_a_oprod_mapping_scattered_falls_back) {
  namespace blas = TiledArray::math::blas;
  const std::size_t NB = 2;            // Hadamard-folded batch
  const std::size_t P0 = 3, Q = 4;     // left P (from k=0), right Q
  const std::size_t VOL = 3;           // within-tile contraction cells (K)
  // Left operand: outer Range{VOL}, nbatch=NB, inner {P0} EXCEPT one ragged
  // k-cell (k==1) in EVERY batch gets inner {P0+1} -> kernel's clean-check
  // rejects (non-uniform left size) and falls back to inline per-k for the
  // result cell. (The ragged cell is then skipped under the size guard.)
  Outer L = TA::detail::arena_outer_init<Outer>(
      TA::Range{VOL}, NB, [&](std::size_t ord) {
        const std::size_t k = ord % VOL;  // ord runs batch-major over (b,k)
        return TA::Range{k == 1 ? P0 + 1 : P0};
      });
  for (std::size_t o = 0; o < NB * VOL; ++o)
    for (std::size_t e = 0; e < L.data()[o].size(); ++e)
      L.data()[o].data()[e] = 1.0 + 0.01 * o + e;
  // Right operand: clean, inner {Q}.
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{VOL}, NB, [&](std::size_t) { return TA::Range{Q}; });
  for (std::size_t o = 0; o < NB * VOL; ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.02 * o + e;
  // Result: outer Range{NB}, single batch; each cell inner {P0,Q}, zero-init.
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{NB}, /*batch=*/1, [&](std::size_t) { return TA::Range{P0, Q}; });

  auto cview = C.reshape(TA::Range{1}, NB);
  TA::detail::arena_strided_dgemm_ce_e(cview, L, R, /*M=*/std::size_t{1},
                                       /*N=*/std::size_t{1}, /*K=*/VOL,
                                       blas::NoTranspose, blas::NoTranspose, 1.0);

  // Reference: per batch b, sum rank-1 outer products over k, but ONLY for k
  // whose left inner size == P0 (the kernel's fallback guard skips the ragged k).
  for (std::size_t b = 0; b < NB; ++b) {
    std::vector<double> ref(P0 * Q, 0.0);
    for (std::size_t k = 0; k < VOL; ++k) {
      const auto& lk = L.data()[b * VOL + k];
      if (lk.size() != P0) continue;  // guard: ragged k-cell does not contribute
      const double* l = lk.data();  // P0
      const double* r = R.data()[b * VOL + k].data();  // Q
      for (std::size_t p = 0; p < P0; ++p)
        for (std::size_t q = 0; q < Q; ++q) ref[p * Q + q] += l[p] * r[q];
    }
    const double* got = C.data()[b].data();
    for (std::size_t e = 0; e < P0 * Q; ++e)
      BOOST_CHECK_CLOSE(got[e], ref[e], 1e-12);
  }
  // Counter intentionally NOT asserted: the ragged cell takes the inline
  // fallback (no clean DGEMM), so firing is not the property under test --
  // correctness of the fallback is.
}

// Task 3.3 Step 2a: VOL=1 edge -> K=1, a single rank-1 outer product per batch.
// One operand tile (one strided call). Hand reference + (count build) exactly
// NB DGEMMs (one per batch, one tile, clean).
BOOST_AUTO_TEST_CASE(regime_a_oprod_mapping_vol1) {
  namespace blas = TiledArray::math::blas;
  const std::size_t NB = 2;
  const std::size_t P = 3, Q = 4;
  const std::size_t VOL = 1;  // single contraction cell -> K=1

  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{NB}, /*batch=*/1, [&](std::size_t) { return TA::Range{P, Q}; });
  Outer L = TA::detail::arena_outer_init<Outer>(
      TA::Range{VOL}, NB, [&](std::size_t) { return TA::Range{P}; });
  for (std::size_t o = 0; o < NB * VOL; ++o)
    for (std::size_t e = 0; e < L.data()[o].size(); ++e)
      L.data()[o].data()[e] = 1.0 + 0.01 * o + e;
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{VOL}, NB, [&](std::size_t) { return TA::Range{Q}; });
  for (std::size_t o = 0; o < NB * VOL; ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.02 * o + e;

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_e_calls.store(0);
#endif
  auto cview = C.reshape(TA::Range{1}, NB);
  TA::detail::arena_strided_dgemm_ce_e(cview, L, R, /*M=*/std::size_t{1},
                                       /*N=*/std::size_t{1}, /*K=*/VOL,
                                       blas::NoTranspose, blas::NoTranspose, 1.0);
  for (std::size_t b = 0; b < NB; ++b) {
    std::vector<double> ref(P * Q, 0.0);
    const double* l = L.data()[b * VOL + 0].data();  // P
    const double* r = R.data()[b * VOL + 0].data();  // Q
    for (std::size_t p = 0; p < P; ++p)
      for (std::size_t q = 0; q < Q; ++q) ref[p * Q + q] += l[p] * r[q];
    const double* got = C.data()[b].data();
    for (std::size_t e = 0; e < P * Q; ++e)
      BOOST_CHECK_CLOSE(got[e], ref[e], 1e-12);
  }
#ifdef TA_STRIDED_DGEMM_COUNT
  // one DGEMM per batch, one (clean) tile.
  BOOST_CHECK_EQUAL(TA::detail::g_strided_dgemm_ce_e_calls.load(), NB);
#endif
}

// Task 3.3 Step 2b: P=1 and Q=1 inner-extent edges -> the rank-1 outer product
// degenerates to a scalar*scalar per k. Two operand tiles (like R1), cross-tile
// beta=1; (count build) == 2*NB DGEMMs (one per tile per batch, all clean).
BOOST_AUTO_TEST_CASE(regime_a_oprod_mapping_p1_q1) {
  namespace blas = TiledArray::math::blas;
  const std::size_t NB = 2;
  const std::size_t P = 1, Q = 1;  // degenerate inner extents
  const std::size_t VOL0 = 3, VOL1 = 2;

  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{NB}, /*batch=*/1, [&](std::size_t) { return TA::Range{P, Q}; });
  auto make_L = [&](std::size_t VOL, double seed) {
    Outer L = TA::detail::arena_outer_init<Outer>(
        TA::Range{VOL}, NB, [&](std::size_t) { return TA::Range{P}; });
    for (std::size_t o = 0; o < NB * VOL; ++o)
      for (std::size_t e = 0; e < L.data()[o].size(); ++e)
        L.data()[o].data()[e] = seed + 0.01 * o + e;
    return L;
  };
  auto make_R = [&](std::size_t VOL, double seed) {
    Outer R = TA::detail::arena_outer_init<Outer>(
        TA::Range{VOL}, NB, [&](std::size_t) { return TA::Range{Q}; });
    for (std::size_t o = 0; o < NB * VOL; ++o)
      for (std::size_t e = 0; e < R.data()[o].size(); ++e)
        R.data()[o].data()[e] = seed + 0.02 * o + e;
    return R;
  };
  Outer L0 = make_L(VOL0, 1.0), R0 = make_R(VOL0, 2.0);
  Outer L1 = make_L(VOL1, 5.0), R1 = make_R(VOL1, 7.0);

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_e_calls.store(0);
#endif
  auto cview = C.reshape(TA::Range{1}, NB);
  TA::detail::arena_strided_dgemm_ce_e(cview, L0, R0, /*M=*/std::size_t{1},
                                       /*N=*/std::size_t{1}, /*K=*/VOL0,
                                       blas::NoTranspose, blas::NoTranspose, 1.0);
  auto cview2 = C.reshape(TA::Range{1}, NB);
  TA::detail::arena_strided_dgemm_ce_e(cview2, L1, R1, /*M=*/std::size_t{1},
                                       /*N=*/std::size_t{1}, /*K=*/VOL1,
                                       blas::NoTranspose, blas::NoTranspose, 1.0);

  for (std::size_t b = 0; b < NB; ++b) {
    std::vector<double> ref(P * Q, 0.0);
    auto add_tile = [&](const Outer& L, const Outer& R, std::size_t VOL) {
      for (std::size_t k = 0; k < VOL; ++k) {
        const double* l = L.data()[b * VOL + k].data();  // P==1
        const double* r = R.data()[b * VOL + k].data();  // Q==1
        ref[0] += l[0] * r[0];  // scalar*scalar
      }
    };
    add_tile(L0, R0, VOL0);
    add_tile(L1, R1, VOL1);
    const double* got = C.data()[b].data();
    BOOST_CHECK_CLOSE(got[0], ref[0], 1e-12);
  }
#ifdef TA_STRIDED_DGEMM_COUNT
  BOOST_CHECK_EQUAL(TA::detail::g_strided_dgemm_ce_e_calls.load(),
                    std::size_t{2} * NB);
#endif
}

// ------------------------------- ce+ce ------------------------------------

BOOST_AUTO_TEST_CASE(ce_ce_matches_reference_canonical) {
  const std::size_t Mmu = 3, nK = 2, P = 4, Q = 5;
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P, Q};}, 1.0);
  // canonical (right_op==Transpose) storage: R outer (mu,k), mu slow, k fast.
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu, nK}, 1, [&](std::size_t){return TA::Range{Q};});
  for (std::size_t mu = 0; mu < Mmu; ++mu)
    for (std::size_t k = 0; k < nK; ++k) {
      double* r = R.data()[mu * nK + k].data();
      for (std::size_t e = 0; e < Q; ++e) r[e] = 2.0 + 0.01 * (mu * nK + k) + e;
    }
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu}, 1, [&](std::size_t){return TA::Range{P};});  // zero-init
  namespace blas = TiledArray::math::blas;
  // Mo=1 (no left external), No=Mmu (mu), Ko=nK (k); canonical right_op=Transpose.
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, /*Mo=*/1, /*No=*/Mmu, /*Ko=*/nK,
                                        blas::NoTranspose, blas::Transpose, 1.0);
  auto ref = ref_ce_ce(L, R, Mmu, nK, P, Q, 1.0);
  for (std::size_t mu = 0; mu < Mmu; ++mu) {
    const double* got = C.data()[mu].data();
    for (std::size_t a1 = 0; a1 < P; ++a1)
      BOOST_CHECK_CLOSE(got[a1], ref[mu * P + a1], 1e-12);
  }
}

BOOST_AUTO_TEST_CASE(ce_ce_orientation_aware_no_transpose) {
  const std::size_t Mmu = 3, nK = 2, P = 4, Q = 5;
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P, Q};}, 1.0);
  // non-canonical: R outer (k,mu) -> k slow, mu fast => right_op == NoTranspose
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{nK, Mmu}, 1, [&](std::size_t){return TA::Range{Q};});
  for (std::size_t k = 0; k < nK; ++k)
    for (std::size_t mu = 0; mu < Mmu; ++mu) {
      double* r = R.data()[k * Mmu + mu].data();
      for (std::size_t e = 0; e < Q; ++e) r[e] = 2.0 + 0.01 * (mu * nK + k) + e;
    }
  // canonical mirror (mu*nK+k) holding the SAME (k,mu) data, for the reference
  Outer Rc = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu, nK}, 1, [&](std::size_t){return TA::Range{Q};});
  for (std::size_t k = 0; k < nK; ++k)
    for (std::size_t mu = 0; mu < Mmu; ++mu) {
      const double* s = R.data()[k * Mmu + mu].data();
      double* d = Rc.data()[mu * nK + k].data();
      for (std::size_t e = 0; e < Q; ++e) d[e] = s[e];
    }
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu}, 1, [&](std::size_t){return TA::Range{P};});
  namespace blas = TiledArray::math::blas;
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, 1, Mmu, nK,
                                        blas::NoTranspose, blas::NoTranspose, 1.0);
  auto ref = ref_ce_ce(L, Rc, Mmu, nK, P, Q, 1.0);
  for (std::size_t mu = 0; mu < Mmu; ++mu) {
    const double* got = C.data()[mu].data();
    for (std::size_t a1 = 0; a1 < P; ++a1)
      BOOST_CHECK_CLOSE(got[a1], ref[mu * P + a1], 1e-12);
  }
}

BOOST_AUTO_TEST_CASE(ce_ce_multi_batch) {
  const std::size_t Mmu = 2, nK = 3, P = 3, Q = 4, NB = 2;
  Outer L = TA::detail::arena_outer_init<Outer>(
      TA::Range{nK}, NB, [&](std::size_t){return TA::Range{P, Q};});
  Outer R = TA::detail::arena_outer_init<Outer>(   // (mu,k) canonical
      TA::Range{Mmu, nK}, NB, [&](std::size_t){return TA::Range{Q};});
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu}, NB, [&](std::size_t){return TA::Range{P};});
  for (std::size_t o = 0; o < NB * nK; ++o)
    for (std::size_t e = 0; e < L.data()[o].size(); ++e)
      L.data()[o].data()[e] = 1.0 + 0.01 * o + e;
  for (std::size_t o = 0; o < NB * Mmu * nK; ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  namespace blas = TiledArray::math::blas;
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, 1, Mmu, nK,
                                        blas::NoTranspose, blas::Transpose, 1.0);
  for (std::size_t b = 0; b < NB; ++b) {
    std::vector<double> ref(Mmu * P, 0.0);
    for (std::size_t k = 0; k < nK; ++k) {
      const double* l = L.data()[b * nK + k].data();
      for (std::size_t mu = 0; mu < Mmu; ++mu) {
        const double* r = R.data()[b * Mmu * nK + mu * nK + k].data();
        for (std::size_t a1 = 0; a1 < P; ++a1) {
          double acc = 0;
          for (std::size_t a4 = 0; a4 < Q; ++a4) acc += l[a1 * Q + a4] * r[a4];
          ref[mu * P + a1] += acc;
        }
      }
    }
    for (std::size_t mu = 0; mu < Mmu; ++mu) {
      const double* got = C.data()[b * Mmu + mu].data();
      for (std::size_t a1 = 0; a1 < P; ++a1)
        BOOST_CHECK_CLOSE(got[a1], ref[mu * P + a1], 1e-12);
    }
  }
}

BOOST_AUTO_TEST_CASE(ce_ce_ragged_batch_falls_back_inline) {
  // batch 0 clean; batch 1 ragged Q across mu -> inline fallback for batch 1.
  const std::size_t Mmu = 2, nK = 2, P = 3, Q0 = 4, Q1 = 5, NB = 2;
  Outer L = TA::detail::arena_outer_init<Outer>(
      TA::Range{nK}, NB, [&](std::size_t){return TA::Range{P, Q0};});
  for (std::size_t o = 0; o < NB * nK; ++o)
    for (std::size_t e = 0; e < L.data()[o].size(); ++e)
      L.data()[o].data()[e] = 1.0 + 0.01 * o + e;
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu, nK}, NB, [&](std::size_t o){
        const std::size_t batch = o / (Mmu * nK);
        const std::size_t ord = o % (Mmu * nK);
        const std::size_t mu = ord / nK;
        const bool ragged = (batch == 1 && mu == 1);
        return TA::Range{ragged ? Q1 : Q0};
      });
  for (std::size_t o = 0; o < NB * Mmu * nK; ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu}, NB, [&](std::size_t){return TA::Range{P};});
  namespace blas = TiledArray::math::blas;
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, 1, Mmu, nK,
                                        blas::NoTranspose, blas::Transpose, 1.0);
  for (std::size_t b = 0; b < NB; ++b) {
    std::vector<double> ref(Mmu * P, 0.0);
    for (std::size_t k = 0; k < nK; ++k) {
      const auto& lk = L.data()[b * nK + k];
      const double* l = lk.data();
      for (std::size_t mu = 0; mu < Mmu; ++mu) {
        const auto& rk = R.data()[b * Mmu * nK + mu * nK + k];
        const std::size_t Ql = rk.size();
        if (lk.size() != P * Ql) continue;  // guard
        const double* r = rk.data();
        for (std::size_t a1 = 0; a1 < P; ++a1) {
          double acc = 0;
          for (std::size_t a4 = 0; a4 < Ql; ++a4) acc += l[a1 * Ql + a4] * r[a4];
          ref[mu * P + a1] += acc;
        }
      }
    }
    for (std::size_t mu = 0; mu < Mmu; ++mu) {
      const double* got = C.data()[b * Mmu + mu].data();
      for (std::size_t a1 = 0; a1 < P; ++a1)
        BOOST_CHECK_CLOSE(got[a1], ref[mu * P + a1], 1e-12);
    }
  }
}

BOOST_AUTO_TEST_CASE(ce_ce_applies_factor) {
  const std::size_t Mmu = 2, nK = 4, P = 3, Q = 3;  // nK=4 -> multi-step beta=1
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P, Q};}, 1.0);
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu, nK}, 1, [&](std::size_t){return TA::Range{Q};});
  for (std::size_t mu = 0; mu < Mmu; ++mu)
    for (std::size_t k = 0; k < nK; ++k) {
      double* r = R.data()[mu * nK + k].data();
      for (std::size_t e = 0; e < Q; ++e) r[e] = 2.0 + 0.01 * (mu * nK + k) + e;
    }
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu}, 1, [&](std::size_t){return TA::Range{P};});
  namespace blas = TiledArray::math::blas;
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, 1, Mmu, nK,
                                        blas::NoTranspose, blas::Transpose, 0.5);
  auto ref = ref_ce_ce(L, R, Mmu, nK, P, Q, 0.5);
  for (std::size_t mu = 0; mu < Mmu; ++mu) {
    const double* got = C.data()[mu].data();
    for (std::size_t a1 = 0; a1 < P; ++a1)
      BOOST_CHECK_CLOSE(got[a1], ref[mu * P + a1], 1e-12);
  }
}

// Presence-first clean-check (no stride-probe UB): a genuinely EMPTY mid-run
// R cell drives the presence guard BEFORE any cell-0->1 pointer subtraction.
BOOST_AUTO_TEST_CASE(ce_ce_empty_mid_run_falls_back) {
  const std::size_t Mmu = 3, nK = 2, P = 4, Q = 5;
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P, Q};}, 1.0);
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu, nK}, 1, [&](std::size_t o){
        const std::size_t mu = o / nK, k = o % nK;
        return (mu == 1 && k == 0) ? TA::Range{} : TA::Range{Q};
      });
  for (std::size_t o = 0; o < R.range().volume(); ++o) {
    Inner& c = R.data()[o];
    if (!c) continue;
    for (std::size_t e = 0; e < c.size(); ++e) c.data()[e] = 2.0 + 0.01 * o + e;
  }
  BOOST_REQUIRE(!R.data()[1 * nK + 0]);
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu}, 1, [&](std::size_t){return TA::Range{P};});
  namespace blas = TiledArray::math::blas;
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, /*Mo=*/1, /*No=*/Mmu, /*Ko=*/nK,
                                        blas::NoTranspose, blas::Transpose, 1.0);
  std::vector<double> ref(Mmu * P, 0.0);
  for (std::size_t k = 0; k < nK; ++k) {
    const double* l = L.data()[k].data();
    for (std::size_t mu = 0; mu < Mmu; ++mu) {
      const auto& rk = R.data()[mu * nK + k];
      if (!rk) continue;
      const double* r = rk.data();
      for (std::size_t a1 = 0; a1 < P; ++a1) {
        double acc = 0;
        for (std::size_t a4 = 0; a4 < Q; ++a4) acc += l[a1 * Q + a4] * r[a4];
        ref[mu * P + a1] += acc;
      }
    }
  }
  for (std::size_t mu = 0; mu < Mmu; ++mu) {
    const double* got = C.data()[mu].data();
    for (std::size_t a1 = 0; a1 < P; ++a1)
      BOOST_CHECK_CLOSE(got[a1], ref[mu * P + a1], 1e-12);
  }
}

// Addition B: left-external Mo>1 (single batch). The left-external m rides as
// an outer loop; R (function of k,mu only) is reused across m. Independent
// reference: c[m,mu](a1) = sum_k sum_{a4} L[m,k](a1,a4) * R[mu,k](a4).
BOOST_AUTO_TEST_CASE(ce_ce_left_external) {
  const std::size_t Mo = 2, Mmu = 3, nK = 2, P = 4, Q = 5;
  // L outer (Mo,nK) row-major (m slow, k fast), inner {P,Q}.
  Outer L = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, nK}, 1, [&](std::size_t){return TA::Range{P, Q};});
  for (std::size_t o = 0; o < Mo * nK; ++o)
    for (std::size_t e = 0; e < L.data()[o].size(); ++e)
      L.data()[o].data()[e] = 1.0 + 0.01 * o + e;
  // R outer (Mmu,nK) canonical (mu slow, k fast), inner {Q}.
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu, nK}, 1, [&](std::size_t){return TA::Range{Q};});
  for (std::size_t o = 0; o < Mmu * nK; ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  // C outer (Mo,Mmu) row-major (m slow, mu fast), inner {P}.
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, Mmu}, 1, [&](std::size_t){return TA::Range{P};});
  namespace blas = TiledArray::math::blas;
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, /*Mo=*/Mo, /*No=*/Mmu,
                                        /*Ko=*/nK, blas::NoTranspose,
                                        blas::Transpose, 1.0);
  for (std::size_t m = 0; m < Mo; ++m) {
    std::vector<double> ref(Mmu * P, 0.0);
    for (std::size_t k = 0; k < nK; ++k) {
      const double* l = L.data()[m * nK + k].data();  // P x Q row-major
      for (std::size_t mu = 0; mu < Mmu; ++mu) {
        const double* r = R.data()[mu * nK + k].data();
        for (std::size_t a1 = 0; a1 < P; ++a1) {
          double acc = 0;
          for (std::size_t a4 = 0; a4 < Q; ++a4) acc += l[a1 * Q + a4] * r[a4];
          ref[mu * P + a1] += acc;
        }
      }
    }
    for (std::size_t mu = 0; mu < Mmu; ++mu) {
      const double* got = C.data()[m * Mmu + mu].data();
      for (std::size_t a1 = 0; a1 < P; ++a1)
        BOOST_CHECK_CLOSE(got[a1], ref[mu * P + a1], 1e-12);
    }
  }
}

// Addition B: Mo>1 AND nbatch>1.
BOOST_AUTO_TEST_CASE(ce_ce_left_external_multi_batch) {
  const std::size_t Mo = 2, Mmu = 2, nK = 3, P = 3, Q = 4, NB = 2;
  Outer L = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, nK}, NB, [&](std::size_t){return TA::Range{P, Q};});
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu, nK}, NB, [&](std::size_t){return TA::Range{Q};});
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, Mmu}, NB, [&](std::size_t){return TA::Range{P};});
  for (std::size_t o = 0; o < NB * Mo * nK; ++o)
    for (std::size_t e = 0; e < L.data()[o].size(); ++e)
      L.data()[o].data()[e] = 1.0 + 0.01 * o + e;
  for (std::size_t o = 0; o < NB * Mmu * nK; ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  namespace blas = TiledArray::math::blas;
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, Mo, Mmu, nK,
                                        blas::NoTranspose, blas::Transpose, 1.0);
  for (std::size_t b = 0; b < NB; ++b)
    for (std::size_t m = 0; m < Mo; ++m) {
      std::vector<double> ref(Mmu * P, 0.0);
      for (std::size_t k = 0; k < nK; ++k) {
        const double* l = L.data()[b * Mo * nK + m * nK + k].data();
        for (std::size_t mu = 0; mu < Mmu; ++mu) {
          const double* r = R.data()[b * Mmu * nK + mu * nK + k].data();
          for (std::size_t a1 = 0; a1 < P; ++a1) {
            double acc = 0;
            for (std::size_t a4 = 0; a4 < Q; ++a4) acc += l[a1 * Q + a4] * r[a4];
            ref[mu * P + a1] += acc;
          }
        }
      }
      for (std::size_t mu = 0; mu < Mmu; ++mu) {
        const double* got = C.data()[b * Mo * Mmu + m * Mmu + mu].data();
        for (std::size_t a1 = 0; a1 < P; ++a1)
          BOOST_CHECK_CLOSE(got[a1], ref[mu * P + a1], 1e-12);
      }
    }
}

// Addition B coverage: Mo>1 with the non-canonical right orientation
// (right_op == NoTranspose, R outer (k,mu)). Mirrors
// ce_ce_orientation_aware_no_transpose but with a left-external loop, so the
// orientation-aware r_off is exercised together with the m-loop.
BOOST_AUTO_TEST_CASE(ce_ce_left_external_orientation_no_transpose) {
  const std::size_t Mo = 2, Mmu = 3, nK = 2, P = 4, Q = 5;
  Outer L = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, nK}, 1, [&](std::size_t){return TA::Range{P, Q};});
  for (std::size_t o = 0; o < Mo * nK; ++o)
    for (std::size_t e = 0; e < L.data()[o].size(); ++e)
      L.data()[o].data()[e] = 1.0 + 0.01 * o + e;
  // non-canonical: R outer (k,mu) -> k slow, mu fast => right_op==NoTranspose.
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{nK, Mmu}, 1, [&](std::size_t){return TA::Range{Q};});
  for (std::size_t k = 0; k < nK; ++k)
    for (std::size_t mu = 0; mu < Mmu; ++mu) {
      double* r = R.data()[k * Mmu + mu].data();
      for (std::size_t e = 0; e < Q; ++e) r[e] = 2.0 + 0.01 * (k * Mmu + mu) + e;
    }
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, Mmu}, 1, [&](std::size_t){return TA::Range{P};});
  namespace blas = TiledArray::math::blas;
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, Mo, Mmu, nK, blas::NoTranspose,
                                        blas::NoTranspose, 1.0);
  for (std::size_t m = 0; m < Mo; ++m) {
    std::vector<double> ref(Mmu * P, 0.0);
    for (std::size_t k = 0; k < nK; ++k) {
      const double* l = L.data()[m * nK + k].data();
      for (std::size_t mu = 0; mu < Mmu; ++mu) {
        const double* r = R.data()[k * Mmu + mu].data();  // (k,mu) layout
        for (std::size_t a1 = 0; a1 < P; ++a1) {
          double acc = 0;
          for (std::size_t a4 = 0; a4 < Q; ++a4) acc += l[a1 * Q + a4] * r[a4];
          ref[mu * P + a1] += acc;
        }
      }
    }
    for (std::size_t mu = 0; mu < Mmu; ++mu) {
      const double* got = C.data()[m * Mmu + mu].data();
      for (std::size_t a1 = 0; a1 < P; ++a1)
        BOOST_CHECK_CLOSE(got[a1], ref[mu * P + a1], 1e-12);
    }
  }
}

// Addition B coverage: ragged inner size under Mo>1. One R mu-cell is ragged,
// so the strided clean-check declines (per-m) and the inline per-cell fallback
// runs for EACH left-external block (each cell computed exactly once, the
// size-mismatched (mu,k) skipped just as the kernel skips it).
BOOST_AUTO_TEST_CASE(ce_ce_left_external_ragged_falls_back) {
  const std::size_t Mo = 2, Mmu = 2, nK = 2, P = 3, Q0 = 4, Q1 = 5;
  Outer L = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, nK}, 1, [&](std::size_t){return TA::Range{P, Q0};});
  for (std::size_t o = 0; o < Mo * nK; ++o)
    for (std::size_t e = 0; e < L.data()[o].size(); ++e)
      L.data()[o].data()[e] = 1.0 + 0.01 * o + e;
  // R[mu=1,k=0] ragged (Q1) -> mu-run not uniform -> clean-check fails (both m).
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu, nK}, 1, [&](std::size_t o){
        const std::size_t mu = o / nK, k = o % nK;
        return TA::Range{(mu == 1 && k == 0) ? Q1 : Q0};
      });
  for (std::size_t o = 0; o < Mmu * nK; ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, Mmu}, 1, [&](std::size_t){return TA::Range{P};});
  namespace blas = TiledArray::math::blas;
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, Mo, Mmu, nK, blas::NoTranspose,
                                        blas::Transpose, 1.0);
  for (std::size_t m = 0; m < Mo; ++m) {
    std::vector<double> ref(Mmu * P, 0.0);
    for (std::size_t k = 0; k < nK; ++k) {
      const auto& lk = L.data()[m * nK + k];
      const double* l = lk.data();
      for (std::size_t mu = 0; mu < Mmu; ++mu) {
        const auto& rk = R.data()[mu * nK + k];
        const std::size_t Ql = rk.size();
        if (lk.size() != P * Ql) continue;  // mirror kernel fallback skip
        const double* r = rk.data();
        for (std::size_t a1 = 0; a1 < P; ++a1) {
          double acc = 0;
          for (std::size_t a4 = 0; a4 < Ql; ++a4) acc += l[a1 * Ql + a4] * r[a4];
          ref[mu * P + a1] += acc;
        }
      }
    }
    for (std::size_t mu = 0; mu < Mmu; ++mu) {
      const double* got = C.data()[m * Mmu + mu].data();
      for (std::size_t a1 = 0; a1 < P; ++a1)
        BOOST_CHECK_CLOSE(got[a1], ref[mu * P + a1], 1e-12);
    }
  }
}

// LEFT-clean mirror core: the LEFT operand inner is the pure contraction vector
// L[m,k](a4); the RIGHT operand carries the inner external R[k,n](a4,b1); result
// inner is the right inner-external b1. The kernel rides the LEFT-external m into
// BLAS M and loops the RIGHT-external n as an outer loop. Independent reference:
//   c[m,n](p) = sum_k sum_{a4} L[m,k](a4) * R[k,n](a4,p).
BOOST_AUTO_TEST_CASE(ce_ce_left_clean_core) {
  const std::size_t Mo = 2, No = 3, nK = 2, P = 4, Q = 5;
  // L (clean) outer (Mo,nK) row-major (m slow, k fast), inner {Q}.
  Outer L = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, nK}, 1, [&](std::size_t) { return TA::Range{Q}; });
  for (std::size_t o = 0; o < Mo * nK; ++o)
    for (std::size_t e = 0; e < L.data()[o].size(); ++e)
      L.data()[o].data()[e] = 1.0 + 0.01 * o + e;
  // R (matrix) outer (nK,No) canonical (k slow, n fast), inner {Q,P} row-major.
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{nK, No}, 1, [&](std::size_t) { return TA::Range{Q, P}; });
  for (std::size_t o = 0; o < nK * No; ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  // C outer (Mo,No) row-major (m slow, n fast), inner {P}.
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, No}, 1, [&](std::size_t) { return TA::Range{P}; });
  namespace blas = TiledArray::math::blas;
  TA::detail::arena_strided_dgemm_ce_ce_left(C, L, R, /*Mo=*/Mo, /*No=*/No,
                                             /*Ko=*/nK, blas::NoTranspose,
                                             blas::NoTranspose, 1.0);
  for (std::size_t m = 0; m < Mo; ++m)
    for (std::size_t n = 0; n < No; ++n) {
      std::vector<double> ref(P, 0.0);
      for (std::size_t k = 0; k < nK; ++k) {
        const double* l = L.data()[m * nK + k].data();  // Q vector
        const double* r = R.data()[k * No + n].data();  // Q x P row-major
        for (std::size_t p = 0; p < P; ++p) {
          double acc = 0;
          for (std::size_t a4 = 0; a4 < Q; ++a4) acc += l[a4] * r[a4 * P + p];
          ref[p] += acc;
        }
      }
      const double* got = C.data()[m * No + n].data();
      for (std::size_t p = 0; p < P; ++p)
        BOOST_CHECK_CLOSE(got[p], ref[p], 1e-12);
    }
}

namespace {
// Compare every result cell against the sparsity-aware reference: a present
// reference vector must match to 1e-10. An empty reference vector means the
// cell received no contribution: an ABSENT (hole) result cell must stay absent
// (invariant 2 / "no writes to holes"), and a PRESENT result cell (e.g. a dense
// C whose only k was size-mismatched) must remain exactly its zero-init value.
// ordinal of C[b,row,col] = (b*nrow+row)*ncol + col.
void check_ce_ce(const Outer& C, const std::vector<std::vector<double>>& ref,
                 std::size_t nbatch, std::size_t nrow, std::size_t ncol,
                 std::size_t P) {
  for (std::size_t b = 0; b < nbatch; ++b)
    for (std::size_t r = 0; r < nrow; ++r)
      for (std::size_t cc = 0; cc < ncol; ++cc) {
        const std::size_t ord = (b * nrow + r) * ncol + cc;
        const Inner& cell = C.data()[ord];
        const std::vector<double>& want = ref[ord];
        if (want.empty()) {
          // No contribution: a hole stays absent; a present cell stays zero
          // (the kernel must not have written anything to it).
          if (cell)
            for (std::size_t a1 = 0; a1 < cell.size(); ++a1)
              BOOST_CHECK_SMALL(cell.data()[a1], 1e-12);
          continue;
        }
        BOOST_REQUIRE(bool(cell));
        BOOST_REQUIRE_EQUAL(cell.size(), P);
        for (std::size_t a1 = 0; a1 < P; ++a1)
          BOOST_CHECK_CLOSE(cell.data()[a1], want[a1], 1e-10);
      }
}

// Zero-fill the present cells of an already-built result tile (the kernel
// accumulates with beta=1, so C must start at zero).
void zero_result(Outer& C, std::size_t ncells) {
  for (std::size_t o = 0; o < ncells; ++o) {
    Inner& c = C.data()[o];
    if (!c) continue;
    for (std::size_t e = 0; e < c.size(); ++e) c.data()[e] = 0.0;
  }
}
}  // namespace

#ifdef TA_STRIDED_DGEMM_COUNT
BOOST_AUTO_TEST_CASE(ce_ce_fires_clean_path) {
  const std::size_t Mmu = 3, nK = 2, P = 4, Q = 5, NB = 2;
  // NB batches, all uniform => every (b,k) DGEMM fires => count == NB*Mo*nK.
  Outer L = TA::detail::arena_outer_init<Outer>(
      TA::Range{nK}, NB, [&](std::size_t){return TA::Range{P, Q};});
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu, nK}, NB, [&](std::size_t){return TA::Range{Q};});
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu}, NB, [&](std::size_t){return TA::Range{P};});
  for (std::size_t o = 0; o < NB * nK; ++o)
    for (std::size_t e = 0; e < L.data()[o].size(); ++e)
      L.data()[o].data()[e] = 1.0 + 0.01 * o + e;
  for (std::size_t o = 0; o < NB * Mmu * nK; ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  namespace blas = TiledArray::math::blas;
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, 1, Mmu, nK,
                                        blas::NoTranspose, blas::Transpose, 1.0);
  // per-DGEMM count: Mo(==1) * nK per batch
  BOOST_CHECK_EQUAL(TA::detail::g_strided_dgemm_ce_ce_right_calls.load(), NB * 1 * nK);
}

// Addition B firing count: Mo>1, nbatch>1 -> clean count == NB*Mo*nK.
BOOST_AUTO_TEST_CASE(ce_ce_left_external_fires_clean_path) {
  const std::size_t Mo = 2, Mmu = 2, nK = 3, P = 3, Q = 4, NB = 2;
  Outer L = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, nK}, NB, [&](std::size_t){return TA::Range{P, Q};});
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu, nK}, NB, [&](std::size_t){return TA::Range{Q};});
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, Mmu}, NB, [&](std::size_t){return TA::Range{P};});
  for (std::size_t o = 0; o < NB * Mo * nK; ++o)
    for (std::size_t e = 0; e < L.data()[o].size(); ++e)
      L.data()[o].data()[e] = 1.0 + 0.01 * o + e;
  for (std::size_t o = 0; o < NB * Mmu * nK; ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  namespace blas = TiledArray::math::blas;
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, Mo, Mmu, nK,
                                        blas::NoTranspose, blas::Transpose, 1.0);
  BOOST_CHECK_EQUAL(TA::detail::g_strided_dgemm_ce_ce_right_calls.load(), NB * Mo * nK);
}

// Was "does_not_fire": under per-k segmentation a mid-run hole no longer drops
// the whole run to scalar -- it splits into per-k contiguous segments that still
// fire as strided GEMMs. Correctness is unchanged; the count now reflects the
// segments the walker issues (the empty mu=1 row breaks each k's run).
BOOST_AUTO_TEST_CASE(ce_ce_scattered_run_segments_and_is_correct) {
  const std::size_t Mmu = 3, nK = 2, P = 4, Q0 = 5, Q1 = 6;
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P, Q0};}, 1.0);
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu, nK}, 1, [&](std::size_t o){
        const std::size_t mu = o / nK;
        return TA::Range{mu == 1 ? Q1 : Q0};
      });
  for (std::size_t o = 0; o < R.range().volume(); ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu}, 1, [&](std::size_t){return TA::Range{P};});
  namespace blas = TiledArray::math::blas;
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, /*Mo=*/1, /*No=*/Mmu, /*Ko=*/nK,
                                        blas::NoTranspose, blas::Transpose, 1.0);
  // mu=1 has a mismatched size, so per k the walker emits segments {0} and {2}
  // (the size-mismatched mu=1 cannot join either) => 2 segments x nK(=2) = 4.
  BOOST_CHECK_EQUAL(TA::detail::g_strided_dgemm_ce_ce_right_calls.load(), 4u);
  std::vector<double> ref(Mmu * P, 0.0);
  for (std::size_t k = 0; k < nK; ++k) {
    const auto& lk = L.data()[k];
    const double* l = lk.data();
    for (std::size_t mu = 0; mu < Mmu; ++mu) {
      const auto& rk = R.data()[mu * nK + k];
      const std::size_t Ql = rk.size();
      if (lk.size() != P * Ql) continue;
      const double* r = rk.data();
      for (std::size_t a1 = 0; a1 < P; ++a1) {
        double acc = 0;
        for (std::size_t a4 = 0; a4 < Ql; ++a4) acc += l[a1 * Ql + a4] * r[a4];
        ref[mu * P + a1] += acc;
      }
    }
  }
  for (std::size_t mu = 0; mu < Mmu; ++mu) {
    const double* got = C.data()[mu].data();
    for (std::size_t a1 = 0; a1 < P; ++a1)
      BOOST_CHECK_CLOSE(got[a1], ref[mu * P + a1], 1e-12);
  }
}

// Page-jump / constant-stride guard in isolation: every R cell has the SAME
// size Q (uniform-size guard passes), but the per-k mu-run is laid at a
// NON-CONSTANT stride -> page-jump guard rejects. Fully-independent reference.
// Was "does_not_fire": a page-jump (non-constant inter-cell stride) no longer
// drops the whole run -- the walker ends a segment at the stride break and
// starts a new strided GEMM, so it fires (correctly) while staying exact.
BOOST_AUTO_TEST_CASE(ce_ce_page_jump_run_segments_and_is_correct) {
  const std::size_t Mmu = 3, nK = 2, P = 4, Q = 5;
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P, Q};}, 1.0);
  Outer Rsrc = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu, nK}, 1, [&](std::size_t){return TA::Range{Q};});
  for (std::size_t o = 0; o < Rsrc.range().volume(); ++o)
    for (std::size_t e = 0; e < Rsrc.data()[o].size(); ++e)
      Rsrc.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  const std::ptrdiff_t S = Rsrc.data()[1].data() - Rsrc.data()[0].data();
  BOOST_REQUIRE_GT(S, 0);
  for (std::size_t o = 0; o < Rsrc.range().volume(); ++o)
    BOOST_REQUIRE_EQUAL(Rsrc.data()[o].data(),
                        Rsrc.data()[0].data() + static_cast<std::ptrdiff_t>(o) * S);
  const std::size_t perm[Mmu] = {0, 2, 1};  // non-monotone -> non-constant stride
  std::vector<std::size_t> phys(Mmu * nK);
  for (std::size_t mu = 0; mu < Mmu; ++mu)
    for (std::size_t k = 0; k < nK; ++k)
      phys[mu * nK + k] = perm[mu] * nK + k;
  Outer R = assemble_aliased(Rsrc, TA::Range{Mmu, nK}, phys);
  for (std::size_t o = 0; o < R.range().volume(); ++o)
    BOOST_REQUIRE_EQUAL(R.data()[o].size(), Q);
  for (std::size_t k = 0; k < nK; ++k) {
    const std::ptrdiff_t d01 =
        R.data()[1 * nK + k].data() - R.data()[0 * nK + k].data();
    const std::ptrdiff_t d12 =
        R.data()[2 * nK + k].data() - R.data()[1 * nK + k].data();
    BOOST_REQUIRE_NE(d01, d12);
  }
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu}, 1, [&](std::size_t){return TA::Range{P};});
  namespace blas = TiledArray::math::blas;
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, /*Mo=*/1, /*No=*/Mmu, /*Ko=*/nK,
                                        blas::NoTranspose, blas::Transpose, 1.0);
  // perm {0,2,1} makes the inter-cell stride non-constant, so per k the walker
  // breaks the run into constant-stride segments {0,1} and {2} => 2 x nK(=2) = 4.
  BOOST_CHECK_EQUAL(TA::detail::g_strided_dgemm_ce_ce_right_calls.load(), 4u);
  auto ref = ref_ce_ce(L, R, Mmu, nK, P, Q, 1.0);
  for (std::size_t mu = 0; mu < Mmu; ++mu) {
    const double* got = C.data()[mu].data();
    for (std::size_t a1 = 0; a1 < P; ++a1)
      BOOST_CHECK_CLOSE(got[a1], ref[mu * P + a1], 1e-12);
  }
}

// LEFT-clean firing count: Mo>1, No>1, single batch -> one DGEMM per (n,k)
// (the left-external m is ridden into BLAS M) => count == No*nK.
BOOST_AUTO_TEST_CASE(ce_ce_left_clean_fires_clean_path) {
  const std::size_t Mo = 2, No = 3, nK = 2, P = 4, Q = 5;
  Outer L = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, nK}, 1, [&](std::size_t) { return TA::Range{Q}; });
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{nK, No}, 1, [&](std::size_t) { return TA::Range{Q, P}; });
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, No}, 1, [&](std::size_t) { return TA::Range{P}; });
  for (std::size_t o = 0; o < Mo * nK; ++o)
    for (std::size_t e = 0; e < L.data()[o].size(); ++e)
      L.data()[o].data()[e] = 1.0 + 0.01 * o + e;
  for (std::size_t o = 0; o < nK * No; ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  namespace blas = TiledArray::math::blas;
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
  TA::detail::arena_strided_dgemm_ce_ce_left(C, L, R, Mo, No, nK,
                                             blas::NoTranspose, blas::NoTranspose,
                                             1.0);
  BOOST_CHECK_EQUAL(TA::detail::g_strided_dgemm_ce_ce_left_calls.load(),
                    No * nK);
}

// LEFT page-jump: a non-constant stride on the strided LEFT operand L (along m)
// must break the m-run into constant-stride segments. perm {0,2,1} over Mo=3
// makes L[1],L[2] non-uniformly spaced => per (n,k) the walker emits segments
// {0,1} and {2} => 2 x No(=1) x nK(=2) = 4 segment GEMMs. Result stays exact.
BOOST_AUTO_TEST_CASE(ce_ce_left_page_jump_run_segments_and_is_correct) {
  const std::size_t Mo = 3, No = 1, nK = 2, P = 4, Q = 5;
  Outer Lsrc = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, nK}, 1, [&](std::size_t){ return TA::Range{Q}; });
  for (std::size_t o = 0; o < Lsrc.range().volume(); ++o)
    for (std::size_t e = 0; e < Lsrc.data()[o].size(); ++e)
      Lsrc.data()[o].data()[e] = 1.0 + 0.01 * o + e;
  // Alias L cells in a non-monotone m order (k stays fast): logical (m,k) ->
  // physical (perm[m],k). This yields a uniform-size run with non-constant stride.
  const std::size_t perm[Mo] = {0, 2, 1};
  std::vector<std::size_t> phys(Mo * nK);
  for (std::size_t m = 0; m < Mo; ++m)
    for (std::size_t k = 0; k < nK; ++k)
      phys[m * nK + k] = perm[m] * nK + k;
  Outer L = assemble_aliased(Lsrc, TA::Range{Mo, nK}, phys);
  // Confirm the stride really is non-constant for at least one k.
  for (std::size_t k = 0; k < nK; ++k) {
    const std::ptrdiff_t d01 =
        L.data()[1 * nK + k].data() - L.data()[0 * nK + k].data();
    const std::ptrdiff_t d12 =
        L.data()[2 * nK + k].data() - L.data()[1 * nK + k].data();
    BOOST_REQUIRE_NE(d01, d12);
  }
  Outer R = make_filled(TA::Range{nK, No},
                        [&](std::size_t){ return TA::Range{Q, P}; }, 2.0);
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, No}, 1, [&](std::size_t){ return TA::Range{P}; });
  zero_result(C, C.range().volume());
  namespace blas = TiledArray::math::blas;
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
  TA::detail::arena_strided_dgemm_ce_ce_left(C, L, R, Mo, No, nK,
      blas::NoTranspose, blas::NoTranspose, 1.0);
  BOOST_CHECK_EQUAL(TA::detail::g_strided_dgemm_ce_ce_left_calls.load(), 4u);
  auto ref = ref_ce_ce_left_sparse(L, R, Mo, No, nK, P, 1.0);
  check_ce_ce(C, ref, 1, Mo, No, P);
}

// RIGHT result-C page-jump: a non-constant stride on the RESULT C cells (along
// μ̃) must also break segments (the sC guard, distinct from the sR operand
// guard). perm {0,2,1} over Mmu=3 aliases C in non-monotone order. The kernel
// writes to logical C[μ̃]; with non-constant C stride it segments {0,1},{2}
// per k => 2 x nK(=2) = 4 segment GEMMs, still exact.
BOOST_AUTO_TEST_CASE(ce_ce_right_result_page_jump_segments_and_is_correct) {
  const std::size_t Mmu = 3, nK = 2, P = 4, Q = 5;
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P, Q};}, 1.0);
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu, nK}, 1, [&](std::size_t){return TA::Range{Q};});
  for (std::size_t o = 0; o < R.range().volume(); ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  // Build a dense C, then alias its cells in non-monotone μ̃ order so the result
  // run has a non-constant .data() stride. C is 1-D over μ̃ (Mo=1).
  Outer Csrc = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu}, 1, [&](std::size_t){return TA::Range{P};});
  zero_result(Csrc, Csrc.range().volume());
  const std::size_t perm[Mmu] = {0, 2, 1};
  std::vector<std::size_t> phys(Mmu);
  for (std::size_t mu = 0; mu < Mmu; ++mu) phys[mu] = perm[mu];
  Outer C = assemble_aliased(Csrc, TA::Range{Mmu}, phys);
  const std::ptrdiff_t d01 = C.data()[1].data() - C.data()[0].data();
  const std::ptrdiff_t d12 = C.data()[2].data() - C.data()[1].data();
  BOOST_REQUIRE_NE(d01, d12);  // non-constant result stride
  namespace blas = TiledArray::math::blas;
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, /*Mo=*/1, /*No=*/Mmu, /*Ko=*/nK,
      blas::NoTranspose, blas::Transpose, 1.0);
  BOOST_CHECK_EQUAL(TA::detail::g_strided_dgemm_ce_ce_right_calls.load(), 4u);
  auto ref = ref_ce_ce(L, R, Mmu, nK, P, Q, 1.0);
  for (std::size_t mu = 0; mu < Mmu; ++mu) {
    const double* got = C.data()[mu].data();
    for (std::size_t a1 = 0; a1 < P; ++a1)
      BOOST_CHECK_CLOSE(got[a1], ref[mu * P + a1], 1e-12);
  }
}
#endif

// ===========================================================================
// Per-k segmented strided-DGEMM tests (T1-T15). The kernel walks each present k
// and emits one strided GEMM per maximal contiguous (present + uniform-stride +
// size-matched) segment along the strided axis, skipping holes, accumulating
// with beta=1 across k and across segments, never touching absent result cells.
// All use the canonical right convention (right_op=Transpose: R outer mu slow,
// k fast). Result/operand outer layouts match ref_ce_ce_right/left_sparse.
// ===========================================================================

// T1: dense run, no holes -> one full-run segment per k == today's clean path.
// Pins golden values (also matches the original dense ref_ce_ce).
BOOST_AUTO_TEST_CASE(ce_ce_seg_dense_regression) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 1, Mmu = 6, nK = 2, P = 3, Q = 4;
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P, Q};}, 1.0);
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu, nK}, 1, [&](std::size_t){ return TA::Range{Q}; });
  for (std::size_t o = 0; o < R.range().volume(); ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu}, 1, [&](std::size_t){ return TA::Range{P}; });
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, Mo, Mmu, nK,
      blas::NoTranspose, blas::Transpose, 1.0);
  auto ref = ref_ce_ce_right_sparse(L, R, Mo, Mmu, nK, P, 1.0);
  check_ce_ce(C, ref, 1, Mo, Mmu, P);
  // Independent naive-oracle cross-check.
  auto ref0 = ref_ce_ce(L, R, Mmu, nK, P, Q, 1.0);
  for (std::size_t mu = 0; mu < Mmu; ++mu)
    for (std::size_t a1 = 0; a1 < P; ++a1)
      BOOST_CHECK_CLOSE(C.data()[mu].data()[a1], ref0[mu * P + a1], 1e-10);
  // Frozen golden baseline: the verified dense-path output for exactly this
  // Mmu=6,nK=2,P=3,Q=4 dense fill, cross-validated by the independent naive
  // ref_ce_ce oracle (checked above at 1e-10). The segmented kernel on a dense
  // run must reproduce these bitwise-stable values; a drift here flags a
  // regression the recomputed references could not (invariant 4 / T1 golden).
  static const double T1_GOLD[18] = {
      80.2404000000, 192.4004000000, 304.5604000000,
      80.6412000000, 193.4412000000, 306.2412000000,
      81.0420000000, 194.4820000000, 307.9220000000,
      81.4428000000, 195.5228000000, 309.6028000000,
      81.8436000000, 196.5636000000, 311.2836000000,
      82.2444000000, 197.6044000000, 312.9644000000,
  };
  for (std::size_t mu = 0; mu < Mmu; ++mu)
    for (std::size_t a1 = 0; a1 < P; ++a1)
      BOOST_CHECK_CLOSE(C.data()[mu].data()[a1], T1_GOLD[mu * P + a1], 1e-10);
}

// T2: single interior hole at mu=4 in BOTH C and R (all k) -> two segments
// [0,4),[5,8); C[4] stays absent; rest exact.
BOOST_AUTO_TEST_CASE(ce_ce_seg_single_interior_hole) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 1, Mmu = 8, nK = 2, P = 3, Q = 4;
  auto rhole = [&](std::size_t o) { return (o / nK) == 4; };  // R[mu=4,*]
  auto chole = [&](std::size_t o) { return o == 4; };          // C[mu=4]
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P, Q};}, 1.0);
  Outer R = make_sparse(TA::Range{Mmu, nK}, 1,
                        [&](std::size_t){ return TA::Range{Q}; }, rhole, 2.0);
  Outer C = make_sparse(TA::Range{Mmu}, 1,
                        [&](std::size_t){ return TA::Range{P}; }, chole, 0.0);
  zero_result(C, Mmu);
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, Mo, Mmu, nK,
      blas::NoTranspose, blas::Transpose, 1.0);
  auto ref = ref_ce_ce_right_sparse(L, R, Mo, Mmu, nK, P, 1.0);
  check_ce_ce(C, ref, 1, Mo, Mmu, P);
}

// Invariant 2 (no writes to absent result cells): a result run with holes must
// leave those hole cells ABSENT (null) after the kernel runs -- the segmenter
// only ever writes into pre-existing present cells, never allocates a tile in a
// hole. Strictly pins what check_ce_ce's empty-ref branch only checks leniently.
BOOST_AUTO_TEST_CASE(ce_ce_seg_holes_stay_absent) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 1, Mmu = 5, nK = 1, P = 3, Q = 4;
  auto rhole = [&](std::size_t o) { return o == 1 || o == 3; };
  auto chole = [&](std::size_t o) { return o == 1 || o == 3; };
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P, Q};}, 1.0);
  Outer R = make_sparse(TA::Range{Mmu, nK}, 1,
                        [&](std::size_t){ return TA::Range{Q}; }, rhole, 2.0);
  Outer C = make_sparse(TA::Range{Mmu}, 1,
                        [&](std::size_t){ return TA::Range{P}; }, chole, 0.0);
  zero_result(C, Mmu);
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, Mo, Mmu, nK,
      blas::NoTranspose, blas::Transpose, 1.0);
  // hole cells stay absent; present cells stay present.
  for (std::size_t mu = 0; mu < Mmu; ++mu) {
    if (chole(mu))
      BOOST_CHECK(!C.data()[mu]);
    else
      BOOST_CHECK(bool(C.data()[mu]));
  }
  auto ref = ref_ce_ce_right_sparse(L, R, Mo, Mmu, nK, P, 1.0);
  check_ce_ce(C, ref, 1, Mo, Mmu, P);
}

// T3: holes at both edges (mu=0 and mu=Mmu-1) -> one interior segment.
BOOST_AUTO_TEST_CASE(ce_ce_seg_edge_holes) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 1, Mmu = 6, nK = 2, P = 3, Q = 4;
  auto rhole = [&](std::size_t o) {
    const std::size_t mu = o / nK;
    return mu == 0 || mu == Mmu - 1;
  };
  auto chole = [&](std::size_t o) { return o == 0 || o == Mmu - 1; };
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P, Q};}, 1.0);
  Outer R = make_sparse(TA::Range{Mmu, nK}, 1,
                        [&](std::size_t){ return TA::Range{Q}; }, rhole, 2.0);
  Outer C = make_sparse(TA::Range{Mmu}, 1,
                        [&](std::size_t){ return TA::Range{P}; }, chole, 0.0);
  zero_result(C, Mmu);
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, Mo, Mmu, nK,
      blas::NoTranspose, blas::Transpose, 1.0);
  auto ref = ref_ce_ce_right_sparse(L, R, Mo, Mmu, nK, P, 1.0);
  check_ce_ce(C, ref, 1, Mo, Mmu, P);
}

// T4: present at even mu, absent at odd (all k) -> every segment is M=1 (GEMV).
BOOST_AUTO_TEST_CASE(ce_ce_seg_alternating_gemv) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 1, Mmu = 6, nK = 1, P = 3, Q = 4;
  auto rhole = [&](std::size_t o) { return ((o / nK) % 2) == 1; };
  auto chole = [&](std::size_t o) { return (o % 2) == 1; };
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P, Q};}, 1.0);
  Outer R = make_sparse(TA::Range{Mmu, nK}, 1,
                        [&](std::size_t){ return TA::Range{Q}; }, rhole, 2.0);
  Outer C = make_sparse(TA::Range{Mmu}, 1,
                        [&](std::size_t){ return TA::Range{P}; }, chole, 0.0);
  zero_result(C, Mmu);
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, Mo, Mmu, nK,
      blas::NoTranspose, blas::Transpose, 1.0);
  auto ref = ref_ce_ce_right_sparse(L, R, Mo, Mmu, nK, P, 1.0);
  check_ce_ce(C, ref, 1, Mo, Mmu, P);
}

// T5 (crux): per-k misaligned holes. nK=2, Mmu=3, Mo=1.
//   R canonical index = mu*nK + k. k=0 present at mu={0,2} (hole mu=1,k=0 -> 2);
//   k=1 present at mu={1,2} (hole mu=0,k=1 -> 1). C present at {0,1,2} (union).
//   Per-k segmentation: k=0 -> {0},{2}; k=1 -> {1,2}. C[2] both-k accumulated.
BOOST_AUTO_TEST_CASE(ce_ce_seg_per_k_misaligned) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 1, Mmu = 3, nK = 2, P = 3, Q = 4;
  auto rhole = [&](std::size_t o) { return o == 2 || o == 1; };  // (mu=1,k=0),(mu=0,k=1)
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P, Q};}, 1.0);
  Outer R = make_sparse(TA::Range{Mmu, nK}, 1,
                        [&](std::size_t){ return TA::Range{Q}; }, rhole, 2.0);
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu}, 1, [&](std::size_t){ return TA::Range{P}; });
  zero_result(C, Mmu);
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, Mo, Mmu, nK,
      blas::NoTranspose, blas::Transpose, 1.0);
  auto ref = ref_ce_ce_right_sparse(L, R, Mo, Mmu, nK, P, 1.0);
  check_ce_ce(C, ref, 1, Mo, Mmu, P);
}

// T6: one full k has L[m,k] absent -> that k skipped; others contribute.
BOOST_AUTO_TEST_CASE(ce_ce_seg_absent_k) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 1, Mmu = 5, nK = 3, P = 3, Q = 4;
  auto lhole = [&](std::size_t o) { return o == 1; };  // L[k=1] absent
  Outer L = make_sparse(TA::Range{nK}, 1,
                        [&](std::size_t){ return TA::Range{P, Q}; }, lhole, 1.0);
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu, nK}, 1, [&](std::size_t){ return TA::Range{Q}; });
  for (std::size_t o = 0; o < R.range().volume(); ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu}, 1, [&](std::size_t){ return TA::Range{P}; });
  zero_result(C, Mmu);
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, Mo, Mmu, nK,
      blas::NoTranspose, blas::Transpose, 1.0);
  auto ref = ref_ce_ce_right_sparse(L, R, Mo, Mmu, nK, P, 1.0);
  check_ce_ce(C, ref, 1, Mo, Mmu, P);
}

// T7: left_inner_transposed=true together with an interior hole; verifies the
// transb folding is correct per segment. L stored Q x P (matrix_transpose).
BOOST_AUTO_TEST_CASE(ce_ce_seg_transposed_inner) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 1, Mmu = 6, nK = 2, P = 3, Q = 4;
  auto rhole = [&](std::size_t o) { return (o / nK) == 3; };
  auto chole = [&](std::size_t o) { return o == 3; };
  // L stored Q x P (transposed layout), filled deterministically.
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{Q, P};}, 1.0);
  Outer R = make_sparse(TA::Range{Mmu, nK}, 1,
                        [&](std::size_t){ return TA::Range{Q}; }, rhole, 2.0);
  Outer C = make_sparse(TA::Range{Mmu}, 1,
                        [&](std::size_t){ return TA::Range{P}; }, chole, 0.0);
  zero_result(C, Mmu);
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, Mo, Mmu, nK,
      blas::NoTranspose, blas::Transpose, 1.0, /*left_inner_transposed=*/true);
  auto ref = ref_ce_ce_right_sparse(L, R, Mo, Mmu, nK, P, 1.0, 1, /*lt=*/true);
  check_ce_ce(C, ref, 1, Mo, Mmu, P);
}

// T8: NB=2; batch 0 dense, batch 1 has the T5 per-k misaligned pattern.
BOOST_AUTO_TEST_CASE(ce_ce_seg_multi_batch_sparse) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 1, Mmu = 3, nK = 2, P = 3, Q = 4, NB = 2;
  // R index within batch = mu*nK + k; holes only in batch 1 (offsets 2 and 1).
  auto rhole = [&](std::size_t o) {
    const std::size_t per = Mmu * nK;
    return (o / per) == 1 && ((o % per) == 2 || (o % per) == 1);
  };
  // NB batches of L (the kernel indexes L per-batch).
  Outer Lb = TA::detail::arena_outer_init<Outer>(
      TA::Range{nK}, NB, [&](std::size_t){ return TA::Range{P, Q}; });
  for (std::size_t o = 0; o < NB * nK; ++o)
    for (std::size_t e = 0; e < Lb.data()[o].size(); ++e)
      Lb.data()[o].data()[e] = 1.0 + 0.01 * o + e;
  Outer R = make_sparse(TA::Range{Mmu, nK}, NB,
                        [&](std::size_t){ return TA::Range{Q}; }, rhole, 2.0);
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu}, NB, [&](std::size_t){ return TA::Range{P}; });
  zero_result(C, NB * Mmu);
  TA::detail::arena_strided_dgemm_ce_ce_right(C, Lb, R, Mo, Mmu, nK,
      blas::NoTranspose, blas::Transpose, 1.0);
  auto ref = ref_ce_ce_right_sparse(Lb, R, Mo, Mmu, nK, P, 1.0, NB);
  check_ce_ce(C, ref, NB, Mo, Mmu, P);
}

// T9: T2 hole pattern with factor=2.5 -> scaling correct with holes.
BOOST_AUTO_TEST_CASE(ce_ce_seg_applies_factor) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 1, Mmu = 8, nK = 2, P = 3, Q = 4;
  auto rhole = [&](std::size_t o) { return (o / nK) == 4; };
  auto chole = [&](std::size_t o) { return o == 4; };
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P, Q};}, 1.0);
  Outer R = make_sparse(TA::Range{Mmu, nK}, 1,
                        [&](std::size_t){ return TA::Range{Q}; }, rhole, 2.0);
  Outer C = make_sparse(TA::Range{Mmu}, 1,
                        [&](std::size_t){ return TA::Range{P}; }, chole, 0.0);
  zero_result(C, Mmu);
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, Mo, Mmu, nK,
      blas::NoTranspose, blas::Transpose, 2.5);
  auto ref = ref_ce_ce_right_sparse(L, R, Mo, Mmu, nK, P, 2.5);
  check_ce_ce(C, ref, 1, Mo, Mmu, P);
}

// T10: one R[mu,k] present but the WRONG inner size (!= Q) for the single k.
// Defensive: a segment cannot include it (size mismatch), so it is skipped; the
// reference skips it too via its lk.size()!=P*Q / Q-mismatch guard. No crash.
BOOST_AUTO_TEST_CASE(ce_ce_seg_size_mismatch_defensive) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 1, Mmu = 5, nK = 1, P = 3, Q = 4;
  // R[mu=2,k=0] gets size Q+1; that breaks size-match so it cannot join a Q-run.
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P, Q};}, 1.0);
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu, nK}, 1,
      [&](std::size_t o){ return (o / nK) == 2 ? TA::Range{Q + 1} : TA::Range{Q}; });
  for (std::size_t o = 0; o < R.range().volume(); ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu}, 1, [&](std::size_t){ return TA::Range{P}; });
  zero_result(C, Mmu);
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, Mo, Mmu, nK,
      blas::NoTranspose, blas::Transpose, 1.0);
  auto ref = ref_ce_ce_right_sparse(L, R, Mo, Mmu, nK, P, 1.0);
  check_ce_ce(C, ref, 1, Mo, Mmu, P);
}

// T11: T5 mirrored for the LEFT kernel (strided over m). nK=2, Mo=3, No=1.
//   L canonical index = m*nK + k. k=0 present at m={0,2} (hole m=1,k=0 -> 2);
//   k=1 present at m={1,2} (hole m=0,k=1 -> 1). C present at {0,1,2} (union).
BOOST_AUTO_TEST_CASE(ce_ce_left_seg_per_k_misaligned) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 3, No = 1, nK = 2, P = 3, Q = 4;
  auto lhole = [&](std::size_t o) { return o == 2 || o == 1; };
  Outer L = make_sparse(TA::Range{Mo, nK}, 1,
                        [&](std::size_t){ return TA::Range{Q}; }, lhole, 1.0);
  // R (matrix) canonical (k slow, n fast) outer (nK,No), inner {Q,P} row-major.
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{nK, No}, 1, [&](std::size_t){ return TA::Range{Q, P}; });
  for (std::size_t o = 0; o < R.range().volume(); ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, No}, 1, [&](std::size_t){ return TA::Range{P}; });
  zero_result(C, Mo * No);
  TA::detail::arena_strided_dgemm_ce_ce_left(C, L, R, Mo, No, nK,
      blas::NoTranspose, blas::NoTranspose, 1.0);
  auto ref = ref_ce_ce_left_sparse(L, R, Mo, No, nK, P, 1.0);
  check_ce_ce(C, ref, 1, Mo, No, P);
}

// T12: T2 mirrored for the LEFT kernel: single interior hole along m (m=4).
BOOST_AUTO_TEST_CASE(ce_ce_left_seg_single_interior_hole) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 8, No = 1, nK = 2, P = 3, Q = 4;
  auto lhole = [&](std::size_t o) { return (o / nK) == 4; };  // L[m=4,*]
  auto chole = [&](std::size_t o) { return (o / No) == 4; };  // C[m=4]
  Outer L = make_sparse(TA::Range{Mo, nK}, 1,
                        [&](std::size_t){ return TA::Range{Q}; }, lhole, 1.0);
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{nK, No}, 1, [&](std::size_t){ return TA::Range{Q, P}; });
  for (std::size_t o = 0; o < R.range().volume(); ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  Outer C = make_sparse(TA::Range{Mo, No}, 1,
                        [&](std::size_t){ return TA::Range{P}; }, chole, 0.0);
  zero_result(C, Mo * No);
  TA::detail::arena_strided_dgemm_ce_ce_left(C, L, R, Mo, No, nK,
      blas::NoTranspose, blas::NoTranspose, 1.0);
  auto ref = ref_ce_ce_left_sparse(L, R, Mo, No, nK, P, 1.0);
  check_ce_ce(C, ref, 1, Mo, No, P);
}

// ---- Segment-count assertions (need -DTA_STRIDED_DGEMM_COUNT) ----

// T13: dense Mmu=6, nK=2 -> 2 segment-GEMMs (one full-run segment per k).
BOOST_AUTO_TEST_CASE(ce_ce_seg_count_dense) {
#ifdef TA_STRIDED_DGEMM_COUNT
  namespace blas = TA::math::blas;
  const std::size_t Mo = 1, Mmu = 6, nK = 2, P = 3, Q = 4;
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P, Q};}, 1.0);
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu, nK}, 1, [&](std::size_t){ return TA::Range{Q}; });
  for (std::size_t o = 0; o < R.range().volume(); ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu}, 1, [&](std::size_t){ return TA::Range{P}; });
  zero_result(C, Mmu);
  TA::detail::g_strided_dgemm_ce_ce_right_calls = 0;
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, Mo, Mmu, nK,
      blas::NoTranspose, blas::Transpose, 1.0);
  BOOST_CHECK_EQUAL(
      TA::detail::g_strided_dgemm_ce_ce_right_calls.load(), std::size_t{2});
#else
  BOOST_TEST_MESSAGE("ce_ce_seg_count_dense skipped (no TA_STRIDED_DGEMM_COUNT)");
#endif
}

// T14: T5 pattern -> 3 segment-GEMMs total (k=0: {0},{2}=2; k=1: {1,2}=1).
BOOST_AUTO_TEST_CASE(ce_ce_seg_count_per_k_misaligned) {
#ifdef TA_STRIDED_DGEMM_COUNT
  namespace blas = TA::math::blas;
  const std::size_t Mo = 1, Mmu = 3, nK = 2, P = 3, Q = 4;
  auto rhole = [&](std::size_t o) { return o == 2 || o == 1; };
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P, Q};}, 1.0);
  Outer R = make_sparse(TA::Range{Mmu, nK}, 1,
                        [&](std::size_t){ return TA::Range{Q}; }, rhole, 2.0);
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu}, 1, [&](std::size_t){ return TA::Range{P}; });
  zero_result(C, Mmu);
  TA::detail::g_strided_dgemm_ce_ce_right_calls = 0;
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, Mo, Mmu, nK,
      blas::NoTranspose, blas::Transpose, 1.0);
  BOOST_CHECK_EQUAL(
      TA::detail::g_strided_dgemm_ce_ce_right_calls.load(), std::size_t{3});
#else
  BOOST_TEST_MESSAGE(
      "ce_ce_seg_count_per_k_misaligned skipped (no TA_STRIDED_DGEMM_COUNT)");
#endif
}

// T15: T4 pattern (even present, Mmu=6, nK=1) -> 3 segment-GEMMs (M=1 each).
BOOST_AUTO_TEST_CASE(ce_ce_seg_count_alternating) {
#ifdef TA_STRIDED_DGEMM_COUNT
  namespace blas = TA::math::blas;
  const std::size_t Mo = 1, Mmu = 6, nK = 1, P = 3, Q = 4;
  auto rhole = [&](std::size_t o) { return ((o / nK) % 2) == 1; };
  auto chole = [&](std::size_t o) { return (o % 2) == 1; };
  Outer L = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P, Q};}, 1.0);
  Outer R = make_sparse(TA::Range{Mmu, nK}, 1,
                        [&](std::size_t){ return TA::Range{Q}; }, rhole, 2.0);
  Outer C = make_sparse(TA::Range{Mmu}, 1,
                        [&](std::size_t){ return TA::Range{P}; }, chole, 0.0);
  zero_result(C, Mmu);
  TA::detail::g_strided_dgemm_ce_ce_right_calls = 0;
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, Mo, Mmu, nK,
      blas::NoTranspose, blas::Transpose, 1.0);
  BOOST_CHECK_EQUAL(
      TA::detail::g_strided_dgemm_ce_ce_right_calls.load(), std::size_t{3});
#else
  BOOST_TEST_MESSAGE(
      "ce_ce_seg_count_alternating skipped (no TA_STRIDED_DGEMM_COUNT)");
#endif
}


// T16: _left mirror of T14 -- per-k misaligned along m yields 3 segment-GEMMs
// (k=0: m in {0},{2} = 2 segs; k=1: m in {1,2} = 1 seg). Proves the LEFT kernel
// segments rather than dropping the whole run to the scalar fallback.
BOOST_AUTO_TEST_CASE(ce_ce_left_seg_count_per_k_misaligned) {
#ifdef TA_STRIDED_DGEMM_COUNT
  namespace blas = TA::math::blas;
  const std::size_t Mo = 3, No = 1, nK = 2, P = 3, Q = 4;
  // L outer (Mo,nK) ordinal = m*nK + k. Present set: k=0 -> m{0,2}; k=1 -> m{1,2}.
  // Holes: (m=1,k=0)=ord2, (m=0,k=1)=ord1.
  auto lhole = [&](std::size_t o) { return o == 2 || o == 1; };
  Outer L = make_sparse(TA::Range{Mo, nK}, 1,
                        [&](std::size_t){ return TA::Range{Q}; }, lhole, 1.0);
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{nK, No}, 1, [&](std::size_t){ return TA::Range{Q, P}; });
  for (std::size_t o = 0; o < R.range().volume(); ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, No}, 1, [&](std::size_t){ return TA::Range{P}; });
  zero_result(C, Mo * No);
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
  TA::detail::arena_strided_dgemm_ce_ce_left(C, L, R, Mo, No, nK,
      blas::NoTranspose, blas::NoTranspose, 1.0);
  BOOST_CHECK_EQUAL(TA::detail::g_strided_dgemm_ce_ce_left_calls.load(),
                    std::size_t{3});
  // also correctness against the sparsity-aware reference.
  auto ref = ref_ce_ce_left_sparse(L, R, Mo, No, nK, P, 1.0);
  check_ce_ce(C, ref, 1, Mo, No, P);
#else
  BOOST_TEST_MESSAGE(
      "ce_ce_left_seg_count_per_k_misaligned skipped (no TA_STRIDED_DGEMM_COUNT)");
#endif
}

// T7 mirror for the LEFT kernel: right_inner_transposed=true with a hole along
// m. Pins the transb=T / ldb=Q segment path and its scalar-fallback indexing.
BOOST_AUTO_TEST_CASE(ce_ce_left_seg_transposed_inner) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 8, No = 1, nK = 2, P = 3, Q = 4;
  auto lhole = [&](std::size_t o) { return (o / nK) == 3; };  // L[m=3,*]
  auto chole = [&](std::size_t o) { return (o / No) == 3; };  // C[m=3]
  Outer L = make_sparse(TA::Range{Mo, nK}, 1,
                        [&](std::size_t){ return TA::Range{Q}; }, lhole, 1.0);
  // R stored P x Q (matrix_transpose layout): (b1,a4) row-major.
  Outer R = make_filled(TA::Range{nK, No},
                        [&](std::size_t){ return TA::Range{P, Q}; }, 2.0);
  Outer C = make_sparse(TA::Range{Mo, No}, 1,
                        [&](std::size_t){ return TA::Range{P}; }, chole, 0.0);
  zero_result(C, Mo * No);
  TA::detail::arena_strided_dgemm_ce_ce_left(
      C, L, R, Mo, No, nK, blas::NoTranspose, blas::NoTranspose, 1.0,
      /*right_inner_transposed=*/true);
  auto ref = ref_ce_ce_left_sparse(L, R, Mo, No, nK, P, 1.0, 1, /*rt=*/true);
  check_ce_ce(C, ref, 1, Mo, No, P);
}

// Non-canonical outer orientation for _right: left_op=Transpose exercises the
// transposed l_off branch (L physically laid out (k,m) = k*Mo+m). Dense -> one
// full-run segment per k; compared to an inline reference over logical operands.
BOOST_AUTO_TEST_CASE(ce_ce_right_seg_left_op_transpose) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 2, Mmu = 3, nK = 2, P = 2, Q = 3;
  Outer L = make_filled(TA::Range{nK, Mo},
                        [&](std::size_t){ return TA::Range{P, Q}; }, 1.0);
  Outer R = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mmu, nK}, 1, [&](std::size_t){ return TA::Range{Q}; });
  for (std::size_t o = 0; o < R.range().volume(); ++o)
    for (std::size_t e = 0; e < R.data()[o].size(); ++e)
      R.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, Mmu}, 1, [&](std::size_t){ return TA::Range{P}; });
  zero_result(C, Mo * Mmu);
  TA::detail::arena_strided_dgemm_ce_ce_right(C, L, R, Mo, Mmu, nK,
      blas::Transpose, blas::Transpose, 1.0);
  // C[m,mu](a1) = sum_k sum_a4 L_phys[k*Mo+m](a1,a4) * R[mu,k](a4)
  for (std::size_t m = 0; m < Mo; ++m)
    for (std::size_t mu = 0; mu < Mmu; ++mu) {
      const double* got = C.data()[m * Mmu + mu].data();
      for (std::size_t a1 = 0; a1 < P; ++a1) {
        double acc = 0.0;
        for (std::size_t k = 0; k < nK; ++k) {
          const double* l = L.data()[k * Mo + m].data();   // P x Q
          const double* r = R.data()[mu * nK + k].data();  // Q
          for (std::size_t a4 = 0; a4 < Q; ++a4) acc += l[a1 * Q + a4] * r[a4];
        }
        BOOST_CHECK_CLOSE(got[a1], acc, 1e-10);
      }
    }
}

// Non-canonical outer orientation for _left: right_op=Transpose exercises the
// transposed r_off branch (R physically laid out (n,k) = n*nK+k). Dense -> one
// full-run segment per k; compared to an inline reference over logical operands.
BOOST_AUTO_TEST_CASE(ce_ce_left_seg_right_op_transpose) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 3, No = 2, nK = 2, P = 2, Q = 3;
  Outer L = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, nK}, 1, [&](std::size_t){ return TA::Range{Q}; });
  for (std::size_t o = 0; o < L.range().volume(); ++o)
    for (std::size_t e = 0; e < L.data()[o].size(); ++e)
      L.data()[o].data()[e] = 1.0 + 0.01 * o + e;
  // R physical (n slow, k fast) = n*nK+k, inner Q x P (canonical a4,b1).
  Outer R = make_filled(TA::Range{No, nK},
                        [&](std::size_t){ return TA::Range{Q, P}; }, 2.0);
  Outer C = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, No}, 1, [&](std::size_t){ return TA::Range{P}; });
  zero_result(C, Mo * No);
  TA::detail::arena_strided_dgemm_ce_ce_left(C, L, R, Mo, No, nK,
      blas::NoTranspose, blas::Transpose, 1.0);
  // C[m,n](b1) = sum_k sum_a4 L[m*nK+k](a4) * R_phys[n*nK+k](a4,b1)
  for (std::size_t m = 0; m < Mo; ++m)
    for (std::size_t n = 0; n < No; ++n) {
      const double* got = C.data()[m * No + n].data();
      for (std::size_t b1 = 0; b1 < P; ++b1) {
        double acc = 0.0;
        for (std::size_t k = 0; k < nK; ++k) {
          const double* l = L.data()[m * nK + k].data();   // Q
          const double* r = R.data()[n * nK + k].data();   // Q x P
          for (std::size_t a4 = 0; a4 < Q; ++a4) acc += l[a4] * r[a4 * P + b1];
        }
        BOOST_CHECK_CLOSE(got[b1], acc, 1e-10);
      }
    }
}

// L1: left mirror of T2/T9 with a hole and a non-unit factor. Hole at m=4 in
// BOTH C and L (all k) -> two m-segments [0,4),[5,8); C[4] absent; factor=2.5.
BOOST_AUTO_TEST_CASE(ce_ce_left_seg_hole_and_factor) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 8, No = 1, nK = 2, P = 3, Q = 4;
  const double factor = 2.5;
  auto lhole = [&](std::size_t o) { return (o / nK) == 4; };  // L[m=4,*]
  auto chole = [&](std::size_t o) { return o == 4; };         // C[m=4,n=0]
  Outer L = make_sparse(TA::Range{Mo, nK}, 1,
                        [&](std::size_t){ return TA::Range{Q}; }, lhole, 1.0);
  Outer R = make_filled(TA::Range{nK, No},
                        [&](std::size_t){ return TA::Range{Q, P}; }, 2.0);
  Outer C = make_sparse(TA::Range{Mo, No}, 1,
                        [&](std::size_t){ return TA::Range{P}; }, chole, 0.0);
  zero_result(C, C.range().volume());
  TA::detail::arena_strided_dgemm_ce_ce_left(C, L, R, Mo, No, nK,
      blas::NoTranspose, blas::NoTranspose, factor);
  auto ref = ref_ce_ce_left_sparse(L, R, Mo, No, nK, P, factor);
  check_ce_ce(C, ref, 1, Mo, No, P);
}

// L2: left mirror of T6. The SINGLE-CELL operand R[k=1,n] is absent for all n
// -> the kernel's `if (!rk) continue;` skips k=1 entirely (beta=1); other k
// contribute. (The complementary "strided operand absent" guard is L2b below.)
BOOST_AUTO_TEST_CASE(ce_ce_left_seg_absent_k) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 5, No = 1, nK = 3, P = 3, Q = 4;
  // R[k=1,n] absent for all n -> k=1 skipped entirely (beta=1).
  auto rhole = [&](std::size_t o) { return (o / No) == 1; };  // R[k=1,*]
  Outer L = make_filled(TA::Range{Mo, nK},
                        [&](std::size_t){ return TA::Range{Q}; }, 1.0);
  Outer R = make_sparse(TA::Range{nK, No}, 1,
                        [&](std::size_t){ return TA::Range{Q, P}; }, rhole, 2.0);
  Outer C = make_filled(TA::Range{Mo, No},
                        [&](std::size_t){ return TA::Range{P}; }, 0.0);
  zero_result(C, C.range().volume());
  TA::detail::arena_strided_dgemm_ce_ce_left(C, L, R, Mo, No, nK,
      blas::NoTranspose, blas::NoTranspose, 1.0);
  auto ref = ref_ce_ce_left_sparse(L, R, Mo, No, nK, P, 1.0);
  check_ce_ce(C, ref, 1, Mo, No, P);
}

// L2b: the STRIDED operand is entirely absent for one k. All L[m,k=1] are holes
// while R[k=1,n] stays present -> Q cannot be discovered for k=1, so the kernel's
// `if (Q <= 0) continue;` skips it; other k contribute. The reference skips the
// same k (its `!lk` branch), so results stay exact.
BOOST_AUTO_TEST_CASE(ce_ce_left_seg_strided_operand_absent_k) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 5, No = 1, nK = 3, P = 3, Q = 4;
  // L[m,k=1] absent for all m (ordinal o = m*nK + k, so k==1).
  auto lhole = [&](std::size_t o) { return (o % nK) == 1; };
  Outer L = make_sparse(TA::Range{Mo, nK}, 1,
                        [&](std::size_t){ return TA::Range{Q}; }, lhole, 1.0);
  Outer R = make_filled(TA::Range{nK, No},
                        [&](std::size_t){ return TA::Range{Q, P}; }, 2.0);
  Outer C = make_filled(TA::Range{Mo, No},
                        [&](std::size_t){ return TA::Range{P}; }, 0.0);
  zero_result(C, C.range().volume());
  TA::detail::arena_strided_dgemm_ce_ce_left(C, L, R, Mo, No, nK,
      blas::NoTranspose, blas::NoTranspose, 1.0);
  auto ref = ref_ce_ce_left_sparse(L, R, Mo, No, nK, P, 1.0);
  check_ce_ce(C, ref, 1, Mo, No, P);
}

// L3: left mirror of T10 (defensive size mismatch). One L[m,k] present but the
// wrong inner size (Q+1) -> that (m,k) is skipped by the per-cell guard, the
// rest stay exact, no crash.
BOOST_AUTO_TEST_CASE(ce_ce_left_seg_size_mismatch_defensive) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 4, No = 1, nK = 2, P = 3, Q = 4;
  Outer L = TA::detail::arena_outer_init<Outer>(
      TA::Range{Mo, nK}, 1, [&](std::size_t o) {
        // L[m=2,k=0] (ordinal 2*nK+0 = 4) has size Q+1; all others size Q.
        return (o == 4) ? TA::Range{Q + 1} : TA::Range{Q};
      });
  for (std::size_t o = 0; o < L.range().volume(); ++o)
    for (std::size_t e = 0; e < L.data()[o].size(); ++e)
      L.data()[o].data()[e] = 1.0 + 0.01 * o + e;
  Outer R = make_filled(TA::Range{nK, No},
                        [&](std::size_t){ return TA::Range{Q, P}; }, 2.0);
  Outer C = make_filled(TA::Range{Mo, No},
                        [&](std::size_t){ return TA::Range{P}; }, 0.0);
  zero_result(C, C.range().volume());
  TA::detail::arena_strided_dgemm_ce_ce_left(C, L, R, Mo, No, nK,
      blas::NoTranspose, blas::NoTranspose, 1.0);
  // Reference skips any k where L[m,k].size() != Q (R[k,n].size()==P*Q gate).
  auto ref = ref_ce_ce_left_sparse(L, R, Mo, No, nK, P, 1.0);
  check_ce_ce(C, ref, 1, Mo, No, P);
}

// L4: left mirror of T8 (per-batch independent segmentation). NB=2; batch 0
// dense, batch 1 has a single interior hole at m=2.
BOOST_AUTO_TEST_CASE(ce_ce_left_seg_multi_batch_sparse) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 4, No = 1, nK = 2, P = 3, Q = 4, NB = 2;
  // hole at batch-1 m=2: ordinal in L is b*Mo*nK + m*nK + k; in C b*Mo*No+m*No+n.
  auto lhole = [&](std::size_t o) {
    const std::size_t b = o / (Mo * nK);
    const std::size_t m = (o % (Mo * nK)) / nK;
    return b == 1 && m == 2;
  };
  auto chole = [&](std::size_t o) {
    const std::size_t b = o / (Mo * No);
    const std::size_t m = (o % (Mo * No)) / No;
    return b == 1 && m == 2;
  };
  Outer L = make_sparse(TA::Range{Mo, nK}, NB,
                        [&](std::size_t){ return TA::Range{Q}; }, lhole, 1.0);
  // R is shared per batch; build NB batches explicitly (make_filled gives nbatch=1).
  Outer Rb = TA::detail::arena_outer_init<Outer>(
      TA::Range{nK, No}, NB, [&](std::size_t){ return TA::Range{Q, P}; });
  for (std::size_t o = 0; o < Rb.range().volume() * NB; ++o)
    for (std::size_t e = 0; e < Rb.data()[o].size(); ++e)
      Rb.data()[o].data()[e] = 2.0 + 0.01 * o + e;
  Outer C = make_sparse(TA::Range{Mo, No}, NB,
                        [&](std::size_t){ return TA::Range{P}; }, chole, 0.0);
  zero_result(C, C.range().volume() * NB);
  TA::detail::arena_strided_dgemm_ce_ce_left(C, L, Rb, Mo, No, nK,
      blas::NoTranspose, blas::NoTranspose, 1.0);
  auto ref = ref_ce_ce_left_sparse(L, Rb, Mo, No, nK, P, 1.0, NB);
  check_ce_ce(C, ref, NB, Mo, No, P);
}

// L5: an entirely-absent result run must be a clean no-op (P<=0 early continue)
// for BOTH orientations -- no writes, no crash.
BOOST_AUTO_TEST_CASE(ce_ce_seg_all_absent_run_is_noop) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 1, Mmu = 4, No = 1, nK = 2, P = 3, Q = 4;
  // RIGHT: every C[mu] absent.
  Outer Lr = make_filled(TA::Range{nK}, [&](std::size_t){return TA::Range{P,Q};}, 1.0);
  Outer Rr = make_filled(TA::Range{Mmu, nK}, [&](std::size_t){return TA::Range{Q};}, 2.0);
  Outer Cr = make_sparse(TA::Range{Mmu}, 1,
                         [&](std::size_t){ return TA::Range{P}; },
                         [](std::size_t){ return true; }, 0.0);  // all holes
  BOOST_REQUIRE_NO_THROW(TA::detail::arena_strided_dgemm_ce_ce_right(
      Cr, Lr, Rr, Mo, Mmu, nK, blas::NoTranspose, blas::Transpose, 1.0));
  for (std::size_t o = 0; o < Cr.range().volume(); ++o)
    BOOST_CHECK(!Cr.data()[o]);  // all stay absent
  // LEFT: every C[m,n] absent.
  Outer Ll = make_filled(TA::Range{Mo, nK}, [&](std::size_t){return TA::Range{Q};}, 1.0);
  Outer Rl = make_filled(TA::Range{nK, No}, [&](std::size_t){return TA::Range{Q,P};}, 2.0);
  Outer Cl = make_sparse(TA::Range{Mo, No}, 1,
                         [&](std::size_t){ return TA::Range{P}; },
                         [](std::size_t){ return true; }, 0.0);
  BOOST_REQUIRE_NO_THROW(TA::detail::arena_strided_dgemm_ce_ce_left(
      Cl, Ll, Rl, Mo, No, nK, blas::NoTranspose, blas::NoTranspose, 1.0));
  for (std::size_t o = 0; o < Cl.range().volume(); ++o)
    BOOST_CHECK(!Cl.data()[o]);
}

// K1: the kill switch is a FAITHFUL equivalent. On a hole-containing,
// per-k-misaligned pattern, the segment-walker path (switch off) and the
// per-cell path (switch on) must produce bitwise-identical results -- this both
// proves the per-cell reference the bench times is correct AND that the switch
// changes only the evaluation strategy, not the math. RIGHT orientation.
BOOST_AUTO_TEST_CASE(ce_ce_seg_killswitch_matches_right) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 1, Mmu = 8, nK = 3, P = 3, Q = 4;
  auto rhole = [&](std::size_t o) {
    const std::size_t mu = o / nK, k = o % nK;
    return ((mu + k) % 3) == 0;  // staggered per k
  };
  auto chole = [&](std::size_t o) {  // C[mu] present iff present for some k
    for (std::size_t k = 0; k < nK; ++k)
      if (!(((o + k) % 3) == 0)) return false;
    return true;
  };
  auto build = [&]() {
    Outer L = make_filled(TA::Range{nK},
                          [&](std::size_t){ return TA::Range{P, Q}; }, 1.0);
    Outer R = make_sparse(TA::Range{Mmu, nK}, 1,
                          [&](std::size_t){ return TA::Range{Q}; }, rhole, 2.0);
    Outer C = make_sparse(TA::Range{Mmu}, 1,
                          [&](std::size_t){ return TA::Range{P}; }, chole, 0.0);
    zero_result(C, C.range().volume());
    return std::make_tuple(std::move(L), std::move(R), std::move(C));
  };
  auto [Ls, Rs, Cs] = build();  // switch OFF (segment walker)
  TA::detail::ce_ce_strided_disabled() = false;
  TA::detail::arena_strided_dgemm_ce_ce_right(Cs, Ls, Rs, Mo, Mmu, nK,
      blas::NoTranspose, blas::Transpose, 1.0);
  auto [Lp, Rp, Cp] = build();  // switch ON (per-cell)
  TA::detail::ce_ce_strided_disabled() = true;
  TA::detail::arena_strided_dgemm_ce_ce_right(Cp, Lp, Rp, Mo, Mmu, nK,
      blas::NoTranspose, blas::Transpose, 1.0);
  TA::detail::ce_ce_strided_disabled() = false;  // restore production default
  for (std::size_t o = 0; o < Cs.range().volume(); ++o) {
    BOOST_REQUIRE_EQUAL(bool(Cs.data()[o]), bool(Cp.data()[o]));
    if (!Cs.data()[o]) continue;
    BOOST_REQUIRE_EQUAL(Cs.data()[o].size(), Cp.data()[o].size());
    for (std::size_t a1 = 0; a1 < Cs.data()[o].size(); ++a1)
      BOOST_CHECK_CLOSE(Cs.data()[o].data()[a1], Cp.data()[o].data()[a1], 1e-12);
  }
}

// K2: same faithful-equivalent check, LEFT orientation, per-k misaligned on L.
BOOST_AUTO_TEST_CASE(ce_ce_seg_killswitch_matches_left) {
  namespace blas = TA::math::blas;
  const std::size_t Mo = 8, No = 1, nK = 3, P = 3, Q = 4;
  auto lhole = [&](std::size_t o) {
    const std::size_t m = o / nK, k = o % nK;
    return ((m + k) % 3) == 0;
  };
  auto chole = [&](std::size_t o) {
    for (std::size_t k = 0; k < nK; ++k)
      if (!(((o + k) % 3) == 0)) return false;
    return true;
  };
  auto build = [&]() {
    Outer L = make_sparse(TA::Range{Mo, nK}, 1,
                          [&](std::size_t){ return TA::Range{Q}; }, lhole, 1.0);
    Outer R = make_filled(TA::Range{nK, No},
                          [&](std::size_t){ return TA::Range{Q, P}; }, 2.0);
    Outer C = make_sparse(TA::Range{Mo, No}, 1,
                          [&](std::size_t){ return TA::Range{P}; }, chole, 0.0);
    zero_result(C, C.range().volume());
    return std::make_tuple(std::move(L), std::move(R), std::move(C));
  };
  auto [Ls, Rs, Cs] = build();
  TA::detail::ce_ce_strided_disabled() = false;
  TA::detail::arena_strided_dgemm_ce_ce_left(Cs, Ls, Rs, Mo, No, nK,
      blas::NoTranspose, blas::NoTranspose, 1.0);
  auto [Lp, Rp, Cp] = build();
  TA::detail::ce_ce_strided_disabled() = true;
  TA::detail::arena_strided_dgemm_ce_ce_left(Cp, Lp, Rp, Mo, No, nK,
      blas::NoTranspose, blas::NoTranspose, 1.0);
  TA::detail::ce_ce_strided_disabled() = false;
  for (std::size_t o = 0; o < Cs.range().volume(); ++o) {
    BOOST_REQUIRE_EQUAL(bool(Cs.data()[o]), bool(Cp.data()[o]));
    if (!Cs.data()[o]) continue;
    for (std::size_t a1 = 0; a1 < Cs.data()[o].size(); ++a1)
      BOOST_CHECK_CLOSE(Cs.data()[o].data()[a1], Cp.data()[o].data()[a1], 1e-12);
  }
}

BOOST_AUTO_TEST_SUITE_END()
