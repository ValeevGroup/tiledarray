#include "TiledArray/expressions/permopt.h"
#include "tiledarray.h"
#include "unit_test_config.h"

#include <TiledArray/tensor/arena_tensor.h>
#include <TiledArray/tensor/arena_kernels.h>   // detail::arena_outer_init, arena_compact
#include <TiledArray/einsum/tiledarray.h>
#include <TiledArray/conversions/foreach.h>     // TA::foreach (used by reorder)
#include <TiledArray/expressions/permopt.h>     // strided_canonicalize_disabled()
#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <string>
#include <vector>

namespace TA = TiledArray;
using TA::expressions::GEMMPermutationOptimizer;
using TA::expressions::IndexList;
using TA::expressions::PermutationType;

namespace {
// Types and conventions MIRROR the existing einsum_tot arena-ToT tests in
// tests/einsum.cpp (ArenaTensor<double, TA::Range> + DensePolicy + arena_outer_init
// builder + owning-ToT cross-oracle), so this suite is consistent with the
// proven-firing tests there and reuses their independent ground-truth pattern.
// A standalone file is justified: this is a focused outer-ordering scramble
// matrix and einsum.cpp is already ~3500 lines; keeping the matrix separate
// keeps it readable while deliberately adopting einsum.cpp's types and oracle.
using ArenaInner = TA::ArenaTensor<double, TA::Range>;
using ArenaOuter = TA::Tensor<ArenaInner>;
using ArrayToT   = TA::DistArray<ArenaOuter, TA::DensePolicy>;          // arena (view cells)
using OwnInner   = TA::Tensor<double>;
using OwnOuter   = TA::Tensor<OwnInner>;
using OwnArr     = TA::DistArray<OwnOuter, TA::DensePolicy>;            // owning ToT (ground truth)
using ArrayT     = TA::DistArray<TA::Tensor<double>, TA::DensePolicy>;  // plain

TA::TiledRange1 tr1(std::size_t n, std::size_t ts) {
  if (ts == 0 || ts > n) ts = n;
  std::vector<std::size_t> b;
  for (std::size_t x = 0; x < n; x += ts) b.push_back(x);
  b.push_back(n);
  return TA::TiledRange1(b.begin(), b.end());
}

// CRITICAL for test validity: operand data MUST depend on the outer element
// index. If every cell held the same values (the naive `1.0 + 1e-3*e` fill),
// the operand would be symmetric under exactly the permutations the suite
// scrambles, and the tests would pass even if canonicalization were broken or a
// no-op. `outer_seed` produces a distinct scalar per outer coordinate so that
// permuting modes produces an observably different physical layout.
template <typename Index>
double outer_seed(const Index& oix) {
  double seed = 0.0, f = 1.0;
  for (auto c : oix) { seed += double(c) * f; f *= 31.0; }
  return seed;
}
inline double cell_val(double outer_s, std::size_t e) {
  return 1.0 + 1e-3 * double(e) + 1e-2 * outer_s;
}

// Plain dense tensor (DensePolicy), filled by ABSOLUTE element coordinate (so a
// permuted copy represents the same tensor in a different mode order).
ArrayT make_plain(TA::World& w, const TA::TiledRange& tr) {
  ArrayT g(w, tr);
  g.init_tiles([](const TA::Range& r) {
    TA::Tensor<double> t(r);
    for (const auto& idx : r) t(idx) = 1.0 + 1e-2 * outer_seed(idx);
    return t;
  });
  return g;
}

// Arena ToT, single-page by construction via arena_outer_init (no separate
// compaction needed -- mirrors tests/einsum.cpp). `inner` is the per-cell inner
// range; data varies per outer cell (outer_seed) so outer permutations are
// observable, making ordering the ONLY possible strided blocker.
ArrayToT make_arena(TA::World& w, const TA::TiledRange& tr, const TA::Range& inner) {
  ArrayToT x(w, tr);
  x.init_tiles([inner](const TA::Range& t_outer) {
    ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(
        t_outer, 1, [inner](std::size_t) { return inner; });
    std::size_t o = 0;
    for (const auto& idx : t_outer) {
      ArenaInner& c = t.data()[o++];
      if (!c) continue;
      const double s = outer_seed(idx);
      for (std::size_t e = 0; e < c.size(); ++e) c.data()[e] = cell_val(s, e);
    }
    return t;
  });
  return x;
}

// Owning-ToT twin with IDENTICAL data: an independent (non-strided) code path,
// the numerical GROUND TRUTH for the arena result (the einsum_tot oracle).
OwnArr make_own(TA::World& w, const TA::TiledRange& tr, const TA::Range& inner) {
  OwnArr x(w, tr);
  x.init_tiles([inner](const TA::Range& t_outer) {
    OwnOuter t(t_outer);
    std::size_t o = 0;
    for (const auto& idx : t_outer) {
      OwnInner cell(inner);
      const double s = outer_seed(idx);
      for (std::size_t e = 0; e < cell.size(); ++e) cell.data()[e] = cell_val(s, e);
      t.data()[o++] = cell;
    }
    return t;
  });
  return x;
}

// Convenience: 1-D and 2-D inner extents, arena and owning twins.
ArrayToT make_arena_tot (TA::World& w, const TA::TiledRange& tr, long a)           { return make_arena(w, tr, TA::Range{a}); }
ArrayToT make_arena_tot2(TA::World& w, const TA::TiledRange& tr, long e0, long e1) { return make_arena(w, tr, TA::Range{e0, e1}); }
OwnArr   make_own_tot   (TA::World& w, const TA::TiledRange& tr, long a)           { return make_own(w, tr, TA::Range{a}); }
OwnArr   make_own_tot2  (TA::World& w, const TA::TiledRange& tr, long e0, long e1) { return make_own(w, tr, TA::Range{e0, e1}); }

// PRIMARY correctness oracle: max RELATIVE element-wise error between arena
// result A and owning ground truth B, allreduced. Relative (with a unit floor on
// the denominator so near-zero references behave like an absolute check) so the
// SAME tolerance is valid regardless of result magnitude: a high-rank outer
// shape (e.g. hce+ce with 5-6 outer modes) sums to ~1e10, where a single double
// rounding is ~1e-6 ABSOLUTE but ~1e-16 RELATIVE -- an absolute oracle would
// false-fail there. Templated so an arena result (ArrayToT) is compared against
// the owning ground truth (OwnArr). Catches mislaid elements a scalar sum would
// miss. Fetches B's tile by index via find() so it works at np>1 even if pmaps
// differ.
template <typename ArrA, typename ArrB>
double array_max_reldiff(const ArrA& A, const ArrB& B) {
  double d = 0.0;
  for (auto it = A.begin(); it != A.end(); ++it) {
    const auto a_tile = (*it).get();
    if (a_tile.empty()) continue;
    const auto b_tile = B.find(it.index()).get();   // local or remote copy
    if (a_tile.size() != b_tile.size()) { d = std::max(d, 1.0); continue; }
    for (std::size_t o = 0; o < a_tile.size(); ++o) {
      const auto& ac = a_tile[o];
      const auto& bc = b_tile[o];
      if (ac.size() != bc.size()) { d = std::max(d, 1.0); continue; }
      for (std::size_t e = 0; e < ac.size(); ++e) {
        const double av = double(ac.data()[e]), bv = double(bc.data()[e]);
        d = std::max(d, std::abs(av - bv) / std::max(1.0, std::abs(bv)));
      }
    }
  }
  A.world().gop.max(&d, 1);
  return d;
}

// All distinct orderings of a comma-joined label string, capped to keep the
// matrix bounded; always includes the input order and its full reverse.
std::vector<std::string> orderings(std::string labels, std::size_t cap = 8) {
  std::vector<std::string> toks;
  for (std::size_t p = 0, c; p != std::string::npos; p = (c==std::string::npos?c:c+1)) {
    c = labels.find(',', p);
    toks.push_back(labels.substr(p, c==std::string::npos?c:c-p));
    if (c == std::string::npos) break;
  }
  std::sort(toks.begin(), toks.end());
  std::vector<std::string> out;
  do {
    std::string s; for (std::size_t i=0;i<toks.size();++i) s += (i?",":"") + toks[i];
    out.push_back(s);
    if (out.size() >= cap) break;
  } while (std::next_permutation(toks.begin(), toks.end()));
  return out;
}

// Split a bipartite annotation "outer;inner" into its parts. inner_of keeps the
// leading ';' (or "" if no inner).
std::string outer_of(const std::string& ann) {
  auto p = ann.find(';'); return p == std::string::npos ? ann : ann.substr(0, p);
}
std::string inner_of(const std::string& ann) {
  auto p = ann.find(';'); return p == std::string::npos ? std::string() : ann.substr(p);
}

// Build a single-page arena ToT in a given full-annotation order from a
// canonical array by permuting + recompacting (math invariant, isolates
// ordering). `from`/`to` are FULL annotations; inner is unchanged between them.
ArrayToT reorder(const ArrayToT& src, const std::string& from,
                 const std::string& to) {
  if (from == to) return src;
  ArrayToT dst;
  dst(to) = src(from);            // outer permutation via expression DSL
  src.world().gop.fence();
  using TileT = typename ArrayToT::value_type;
  // foreach's non-inplace overload requires the op to return void (see
  // conversions/foreach.h:187); the brief's `-> float` signature does not
  // resolve, so write into `r` and return void (same intent: recompact each
  // tile into a fresh single-page arena tile).
  return TA::foreach(dst, [](TileT& r, const TileT& a) {
    r = TiledArray::detail::arena_compact(a);
  });
}

// lcanon/rcanon are FULL canonical annotations; los/ros are OUTER orderings.
// ref_own is the SAME contraction computed via the owning-ToT (non-strided)
// path = independent GROUND TRUTH. Every arena scramble (canonicalization ON,
// the default) must match it element-wise and (under the macro) FIRE the strided
// kernel. counter/counter2: pass both ce_ce_{right,left} for ce+ce (work may
// route to either arm); pass one (+nullptr) otherwise.
void run_scrambles(const ArrayToT& L0, const std::string& lcanon,
                   const ArrayToT& R0, const std::string& rcanon,
                   const std::string& result, const OwnArr& ref_own,
                   const std::vector<std::string>& los,
                   const std::vector<std::string>& ros,
                   std::atomic<std::size_t>* counter,
                   std::atomic<std::size_t>* counter2 = nullptr) {
  auto& w = L0.world();
  const std::string linner = inner_of(lcanon), rinner = inner_of(rcanon);
  for (const auto& lo : los)
    for (const auto& ro : ros) {
      const std::string lfull = lo + linner, rfull = ro + rinner;
      ArrayToT L = reorder(L0, lcanon, lfull);
      ArrayToT R = reorder(R0, rcanon, rfull);
      w.gop.fence();
      BOOST_TEST_INFO("left=" << lfull << " right=" << rfull);
#ifdef TA_STRIDED_DGEMM_COUNT
      if (counter) counter->store(0);
      if (counter2) counter2->store(0);
#endif
      ArrayToT C = TA::einsum(L(lfull), R(rfull), result);   // canonicalization ON
      w.gop.fence();
      BOOST_CHECK_SMALL(array_max_reldiff(C, ref_own), 1e-9);  // == owning ground truth
#ifdef TA_STRIDED_DGEMM_COUNT
      std::size_t c = (counter ? counter->load() : std::size_t{0}) +
                      (counter2 ? counter2->load() : std::size_t{0});
      w.gop.sum(&c, 1);                       // np-correct: total fires all ranks
      BOOST_CHECK_GT(c, std::size_t{0});
#endif
    }
}
}  // namespace

BOOST_AUTO_TEST_SUITE(strided_canonicalize_suite, TA_UT_LABEL_SERIAL)

// Left operand "m,i,k" against right "m,j" with target "i,k,j": the contracted
// index m is LEADING on the left => left would be matrix_transpose. With
// transpose_is_free=false it must be reported as `general` (forcing a physical
// permute to canonical NoTranspose), while transpose_is_free=true keeps the
// legacy matrix_transpose.
BOOST_AUTO_TEST_CASE(gemm_optimizer_demotes_transpose_when_not_free) {
  // Left "m,i,k" against right "m,j": contracted m is LEADING on the left, so
  // the left would be matrix_transpose. With transpose_is_free=false it must be
  // reported as `general`; with true it stays matrix_transpose (legacy).
  IndexList left("m,i,k"), right("m,j");
  GEMMPermutationOptimizer free_opt(left, right, /*prefer_left=*/true,
                                    /*transpose_is_free=*/true);
  GEMMPermutationOptimizer forced(left, right, /*prefer_left=*/true,
                                  /*transpose_is_free=*/false);
  BOOST_CHECK(free_opt.left_permtype() == PermutationType::matrix_transpose);
  BOOST_CHECK(forced.left_permtype() == PermutationType::general);

  // No-op invariant (REQUIRED): an already-canonical operand stays `identity`
  // in BOTH modes — demotion must never turn identity into a permute. Here the
  // left "i,k,m" is already (free..., contracted...) canonical for right "m,j".
  IndexList lcanon("i,k,m");
  GEMMPermutationOptimizer canon_free(lcanon, right, /*prefer_left=*/true,
                                      /*transpose_is_free=*/true);
  GEMMPermutationOptimizer canon_forced(lcanon, right, /*prefer_left=*/true,
                                        /*transpose_is_free=*/false);
  BOOST_CHECK(canon_free.left_permtype() == PermutationType::identity);
  BOOST_CHECK(canon_forced.left_permtype() == PermutationType::identity);
}

// mixed/scale: plain g * ToT I -> ToT C. NATURAL CCk order puts contracted m_
// LEADING on the plain operand (left_op would be Transpose). g_nat is the SAME
// tensor as g_can, just permuted (NOT an independent fill). GROUND TRUTH is the
// same contraction via the OWNING ToT (non-strided) path; both the canonical and
// the natural (canonicalization-ON) arena results must match it element-wise.
BOOST_AUTO_TEST_CASE(mixed_natural_order_matches_canonical) {
  auto& w = TA::get_default_world();
  const long a4 = 6;
  auto e = tr1(4, 2), mu = tr1(4, 4), i3 = tr1(4, 2), kx = tr1(6, 6);
  ArrayT   g_can = make_plain(w, TA::TiledRange{i3, kx, mu});       // g(i_3,k_,m_)
  ArrayToT I     = make_arena_tot(w, TA::TiledRange{mu, e, e}, a4); // arena ToT
  OwnArr   I_own = make_own_tot  (w, TA::TiledRange{mu, e, e}, a4); // owning twin
  w.gop.fence();
  // natural-order g is g_can permuted -> identical tensor in (m_,i_3,k_) order
  ArrayT g_nat;
  g_nat("m_,i_3,k_") = g_can("i_3,k_,m_");
  w.gop.fence();

  // independent GROUND TRUTH via the owning (non-strided) path
  OwnArr ref = TA::einsum(g_can("i_3,k_,m_"), I_own("m_,i_1,i_2;a"),
                          "i_3,i_1,i_2,k_;a");
  ArrayToT C_can = TA::einsum(g_can("i_3,k_,m_"), I("m_,i_1,i_2;a"),
                             "i_3,i_1,i_2,k_;a");
  ArrayToT C_nat = TA::einsum(g_nat("m_,i_3,k_"), I("m_,i_1,i_2;a"),
                             "i_3,i_1,i_2,k_;a");
  w.gop.fence();
  BOOST_CHECK_SMALL(array_max_reldiff(C_can, ref), 1e-9);   // arena == owning
  BOOST_CHECK_SMALL(array_max_reldiff(C_nat, ref), 1e-9);   // canonicalized == owning

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_scale_strided_calls[1].store(0);  // t_x_tot
  ArrayToT C_fire = TA::einsum(g_nat("m_,i_3,k_"), I("m_,i_1,i_2;a"),
                              "i_3,i_1,i_2,k_;a");
  w.gop.fence();
  std::size_t fires = TA::detail::g_scale_strided_calls[1].load();
  w.gop.sum(&fires, 1);                                        // np-correct
  BOOST_CHECK_GT(fires, std::size_t{0});                       // canonicalized -> FIRES
  BOOST_CHECK_SMALL(array_max_reldiff(C_fire, C_can), 1e-9);  // and stays correct
#endif

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::expressions::strided_canonicalize_disabled() = true;
  TA::detail::g_scale_strided_calls[1].store(0);
  ArrayToT C_off = TA::einsum(g_nat("m_,i_3,k_"), I("m_,i_1,i_2;a"),
                             "i_3,i_1,i_2,k_;a");
  w.gop.fence();
  std::size_t off_fires = TA::detail::g_scale_strided_calls[1].load();
  w.gop.sum(&off_fires, 1);
  BOOST_CHECK_EQUAL(off_fires, std::size_t{0});               // reverted: no fire
  BOOST_CHECK_SMALL(array_max_reldiff(C_off, C_can), 1e-9);  // still correct
  TA::expressions::strided_canonicalize_disabled() = false;   // restore
#endif
}

// ce+e: pure Contraction ToT x ToT. C(i1,i2,j1,j2; a,b) = A(i1,i2,k; a) *
// B(j1,j2,k; b) -- outer contracted k, two left externals, two right externals,
// inner outer-product (a from left, b from right). Pure-Contraction outer =>
// GEMMPermutationOptimizer => the canonicalization fix applies. Scramble ALL
// outer orderings of BOTH operands; every one must match the owning ground truth
// element-wise AND fire the strided ce+e kernel.
BOOST_AUTO_TEST_CASE(ce_e_scramble_all_outer_orders_fire) {
  auto& w = TA::get_default_world();
  auto t = tr1(4, 2), kk = tr1(4, 4);
  ArrayToT A0 = make_arena_tot(w, TA::TiledRange{t, t, kk}, /*a=*/3);  // a inner
  ArrayToT B0 = make_arena_tot(w, TA::TiledRange{t, t, kk}, /*b=*/5);  // b inner
  OwnArr A0o = make_own_tot(w, TA::TiledRange{t, t, kk}, 3);           // owning twins
  OwnArr B0o = make_own_tot(w, TA::TiledRange{t, t, kk}, 5);
  w.gop.fence();
  // independent GROUND TRUTH via the owning (non-strided) path
  OwnArr ref = TA::einsum(A0o("i1,i2,k;a"), B0o("j1,j2,k;b"), "i1,i2,j1,j2;a,b");
  w.gop.fence();
  run_scrambles(A0, "i1,i2,k;a", B0, "j1,j2,k;b", "i1,i2,j1,j2;a,b", ref,
                orderings(outer_of("i1,i2,k;a")), orderings(outer_of("j1,j2,k;b")),
#ifdef TA_STRIDED_DGEMM_COUNT
                &TA::detail::g_strided_dgemm_ce_e_calls
#else
                nullptr
#endif
  );
}

// ce+ce: outer Contraction with an INNER contraction. C(i1,i2,j1; a) =
// A(i1,i2,k; a,c) * B(j1,k; c) -- outer contracted k, left externals i1,i2,
// right external j1, inner contracted c, inner spectator a (left+result). The
// just-landed gate relaxation (commit 98d0650a) lifts the install gates at
// cont_engine.h:1472/1489, so the inner-contraction kernel should now fire for
// ALL 36 outer orderings. Scramble both operands; every one must match the
// owning ground truth AND fire the strided ce+ce kernel (right or left arm).
BOOST_AUTO_TEST_CASE(ce_ce_scramble_all_outer_orders_fire) {
  auto& w = TA::get_default_world();
  auto t = tr1(4, 2), kk = tr1(4, 4);
  ArrayToT A0 = make_arena_tot2(w, TA::TiledRange{t, t, kk}, /*a=*/3, /*c=*/4);
  ArrayToT B0 = make_arena_tot (w, TA::TiledRange{t, kk},    /*c=*/4);
  OwnArr  A0o = make_own_tot2 (w, TA::TiledRange{t, t, kk}, 3, 4);   // owning twins
  OwnArr  B0o = make_own_tot  (w, TA::TiledRange{t, kk},    4);
  w.gop.fence();
  // independent GROUND TRUTH via the owning (non-strided) path
  OwnArr ref = TA::einsum(A0o("i1,i2,k;a,c"), B0o("j1,k;c"), "i1,i2,j1;a");
  w.gop.fence();
  // pass BOTH ce_ce arms: outer canonicalization should consistently route to
  // the right (clean/vector side = B) arm, but summing both is robust either way
  run_scrambles(A0, "i1,i2,k;a,c", B0, "j1,k;c", "i1,i2,j1;a", ref,
                orderings(outer_of("i1,i2,k;a,c")), orderings(outer_of("j1,k;c")),
#ifdef TA_STRIDED_DGEMM_COUNT
                &TA::detail::g_strided_dgemm_ce_ce_right_calls,
                &TA::detail::g_strided_dgemm_ce_ce_left_calls
#else
                nullptr, nullptr
#endif
  );
}

// hce+e: General product (fused outer Hadamard h1,h2) with inner outer-product.
// C(h1,h2,i1,j1; a,b) = A(h1,h2,i1,k; a) * B(h1,h2,j1,k; b). Routes through
// GeneralPermutationOptimizer (NOT the pure-Contraction path the fix targets);
// per Hadamard batch it uses the ce+e strided kernel. R4 caveat: this asserts
// existing master behavior plus whatever the gate fix (98d0650a) enables.
BOOST_AUTO_TEST_CASE(hce_e_scramble_all_outer_orders_fire) {
  auto& w = TA::get_default_world();
  auto h = tr1(3, 3), t = tr1(4, 2), kk = tr1(4, 4);
  ArrayToT A0 = make_arena_tot(w, TA::TiledRange{h, h, t, kk}, 3);
  ArrayToT B0 = make_arena_tot(w, TA::TiledRange{h, h, t, kk}, 5);
  OwnArr A0o = make_own_tot(w, TA::TiledRange{h, h, t, kk}, 3);
  OwnArr B0o = make_own_tot(w, TA::TiledRange{h, h, t, kk}, 5);
  w.gop.fence();
  OwnArr ref = TA::einsum(A0o("h1,h2,i1,k;a"), B0o("h1,h2,j1,k;b"),
                          "h1,h2,i1,j1;a,b");
  w.gop.fence();
  // hce+e uses the ce+e kernel per Hadamard batch -> ce_e counter. (If a fire
  // is unexpectedly missing, confirm against master per the R4 caveat.)
  run_scrambles(A0, "h1,h2,i1,k;a", B0, "h1,h2,j1,k;b", "h1,h2,i1,j1;a,b", ref,
                orderings("h1,h2,i1,k"), orderings("h1,h2,j1,k"),
#ifdef TA_STRIDED_DGEMM_COUNT
                &TA::detail::g_strided_dgemm_ce_e_calls
#else
                nullptr
#endif
  );
}

// hce+ce: the user's flagship General product. C(i2,h1,i1,h2; a) =
// A(h1,i1,h2,k,i2; a,c) * B(h2,h1,i2,i1,k; c). Two Hadamard (h1,h2), two left
// externals (i1,i2), outer contracted k, inner contraction c (right-clean),
// inner spectator a. Routes through GeneralPermutationOptimizer; uses the ce+ce
// strided kernel per batch. R4 caveat applies.
BOOST_AUTO_TEST_CASE(hce_ce_scramble_representative_fires) {
  auto& w = TA::get_default_world();
  auto h = tr1(3, 3), t = tr1(4, 2), kk = tr1(4, 4);
  // True hce+ce: h1,h2 fused (Hadamard); i1,i2 LEFT externals (A,C only);
  // i3,i4 RIGHT externals (B,C only); k outer-contracted; c inner-contraction,
  // a inner spectator. A outer (h1,i1,h2,k,i2), B outer (h2,h1,i4,i3,k),
  // result C(i2,h1,i3,i1,h2,i4;a).
  ArrayToT A0 = make_arena_tot2(w, TA::TiledRange{h, t, h, kk, t}, 3, 4); // a,c
  ArrayToT B0 = make_arena_tot (w, TA::TiledRange{h, h, t, t, kk}, 4);    // c
  OwnArr A0o = make_own_tot2(w, TA::TiledRange{h, t, h, kk, t}, 3, 4);
  OwnArr B0o = make_own_tot (w, TA::TiledRange{h, h, t, t, kk}, 4);
  w.gop.fence();
  OwnArr ref = TA::einsum(A0o("h1,i1,h2,k,i2;a,c"),
                          B0o("h2,h1,i4,i3,k;c"), "i2,h1,i3,i1,h2,i4;a");
  w.gop.fence();
  // Scramble outer orders (fused+ext+contracted) on both operands; cap to keep
  // the matrix bounded. All must fire and match the owning ground truth.
  run_scrambles(A0, "h1,i1,h2,k,i2;a,c", B0, "h2,h1,i4,i3,k;c",
                "i2,h1,i3,i1,h2,i4;a", ref,
                orderings("h1,i1,h2,k,i2", /*cap=*/6),
                orderings("h2,h1,i4,i3,k", /*cap=*/6),
#ifdef TA_STRIDED_DGEMM_COUNT
                &TA::detail::g_strided_dgemm_ce_ce_right_calls,
                &TA::detail::g_strided_dgemm_ce_ce_left_calls
#else
                nullptr, nullptr
#endif
  );
}

// ce+ce LEFT arm: left-clean mirror of ce_ce_scramble_all_outer_orders_fire.
// The LEFT operand is the CLEAN contraction vector (inner c only); the RIGHT
// operand carries the inner external (a spectator + c contracted). This routes
// to arena_strided_dgemm_ce_ce_left -- the arm the right-clean ce+ce test never
// reaches. Outer: k outer-contracted, j1 left external, i1,i2 right externals.
// Inner: c contracted, a spectator. Every outer scramble must match the owning
// ground truth AND fire a strided arm; a focused canonical sub-check then proves
// the LEFT arm specifically fired (and not the right).
BOOST_AUTO_TEST_CASE(ce_ce_left_scramble_all_outer_orders_fire) {
  auto& w = TA::get_default_world();
  auto t = tr1(4, 2), kk = tr1(4, 4);
  // LEFT operand is the CLEAN contraction vector (inner c only) -> exercises the
  // ce_ce_LEFT arm, which ce_ce_scramble (right-clean) never reaches. This is the
  // left-clean mirror of the right-clean ce+ce; the user's literal point-2 shape
  // A(...;a)*B(...;c,a) is a one-sided inner reduction over c (not a two-operand
  // inner contraction) so it does not map to ce+ce -- the mirror is the correct
  // way to cover the left arm.
  ArrayToT A0 = make_arena_tot (w, TA::TiledRange{t, kk},    /*c=*/4);          // clean vector
  ArrayToT B0 = make_arena_tot2(w, TA::TiledRange{t, t, kk}, /*a=*/3, /*c=*/4); // spectator matrix
  OwnArr  A0o = make_own_tot  (w, TA::TiledRange{t, kk},    4);
  OwnArr  B0o = make_own_tot2 (w, TA::TiledRange{t, t, kk}, 3, 4);
  w.gop.fence();
  OwnArr ref = TA::einsum(A0o("j1,k;c"), B0o("i1,i2,k;a,c"), "i1,i2,j1;a");
  w.gop.fence();
  run_scrambles(A0, "j1,k;c", B0, "i1,i2,k;a,c", "i1,i2,j1;a", ref,
                orderings(outer_of("j1,k;c")), orderings(outer_of("i1,i2,k;a,c")),
#ifdef TA_STRIDED_DGEMM_COUNT
                &TA::detail::g_strided_dgemm_ce_ce_right_calls,
                &TA::detail::g_strided_dgemm_ce_ce_left_calls
#else
                nullptr, nullptr
#endif
  );

#ifdef TA_STRIDED_DGEMM_COUNT
  // Focused proof: ONE canonical einsum hits the LEFT arm specifically.
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  ArrayToT Cone = TA::einsum(A0("j1,k;c"), B0("i1,i2,k;a,c"), "i1,i2,j1;a");
  w.gop.fence();
  std::size_t lf = TA::detail::g_strided_dgemm_ce_ce_left_calls.load();
  std::size_t rf = TA::detail::g_strided_dgemm_ce_ce_right_calls.load();
  w.gop.sum(&lf, 1); w.gop.sum(&rf, 1);
  BOOST_CHECK_GT(lf, std::size_t{0});    // LEFT arm actually fired
  BOOST_CHECK_EQUAL(rf, std::size_t{0}); // and not the right arm
#endif
}

// ce+ce FREE inner transpose: the relaxed inner-perm gate folds a
// matrix_transpose of the EXTERNAL-carrying operand's inner into the GEMM op
// flag (zero-copy), so the strided kernel STILL fires. Right-clean ce+ce: the
// LEFT (external-carrying) operand's inner is written contracted-first (c,a)
// instead of canonical external-first (a,c) -- a contiguous two-block swap =>
// matrix_transpose -- while the CLEAN right vector stays identity and the result
// inner (a) stays in canonical (identity-content) order. The fold keeps it
// firing without a physical inner permute.
BOOST_AUTO_TEST_CASE(ce_ce_free_inner_transpose_fires) {
  auto& w = TA::get_default_world();
  auto t = tr1(4, 2), kk = tr1(4, 4);
  // inner cell holds (c=4, a=3): the LEFT operand's inner is annotated (c,a)
  // (contracted-first), which is a matrix_transpose of canonical (a,c). The
  // clean right operand carries inner (c) only; result inner is (a) -- identity.
  ArrayToT A0 = make_arena(w, TA::TiledRange{t, t, kk}, TA::Range{4, 3}); // inner c=4,a=3
  ArrayToT B0 = make_arena_tot(w, TA::TiledRange{t, kk}, /*c=*/4);
  OwnArr  A0o = make_own (w, TA::TiledRange{t, t, kk}, TA::Range{4, 3});
  OwnArr  B0o = make_own_tot(w, TA::TiledRange{t, kk}, 4);
  w.gop.fence();
  // ground truth via owning (non-strided) path -- left inner (c,a), reduce c,
  // keep a; result inner (a)
  OwnArr ref = TA::einsum(A0o("i1,i2,k;c,a"), B0o("j1,k;c"), "i1,i2,j1;a");
  w.gop.fence();
#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
#endif
  ArrayToT C = TA::einsum(A0("i1,i2,k;c,a"), B0("j1,k;c"), "i1,i2,j1;a");
  w.gop.fence();
  BOOST_CHECK_SMALL(array_max_reldiff(C, ref), 1e-9);   // transpose fold stays correct
#ifdef TA_STRIDED_DGEMM_COUNT
  std::size_t f = TA::detail::g_strided_dgemm_ce_ce_right_calls.load() +
                  TA::detail::g_strided_dgemm_ce_ce_left_calls.load();
  w.gop.sum(&f, 1);
  BOOST_CHECK_GT(f, std::size_t{0});                    // free transpose => STILL fires

  // Arm-specific proof: this is a RIGHT-clean ce+ce, so the transpose fold must
  // fire on the RIGHT arm specifically (not "some arm"). ONE canonical einsum.
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
  ArrayToT Cone = TA::einsum(A0("i1,i2,k;c,a"), B0("j1,k;c"), "i1,i2,j1;a");
  w.gop.fence();
  std::size_t rf = TA::detail::g_strided_dgemm_ce_ce_right_calls.load();
  std::size_t lf = TA::detail::g_strided_dgemm_ce_ce_left_calls.load();
  w.gop.sum(&rf, 1); w.gop.sum(&lf, 1);
  BOOST_CHECK_GT(rf, std::size_t{0});    // RIGHT arm actually fired
  BOOST_CHECK_EQUAL(lf, std::size_t{0}); // and not the left arm
#endif
}

// ce+ce NEGATIVE: a GENUINE non-identity inner RESULT permutation must NOT fire
// any strided kernel (guards the is_identity gate relaxation, commit 98d0650a,
// from being too loose), yet must still compute correctly via the by-cell path.
// Right-clean ce+ce with a 2-mode spectator (a1,a2): the result inner modes are
// requested SWAPPED (a2,a1) -- a real reorder, not identity-content. This makes
// inner(perm_) non-empty and not is_identity -> no_result_inner_perm false ->
// both arms refuse -> no fire.
BOOST_AUTO_TEST_CASE(ce_ce_general_inner_result_perm_does_not_fire) {
  auto& w = TA::get_default_world();
  auto t = tr1(4, 2), kk = tr1(4, 4);
  // inner cell holds (a1=3, a2=2, c=4): reduce c (last), keep a1,a2; result
  // swaps them.
  ArrayToT A0 = make_arena(w, TA::TiledRange{t, t, kk}, TA::Range{3, 2, 4});
  ArrayToT B0 = make_arena_tot(w, TA::TiledRange{t, kk}, /*c=*/4);
  OwnArr  A0o = make_own (w, TA::TiledRange{t, t, kk}, TA::Range{3, 2, 4});
  OwnArr  B0o = make_own_tot(w, TA::TiledRange{t, kk}, 4);
  w.gop.fence();
  // ground truth with SWAPPED result inner (a2,a1)
  OwnArr ref = TA::einsum(A0o("i1,i2,k;a1,a2,c"), B0o("j1,k;c"), "i1,i2,j1;a2,a1");
  w.gop.fence();
#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
#endif
  ArrayToT C = TA::einsum(A0("i1,i2,k;a1,a2,c"), B0("j1,k;c"), "i1,i2,j1;a2,a1");
  w.gop.fence();
  BOOST_CHECK_SMALL(array_max_reldiff(C, ref), 1e-9);   // by-cell path still correct
#ifdef TA_STRIDED_DGEMM_COUNT
  std::size_t f = TA::detail::g_strided_dgemm_ce_ce_right_calls.load() +
                  TA::detail::g_strided_dgemm_ce_ce_left_calls.load();
  w.gop.sum(&f, 1);
  BOOST_CHECK_EQUAL(f, std::size_t{0});                 // genuine inner perm => NO strided fire

  // Would-fire guard: the SAME operands with an IDENTITY result inner
  // ("a1,a2" instead of swapped "a2,a1") DO fire. This proves the negative
  // above is caused by the inner result permutation, not by the shape simply
  // lacking a strided candidate.
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
  ArrayToT Cid = TA::einsum(A0("i1,i2,k;a1,a2,c"), B0("j1,k;c"), "i1,i2,j1;a1,a2");
  w.gop.fence();
  std::size_t fi = TA::detail::g_strided_dgemm_ce_ce_right_calls.load() +
                   TA::detail::g_strided_dgemm_ce_ce_left_calls.load();
  w.gop.sum(&fi, 1);
  BOOST_CHECK_GT(fi, std::size_t{0});                   // identity inner => DOES fire
#endif
}

BOOST_AUTO_TEST_SUITE_END()
