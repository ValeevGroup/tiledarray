/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
 *
 *  Unit tests for the pure per-axis nesting plan and the
 *  T-grid -> U-result ordinal map (contraction_retile.h). No World needed:
 *  TiledRange1 / TiledRange / GemmHelper are all built directly.
 */

#include "tiledarray.h"

#include <TiledArray/conversions/foreach.h>
#include <TiledArray/einsum/tiledarray.h>
#include <TiledArray/expressions/contraction_retile.h>
#include <TiledArray/math/blas.h>
#include <TiledArray/math/gemm_helper.h>
#include <TiledArray/range.h>
#include <TiledArray/tensor/arena_einsum.h>
#include <TiledArray/tensor/arena_kernels.h>
#include <TiledArray/tensor/arena_retile.h>
#include <TiledArray/tensor/arena_tensor.h>
#include <TiledArray/tensor/tensor.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/tiled_range1.h>

#include "unit_test_config.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>

namespace TA = TiledArray;
using TA::expressions::AxisNest;
using TA::expressions::make_retile_plan;
using TA::expressions::NestDir;
using TA::expressions::RetilePlan;

namespace {
// Uniform 1-D tiling of [0,n) with tile size ts (last tile may be short).
TA::TiledRange1 tr1(std::size_t n, std::size_t ts) {
  if (ts == 0 || ts > n) ts = n;
  std::vector<std::size_t> b;
  for (std::size_t x = 0; x < n; x += ts) b.push_back(x);
  b.push_back(n);
  return TA::TiledRange1(b.begin(), b.end());
}
}  // namespace

BOOST_AUTO_TEST_SUITE(summa_two_trange_suite, TA_UT_LABEL_SERIAL)

// identity: empty targets => active==false, every dir==Identity,
// u_result_ordinals(n)=={n}. A simple matrix-product GemmHelper
// C(m,n) = A(m,k) * B(k,n): left outer {m}, right outer {n}, contracted {k}.
BOOST_AUTO_TEST_CASE(plan_identity) {
  auto m = tr1(8, 2), k = tr1(6, 3), n = tr1(4, 2);
  TA::TiledRange left_U{m, k};   // A(m,k)
  TA::TiledRange right_U{k, n};  // B(k,n)
  TA::math::GemmHelper gh(TA::math::blas::NoTranspose,
                          TA::math::blas::NoTranspose,
                          /*result_rank=*/2, /*left_rank=*/2,
                          /*right_rank=*/2);
  // inner helper (per-tile contract-reduce); only its contract-rank count is
  // read, for flag classification, which this nesting test does not inspect.
  TA::math::GemmHelper inner_gh(TA::math::blas::NoTranspose,
                                TA::math::blas::NoTranspose, 2, 2, 2);
  RetilePlan p = make_retile_plan(left_U, right_U, /*targetH=*/{}, /*targetM=*/{},
                                  /*targetN=*/{}, /*targetK=*/{}, gh, inner_gh,
                                  /*n_fused=*/0);
  BOOST_CHECK_EQUAL(p.active, false);
  for (const auto& ax : p.hadamard) BOOST_CHECK(ax.dir == NestDir::Identity);
  for (const auto& ax : p.summaM) BOOST_CHECK(ax.dir == NestDir::Identity);
  for (const auto& ax : p.summaN) BOOST_CHECK(ax.dir == NestDir::Identity);
  for (const auto& ax : p.summaK) BOOST_CHECK(ax.dir == NestDir::Identity);
  for (std::size_t i = 0; i < 5; ++i) {
    auto u = p.u_result_ordinals(i);
    BOOST_REQUIRE_EQUAL(u.size(), std::size_t{1});
    BOOST_CHECK_EQUAL(u[0], i);
  }
}

// coarsen M: U M-axis is tile-size 4 over 49 elements (13 tiles), target M is a
// single tile of 49. T is coarser than U => dir==Coarsen, one group [0,13),
// u_is_finer==true. The single coarse M-cell maps to all 13 U M-ordinals.
BOOST_AUTO_TEST_CASE(plan_coarsen_m) {
  // N is a single tile (result_u_extent_N == 1) so the result-grid ordinal map
  // exposes the M-axis ordinals directly: a coarse M-cell -> all its U M tiles.
  auto mU = tr1(49, 4), k = tr1(6, 3), n = tr1(4, 4);
  TA::TiledRange left_U{mU, k};
  TA::TiledRange right_U{k, n};
  TA::math::GemmHelper gh(TA::math::blas::NoTranspose,
                          TA::math::blas::NoTranspose, 2, 2, 2);
  TA::math::GemmHelper inner_gh(TA::math::blas::NoTranspose,
                                TA::math::blas::NoTranspose, 2, 2, 2);
  auto mT = tr1(49, 49);  // single tile
  RetilePlan p =
      make_retile_plan(left_U, right_U, {}, {mT}, {}, {}, gh, inner_gh, 0);
  BOOST_REQUIRE_EQUAL(p.summaM.size(), std::size_t{1});
  const AxisNest& ax = p.summaM[0];
  BOOST_CHECK(ax.dir == NestDir::Coarsen);
  BOOST_CHECK_EQUAL(ax.u_is_finer, true);
  BOOST_REQUIRE_EQUAL(ax.groups.size(), std::size_t{1});
  BOOST_CHECK_EQUAL(ax.groups[0].first, std::size_t{0});
  BOOST_CHECK_EQUAL(ax.groups[0].second, std::size_t{13});
  BOOST_CHECK_EQUAL(p.active, true);
  // the single coarse M grid-cell (ordinal 0) covers all 13 U M-ordinals
  auto u = p.u_result_ordinals(0);
  BOOST_REQUIRE_EQUAL(u.size(), std::size_t{13});
  for (std::size_t i = 0; i < 13; ++i) BOOST_CHECK_EQUAL(u[i], i);
}

// refine N: U N-axis is a single tile of 52 elements, target N is tile-size 4
// (13 tiles). T is finer than U => dir==Refine, 13 groups, u_is_finer==false.
BOOST_AUTO_TEST_CASE(plan_refine_n) {
  auto m = tr1(8, 2), k = tr1(6, 3), nU = tr1(52, 52);
  TA::TiledRange left_U{m, k};
  TA::TiledRange right_U{k, nU};
  TA::math::GemmHelper gh(TA::math::blas::NoTranspose,
                          TA::math::blas::NoTranspose, 2, 2, 2);
  TA::math::GemmHelper inner_gh(TA::math::blas::NoTranspose,
                                TA::math::blas::NoTranspose, 2, 2, 2);
  auto nT = tr1(52, 4);  // 13 tiles
  RetilePlan p =
      make_retile_plan(left_U, right_U, {}, {}, {nT}, {}, gh, inner_gh, 0);
  BOOST_REQUIRE_EQUAL(p.summaN.size(), std::size_t{1});
  const AxisNest& ax = p.summaN[0];
  BOOST_CHECK(ax.dir == NestDir::Refine);
  BOOST_CHECK_EQUAL(ax.u_is_finer, false);
  BOOST_REQUIRE_EQUAL(ax.groups.size(), std::size_t{13});
  BOOST_CHECK_EQUAL(p.active, true);
}

// non-nesting: U N-axis boundaries {0,4,8}; a target boundary at 6 straddles a
// U tile boundary => make_retile_plan must throw TA::Exception.
BOOST_AUTO_TEST_CASE(plan_non_nesting) {
  auto m = tr1(8, 2), k = tr1(6, 3);
  std::vector<std::size_t> ub{0, 4, 8};
  TA::TiledRange1 nU(ub.begin(), ub.end());
  TA::TiledRange left_U{m, k};
  TA::TiledRange right_U{k, nU};
  TA::math::GemmHelper gh(TA::math::blas::NoTranspose,
                          TA::math::blas::NoTranspose, 2, 2, 2);
  TA::math::GemmHelper inner_gh(TA::math::blas::NoTranspose,
                                TA::math::blas::NoTranspose, 2, 2, 2);
  std::vector<std::size_t> tb{0, 6, 8};  // 6 straddles U boundary {4}
  TA::TiledRange1 nT(tb.begin(), tb.end());
  BOOST_REQUIRE_THROW(
      make_retile_plan(left_U, right_U, {}, {}, {nT}, {}, gh, inner_gh, 0),
      TA::Exception);
}

// flags: ride_on_M / k_is_blas_k derived from the INNER GemmHelper exactly as
// the install gates classify off `contrreduce_op.gemm_helper()`
// The OUTER helper gh
// is used only for role partitioning and must NOT drive the flags.
//   ce_e:  inner num_contract_ranks()==0 => k_is_blas_k.
//   ce_ce: inner num_contract_ranks()>=1 => ride on a SUMMA external
//          (BLAS-M-side) => ride_on_M.
BOOST_AUTO_TEST_CASE(plan_flags) {
  auto m = tr1(8, 2), k = tr1(6, 3), n = tr1(4, 2);
  TA::TiledRange left_U{m, k};
  TA::TiledRange right_U{k, n};

  // OUTER (SUMMA) helper: a plain matrix product C(m,n)=A(m,k)*B(k,n). This
  // fixes the role partition and is the SAME for both sub-cases; the flags below
  // come from inner_gh, NOT from this outer helper (whose contract-rank count
  // is 1, i.e. matches neither sub-case's intent on its own).
  TA::math::GemmHelper gh(TA::math::blas::NoTranspose,
                          TA::math::blas::NoTranspose,
                          /*result_rank=*/2, /*left_rank=*/2,
                          /*right_rank=*/2);
  BOOST_REQUIRE_EQUAL(gh.num_contract_ranks(), 1u);

  // ce_e INNER helper: left inner (a), right inner (b), result inner (a,b):
  // (1 + 1 - 2)/2 = 0 contracted ranks (inner outer-product).
  TA::math::GemmHelper inner_ce_e(TA::math::blas::NoTranspose,
                                  TA::math::blas::NoTranspose,
                                  /*result_rank=*/2, /*left_rank=*/1,
                                  /*right_rank=*/1);
  BOOST_REQUIRE_EQUAL(inner_ce_e.num_contract_ranks(), 0u);
  RetilePlan pe =
      make_retile_plan(left_U, right_U, {}, {}, {}, {}, gh, inner_ce_e, 0);
  BOOST_CHECK_EQUAL(pe.k_is_blas_k, true);
  BOOST_CHECK_EQUAL(pe.ride_on_M, false);
  BOOST_CHECK_EQUAL(pe.ride_on_N, false);

  // ce_ce INNER helper: left inner (a,c), right inner (c), result inner (a):
  // (2 + 1 - 1)/2 = 1 contracted rank (inner contraction).
  TA::math::GemmHelper inner_ce_ce(TA::math::blas::NoTranspose,
                                   TA::math::blas::NoTranspose,
                                   /*result_rank=*/1, /*left_rank=*/2,
                                   /*right_rank=*/1);
  BOOST_REQUIRE_EQUAL(inner_ce_ce.num_contract_ranks(), 1u);
  RetilePlan pc =
      make_retile_plan(left_U, right_U, {}, {}, {}, {}, gh, inner_ce_ce, 0);
  BOOST_CHECK_EQUAL(pc.ride_on_M, true);
  BOOST_CHECK_EQUAL(pc.k_is_blas_k, false);
  BOOST_CHECK_EQUAL(pc.ride_on_N, false);
}

// ---------------------------------------------------------------------------
// Arena gather/carve helpers: arena_gather_block +
// arena_carve_block on TA::Tensor<ArenaTensor<double>> tiles.
// ---------------------------------------------------------------------------
namespace {
using ArenaInner = TA::ArenaTensor<double>;
using ArenaOuter = TA::Tensor<ArenaInner>;
using InnerRange = ArenaInner::range_type;  // btas::zb::RangeNd (rank-1 here)

// Build a fine arena ToT outer tile over [lo, hi) (1-D outer), one batch, with
// inner extent given per outer position by `ext(outer_idx)` and elements filled
// `base + 1000*outer_idx + inner` so every gathered value is distinguishable.
template <typename ExtFn>
ArenaOuter make_fine(std::size_t lo, std::size_t hi, double base, ExtFn ext) {
  TA::Range outer({lo}, {hi});
  auto range_fn = [&](std::size_t ord) -> InnerRange {
    auto idx = outer.idx(ord);
    return InnerRange{long(ext(std::size_t(idx[0])))};
  };
  ArenaOuter t = TA::detail::arena_outer_init<ArenaOuter>(outer, 1, range_fn);
  for (std::size_t ord = 0; ord < outer.volume(); ++ord) {
    auto idx = outer.idx(ord);
    auto& cell = t.data()[ord];
    if (cell.empty()) continue;
    for (std::size_t i = 0; i < cell.size(); ++i)
      cell.data()[i] = base + 1000.0 * double(idx[0]) + double(i);
  }
  return t;
}

// True iff every non-null cell of `t` lies in one contiguous, non-overlapping,
// monotonically-increasing memory span -- the single-arena-page invariant
// (proves the gather did not spill across pages / allocate per cell).
bool single_page(const ArenaOuter& t) {
  const double* prev_end = nullptr;
  const std::size_t N = t.range().volume() * t.nbatch();
  for (std::size_t ord = 0; ord < N; ++ord) {
    const auto& c = t.data()[ord];
    if (c.empty()) continue;
    const double* p = c.data();
    if (prev_end != nullptr && p < prev_end) return false;  // overlap/back-jump
    prev_end = p + c.size();
  }
  return true;
}
}  // namespace

// (a) uniform: equal inner extents over a coarse outer block -> gather is
// single-page, constant stride (classifier clean), values == ordered
// concatenation of fine cells. Carve back (view) reproduces each fine tile;
// owning carve (view=false) is equal and independent of the coarse storage.
BOOST_AUTO_TEST_CASE(arena_gather_uniform) {
  const std::size_t E = 3;  // uniform inner extent
  // Three fine tiles partitioning coarse outer [0,6): [0,2),[2,4),[4,6).
  std::vector<ArenaOuter> fine;
  fine.push_back(make_fine(0, 2, 1.0, [&](std::size_t) { return E; }));
  fine.push_back(make_fine(2, 4, 2.0, [&](std::size_t) { return E; }));
  fine.push_back(make_fine(4, 6, 3.0, [&](std::size_t) { return E; }));
  TA::Range coarse_outer({0}, {6});

  ArenaOuter g =
      TA::detail::arena_gather_block<ArenaOuter>(fine, coarse_outer, 1);

  // single-page + values == ordered concatenation of the fine cells.
  BOOST_CHECK(single_page(g));
  BOOST_REQUIRE_EQUAL(g.range().volume(), std::size_t{6});
  for (std::size_t p = 0; p < 6; ++p) {
    const std::size_t f = p / 2, lp = p % 2;
    const auto& src = fine[f].data()[lp];
    const auto& dst = g.data()[p];
    BOOST_REQUIRE_EQUAL(dst.size(), src.size());
    for (std::size_t i = 0; i < dst.size(); ++i)
      BOOST_CHECK_EQUAL(dst.data()[i], src.data()[i]);
  }

  // constant stride / strided-eligible: the stride-run classifier is clean (0).
  int cls = TA::detail::classify_run(
      [&](std::size_t i) -> const ArenaInner& { return g.data()[i]; }, 6);
  BOOST_CHECK_EQUAL(cls, 0);

  // carve back to the original fine ranges (view = true): each carved sub-tile
  // aliases the coarse storage and reproduces the original fine values.
  std::vector<TA::Range> fine_ranges;
  for (const auto& f : fine) fine_ranges.push_back(f.range());
  auto carved = TA::detail::arena_carve_block<ArenaOuter>(g, fine_ranges,
                                                          /*view=*/true);
  BOOST_REQUIRE_EQUAL(carved.size(), fine.size());
  for (std::size_t f = 0; f < fine.size(); ++f) {
    BOOST_REQUIRE_EQUAL(carved[f].range().volume(), fine[f].range().volume());
    for (std::size_t p = 0; p < fine[f].range().volume(); ++p) {
      const auto& a = carved[f].data()[p];
      const auto& b = fine[f].data()[p];
      BOOST_REQUIRE_EQUAL(a.size(), b.size());
      for (std::size_t i = 0; i < a.size(); ++i)
        BOOST_CHECK_EQUAL(a.data()[i], b.data()[i]);
    }
    // view = true aliases the coarse slab (zero-copy).
    BOOST_CHECK(carved[f].data()[0].data() ==
                g.data()[f * 2].data());
  }

  // owning carve (view = false): equal values but independent storage -- a
  // mutation of the carved copy must not touch the coarse tile.
  auto owned = TA::detail::arena_carve_block<ArenaOuter>(g, fine_ranges,
                                                         /*view=*/false);
  BOOST_REQUIRE_EQUAL(owned.size(), fine.size());
  BOOST_CHECK(single_page(owned[0]));
  // independence: distinct storage from the coarse tile.
  BOOST_CHECK(owned[0].data()[0].data() != g.data()[0].data());
  const double before = g.data()[0].data()[0];
  owned[0].data()[0].data()[0] += 12345.0;
  BOOST_CHECK_EQUAL(g.data()[0].data()[0], before);
  for (std::size_t p = 0; p < fine[0].range().volume(); ++p) {
    const auto& a = owned[0].data()[p];
    const auto& b = fine[0].data()[p];
    for (std::size_t i = 0; i < a.size(); ++i) {
      if (p == 0 && i == 0) continue;  // mutated above
      BOOST_CHECK_EQUAL(a.data()[i], b.data()[i]);
    }
  }
}

// (b) non-uniform: differing inner extents along the strided axis -> gather is
// still single-page with correct values, but the stride-run classifier reports
// nonuniform (2) so the kernel knows to fall back per-cell rather than crash.
BOOST_AUTO_TEST_CASE(arena_gather_nonuniform) {
  // Fine tiles over coarse outer [0,4): [0,2),[2,4). Inner extent varies by
  // outer position: 2,3,4,5 -> differing sizes along the gathered run.
  auto ext = [](std::size_t i) { return i + 2; };
  std::vector<ArenaOuter> fine;
  fine.push_back(make_fine(0, 2, 1.0, ext));
  fine.push_back(make_fine(2, 4, 2.0, ext));
  TA::Range coarse_outer({0}, {4});

  ArenaOuter g =
      TA::detail::arena_gather_block<ArenaOuter>(fine, coarse_outer, 1);

  // single-page + correct values despite the non-uniform extents.
  BOOST_CHECK(single_page(g));
  BOOST_REQUIRE_EQUAL(g.range().volume(), std::size_t{4});
  for (std::size_t p = 0; p < 4; ++p) {
    const std::size_t f = p / 2, lp = p % 2;
    const auto& src = fine[f].data()[lp];
    const auto& dst = g.data()[p];
    BOOST_REQUIRE_EQUAL(dst.size(), src.size());
    BOOST_CHECK_EQUAL(dst.size(), ext(p));
    for (std::size_t i = 0; i < dst.size(); ++i)
      BOOST_CHECK_EQUAL(dst.data()[i], src.data()[i]);
  }

  // The stride-run classifier flags this run as nonuniform (2): the kernel will
  // fall back per-cell, not attempt strided DGEMM.
  int cls = TA::detail::classify_run(
      [&](std::size_t i) -> const ArenaInner& { return g.data()[i]; }, 4);
  BOOST_CHECK_EQUAL(cls, 2);
}

// ---------------------------------------------------------------------------
// .retile(...) plumbing through the engine. Identity (empty / own-U
// targets) must be bit-for-bit the no-retile result (the identity anchor);
// a Hadamard product with .retile must reject (MultEngine guard).
//
// Harness mirrors tests/strided_canonicalize.cpp: ArenaTensor<double> inner
// cells, single-page arena ToT built via arena_outer_init, an owning-ToT twin
// as independent ground truth, and a relative max-diff oracle.
// ---------------------------------------------------------------------------
namespace {
using ArenaInner1 = TA::ArenaTensor<double, TA::Range>;
using ArenaOuter1 = TA::Tensor<ArenaInner1>;
using ArrayToT1 = TA::DistArray<ArenaOuter1, TA::DensePolicy>;
using OwnInner1 = TA::Tensor<double>;
using OwnOuter1 = TA::Tensor<OwnInner1>;
using OwnArr1 = TA::DistArray<OwnOuter1, TA::DensePolicy>;

template <typename Index>
double outer_seed1(const Index& oix) {
  double seed = 0.0, f = 1.0;
  for (auto c : oix) {
    seed += double(c) * f;
    f *= 31.0;
  }
  return seed;
}
inline double cell_val1(double outer_s, std::size_t e) {
  return 1.0 + 1e-3 * double(e) + 1e-2 * outer_s;
}

ArrayToT1 make_arena1(TA::World& w, const TA::TiledRange& tr,
                      const TA::Range& inner) {
  ArrayToT1 x(w, tr);
  x.init_tiles([inner](const TA::Range& t_outer) {
    ArenaOuter1 t = TA::detail::arena_outer_init<ArenaOuter1>(
        t_outer, 1, [inner](std::size_t) { return inner; });
    std::size_t o = 0;
    for (const auto& idx : t_outer) {
      ArenaInner1& c = t.data()[o++];
      if (!c) continue;
      const double s = outer_seed1(idx);
      for (std::size_t e = 0; e < c.size(); ++e) c.data()[e] = cell_val1(s, e);
    }
    return t;
  });
  return x;
}
OwnArr1 make_own1(TA::World& w, const TA::TiledRange& tr,
                  const TA::Range& inner) {
  OwnArr1 x(w, tr);
  x.init_tiles([inner](const TA::Range& t_outer) {
    OwnOuter1 t(t_outer);
    std::size_t o = 0;
    for (const auto& idx : t_outer) {
      OwnInner1 cell(inner);
      const double s = outer_seed1(idx);
      for (std::size_t e = 0; e < cell.size(); ++e)
        cell.data()[e] = cell_val1(s, e);
      t.data()[o++] = cell;
    }
    return t;
  });
  return x;
}

template <typename ArrA, typename ArrB>
double array_max_reldiff1(const ArrA& A, const ArrB& B) {
  double d = 0.0;
  for (auto it = A.begin(); it != A.end(); ++it) {
    const auto a_tile = (*it).get();
    if (a_tile.empty()) continue;
    const auto b_tile = B.find(it.index()).get();
    if (a_tile.size() != b_tile.size()) {
      d = std::max(d, 1.0);
      continue;
    }
    for (std::size_t o = 0; o < a_tile.size(); ++o) {
      const auto& ac = a_tile[o];
      const auto& bc = b_tile[o];
      if (ac.size() != bc.size()) {
        d = std::max(d, 1.0);
        continue;
      }
      for (std::size_t e = 0; e < ac.size(); ++e) {
        const double av = double(ac.data()[e]), bv = double(bc.data()[e]);
        d = std::max(d, std::abs(av - bv) / std::max(1.0, std::abs(bv)));
      }
    }
  }
  A.world().gop.max(&d, 1);
  return d;
}
}  // namespace

// (a) identity: a real strided-eligible ce+e ToT contraction
// C(i1,i2,j1,j2;a,b) = A(i1,i2,k;a) * B(j1,j2,k;b) computed normally, then with
// .retile() using EMPTY targets and with targets == the operands' own U
// tilings. Both retile variants must equal the no-retile result to 1e-12.
BOOST_AUTO_TEST_CASE(retile_identity_ce_e) {
  auto& w = TA::get_default_world();
  auto t = tr1(4, 2), kk = tr1(4, 4);
  ArrayToT1 A0 = make_arena1(w, TA::TiledRange{t, t, kk}, TA::Range{3});
  ArrayToT1 B0 = make_arena1(w, TA::TiledRange{t, t, kk}, TA::Range{5});
  OwnArr1 A0o = make_own1(w, TA::TiledRange{t, t, kk}, TA::Range{3});
  OwnArr1 B0o = make_own1(w, TA::TiledRange{t, t, kk}, TA::Range{5});
  w.gop.fence();

  // no-retile reference (also cross-checked against the owning ground truth)
  OwnArr1 ref = TA::einsum(A0o("i1,i2,k;a"), B0o("j1,j2,k;b"), "i1,i2,j1,j2;a,b");
  ArrayToT1 C_ref =
      TA::einsum(A0("i1,i2,k;a"), B0("j1,j2,k;b"), "i1,i2,j1,j2;a,b");
  w.gop.fence();
  BOOST_CHECK_SMALL(array_max_reldiff1(C_ref, ref), 1e-9);

  // .retile() with EMPTY targets -> plan inactive -> identical result.
  ArrayToT1 C_empty;
  C_empty("i1,i2,j1,j2;a,b") =
      (A0("i1,i2,k;a") * B0("j1,j2,k;b")).retile({}, {}, {}, {});
  w.gop.fence();
  BOOST_CHECK_SMALL(array_max_reldiff1(C_empty, C_ref), 1e-12);

  // .retile() with targets == the operands' own U tilings -> every axis
  // Identity -> plan inactive -> identical result. Roles for this ce+e:
  // M = {i1,i2} (left externals), N = {j1,j2} (right externals), K = {k}.
  ArrayToT1 C_self;
  C_self("i1,i2,j1,j2;a,b") =
      (A0("i1,i2,k;a") * B0("j1,j2,k;b"))
          .retile(/*H=*/{}, /*M=*/{t, t}, /*N=*/{t, t}, /*K=*/{kk});
  w.gop.fence();
  BOOST_CHECK_SMALL(array_max_reldiff1(C_self, C_ref), 1e-12);
}

// (b) Hadamard reject: an elementwise product C(i,j) = A(i,j) * B(i,j) resolves
// to MultEngine (Hadamard outer product), where the retile target has no
// well-defined contraction role partition. Requesting .retile(...) must throw
// TA::Exception (the MultEngine guard).
BOOST_AUTO_TEST_CASE(retile_hadamard_reject) {
  auto& w = TA::get_default_world();
  auto i = tr1(4, 2), j = tr1(4, 2);
  using ArrayD = TA::DistArray<TA::Tensor<double>, TA::DensePolicy>;
  ArrayD A(w, TA::TiledRange{i, j});
  ArrayD B(w, TA::TiledRange{i, j});
  A.fill(1.0);
  B.fill(2.0);
  w.gop.fence();
  ArrayD C;
  BOOST_REQUIRE_THROW(
      (C("i,j") = (A("i,j") * B("i,j")).retile({}, {i}, {j}, {})),
      TA::Exception);
}

// ---------------------------------------------------------------------------
// inbound K-coarsen for a ce+e ToT contraction.
//
// C(i1,i2,j1,j2;a,b) = A(i1,i2,k;a) * B(j1,j2,k;b) with the OUTER contracted
// index k finely tiled in U (4 K-tiles); a .retile() collapses K to ONE tile.
// The externals (M={i1,i2}, N={j1,j2}) stay Identity. The retiled result must
// match the no-retile result (< 1e-9), the plan must be active, the strided
// ce_e DGEMM must fire (the packed coarse K-block rides ONE fat GEMM per
// (m,n)), and the per-coarse-cell gather hook must report the fine K width.
//
// np=1 ONLY (this suite is @serial): the gather is local on a single rank, so
// the operand-locality question at np>1 is deferred to a later phase.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(retile_coarsen_k_ce_e) {
  auto& w = TA::get_default_world();
  // U: K finely tiled into 4 tiles of width 2 over [0,8); externals 2 tiles.
  auto t = tr1(4, 2);
  auto kU = tr1(8, 2);            // 4 K-tiles
  auto kT = tr1(8, 8);           // collapse to 1 K-tile
  const std::size_t k_fine = 4;  // # of U K-tiles per coarse K-cell

  ArrayToT1 A0 = make_arena1(w, TA::TiledRange{t, t, kU}, TA::Range{3});
  ArrayToT1 B0 = make_arena1(w, TA::TiledRange{t, t, kU}, TA::Range{5});
  OwnArr1 A0o = make_own1(w, TA::TiledRange{t, t, kU}, TA::Range{3});
  OwnArr1 B0o = make_own1(w, TA::TiledRange{t, t, kU}, TA::Range{5});
  w.gop.fence();

  // Oracle: no-retile ce_e on the owning twin (independent ground truth) and
  // on the arena twin.
  OwnArr1 ref =
      TA::einsum(A0o("i1,i2,k;a"), B0o("j1,j2,k;b"), "i1,i2,j1,j2;a,b");
  ArrayToT1 C_ref =
      TA::einsum(A0("i1,i2,k;a"), B0("j1,j2,k;b"), "i1,i2,j1,j2;a,b");
  w.gop.fence();
  BOOST_CHECK_SMALL(array_max_reldiff1(C_ref, ref), 1e-9);

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_e_calls.store(0);
  TA::detail::g_summa_gather_block_count.store(0);
  TA::detail::g_summa_plan_active_calls.store(0);
#endif

  // Active: collapse K to a single tile. Externals stay Identity.
  ArrayToT1 C_coarsen;
  C_coarsen("i1,i2,j1,j2;a,b") =
      (A0("i1,i2,k;a") * B0("j1,j2,k;b"))
          .retile(/*H=*/{}, /*M=*/{t, t}, /*N=*/{t, t}, /*K=*/{kT});
  w.gop.fence();

  // Values match the no-retile oracle.
  BOOST_CHECK_SMALL(array_max_reldiff1(C_coarsen, C_ref), 1e-9);

#ifdef TA_STRIDED_DGEMM_COUNT
  // The plan was active.
  std::size_t active = TA::detail::g_summa_plan_active_calls.load();
  w.gop.sum(&active, 1);
  BOOST_CHECK_GT(active, std::size_t{0});

  // The strided ce_e DGEMM fired (one fat GEMM per (m,n) over the packed K).
  std::size_t fires = TA::detail::g_strided_dgemm_ce_e_calls.load();
  w.gop.sum(&fires, 1);
  BOOST_CHECK_GT(fires, std::size_t{0});

  // The gather hook accumulates the number of FINE U K-tiles packed across
  // every coarse-cell gather. Each pack (one per local M-row in get_col, one
  // per local N-col in get_row) gathers exactly the fine K-block width k_fine.
  // With M = 2x2 = 4 grid rows and N = 2x2 = 4 grid cols at np=1, there are
  // (M + N) = 8 packs over the single coarse K-cell, so the total is
  // (M + N) * k_fine = 8 * 4 = 32. The invariant the hook proves: the gather
  // is a multiple of k_fine (every coarse cell packs the full fine block), and
  // the per-cell width is k_fine.
  const std::size_t M = 4, N = 4;  // i1*i2, j1*j2 grid extents
  std::size_t gathered = TA::detail::g_summa_gather_block_count.load();
  w.gop.sum(&gathered, 1);
  BOOST_CHECK_EQUAL(gathered, (M + N) * k_fine);
  BOOST_CHECK_EQUAL(gathered % k_fine, std::size_t{0});
#endif
}

// ---------------------------------------------------------------------------
// result-axis coarsen (SUMMA-M) for a ce+ce ToT
// contraction. C(i1,i2,j1;a) = A(i1,i2,k;a,c) * B(j1,k;c). The SUMMA-M result
// axis i1 is finely tiled in U; a .retile() coarsens it.
//
// This is the full ce_ce ride/M coarsen vertical slice: the engine builds a
// COARSE process grid from the T M/N tile counts (coarse_M_/coarse_N_), the
// left ride (M) operand U-block is gathered + packed into one coarse
// tile, and finalize reconciles the coarse grid to the U result trange via
// plan_.u_result_ordinals + arena_carve_block. Here the coarse grid is 2x2 (M
// coarsened: i1 4 tiles->1, times i2 2 tiles = 2; N = j1 2 tiles, identity)
// while the U result C(i1,i2,j1) is 4*2*2 = 16 tiles, so EACH coarse cell
// covers 4 U result tiles and finalize_active genuinely carves 4 -> 16. The
// reldiff<1e-9 over all 16 U tiles (a missing/mis-carved tile would blow it up)
// is the witness that the >1-U-tile-per-cell carve is correct.
//
// np=1 ONLY (this suite is @serial; np>1 active is rejected at coarse_M/N/K_).
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(retile_coarsen_m_ce_ce) {
  auto& w = TA::get_default_world();
  // U: i1 finely tiled (4 tiles of width 2 over [0,8)); i2,j1 small; K is one
  // tile. inner a=3 spectator, c=4 contracted on A; inner c=4 on B.
  auto i1U = tr1(8, 2);  // 4 U M-tiles on the ride axis
  auto i2 = tr1(4, 2);   // 2 tiles
  auto j1 = tr1(4, 2);   // 2 tiles
  auto kk = tr1(4, 4);   // single K tile
  auto i1T = tr1(8, 8);  // coarsen i1 to ONE tile (covers all 4 U M-tiles)

  ArrayToT1 A0 = make_arena1(w, TA::TiledRange{i1U, i2, kk}, TA::Range{3, 4});
  ArrayToT1 B0 = make_arena1(w, TA::TiledRange{j1, kk}, TA::Range{4});
  OwnArr1 A0o = make_own1(w, TA::TiledRange{i1U, i2, kk}, TA::Range{3, 4});
  OwnArr1 B0o = make_own1(w, TA::TiledRange{j1, kk}, TA::Range{4});
  w.gop.fence();

  // Oracle: no-retile ce_ce on the owning twin (independent ground truth) and
  // on the arena twin.
  OwnArr1 ref = TA::einsum(A0o("i1,i2,k;a,c"), B0o("j1,k;c"), "i1,i2,j1;a");
  ArrayToT1 C_ref =
      TA::einsum(A0("i1,i2,k;a,c"), B0("j1,k;c"), "i1,i2,j1;a");
  w.gop.fence();
  BOOST_CHECK_SMALL(array_max_reldiff1(C_ref, ref), 1e-9);

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
  TA::detail::g_summa_plan_active_calls.store(0);
#endif

  // Active: coarsen the SUMMA-M axis i1 to a single tile. i2,j1,K stay Identity.
  ArrayToT1 C_coarsen;
  C_coarsen("i1,i2,j1;a") =
      (A0("i1,i2,k;a,c") * B0("j1,k;c"))
          .retile(/*H=*/{}, /*M=*/{i1T, i2}, /*N=*/{j1}, /*K=*/{kk});
  w.gop.fence();

  // L3: result trange is the no-retile (U) trange.
  BOOST_CHECK(C_coarsen.trange() == C_ref.trange());

  // Values match the no-retile oracle.
  BOOST_CHECK_SMALL(array_max_reldiff1(C_coarsen, C_ref), 1e-9);

#ifdef TA_STRIDED_DGEMM_COUNT
  std::size_t active = TA::detail::g_summa_plan_active_calls.load();
  w.gop.sum(&active, 1);
  BOOST_CHECK_GT(active, std::size_t{0});

  // The strided ce_ce DGEMM (right or left arm) fired over the coarse M ride.
  std::size_t fires = TA::detail::g_strided_dgemm_ce_ce_right_calls.load() +
                      TA::detail::g_strided_dgemm_ce_ce_left_calls.load();
  w.gop.sum(&fires, 1);
  BOOST_CHECK_GT(fires, std::size_t{0});
#endif
}

BOOST_AUTO_TEST_SUITE_END()

// ===========================================================================
// bit-for-bit identity anchor at np=1 AND np=2.
//
// This suite carries NO label macro (neither @serial nor @distributed), so it
// is excluded by neither the run-np-1 filter (!@distributed) nor the run-np-2
// filter (!@serial): it runs at BOTH world sizes. The serial-labelled suite
// above runs only at np=1.
//
// The Summa ctor member split (fine=U / coarse=T) must be byte-identical to the
// stock SUMMA path whenever the retile plan is inactive. Each case computes a
// contraction with .retile() (empty + own-U identity targets, both inactive
// plans) and asserts the result is EXACTLY equal (reldiff == 0.0) to the
// no-retile path, at both np=1 and np=2. The identity anchor: the plan-active counter
// (incremented only inside plan_.active branches) must read 0 -- proving no
// active code path executed on the inactive runs.
// ===========================================================================
namespace {
// Self-contained helpers (the suite above scopes its make_arena1/etc. inside
// summa_two_trange_suite, so they are not visible here). 2-suffixed twins.
TA::TiledRange1 tr1d(std::size_t n, std::size_t ts) {
  if (ts == 0 || ts > n) ts = n;
  std::vector<std::size_t> b;
  for (std::size_t x = 0; x < n; x += ts) b.push_back(x);
  b.push_back(n);
  return TA::TiledRange1(b.begin(), b.end());
}
template <typename Index>
double outer_seed2(const Index& oix) {
  double seed = 0.0, f = 1.0;
  for (auto c : oix) {
    seed += double(c) * f;
    f *= 31.0;
  }
  return seed;
}
inline double cell_val2(double outer_s, std::size_t e) {
  return 1.0 + 1e-3 * double(e) + 1e-2 * outer_s;
}

using ArenaInner2 = TA::ArenaTensor<double, TA::Range>;
using ArenaOuter2 = TA::Tensor<ArenaInner2>;
using ArrayToT2 = TA::DistArray<ArenaOuter2, TA::DensePolicy>;
using OwnInner2 = TA::Tensor<double>;
using OwnOuter2 = TA::Tensor<OwnInner2>;
using OwnArr2 = TA::DistArray<OwnOuter2, TA::DensePolicy>;

ArrayToT2 make_arena2(TA::World& w, const TA::TiledRange& tr,
                      const TA::Range& inner) {
  ArrayToT2 x(w, tr);
  x.init_tiles([inner](const TA::Range& t_outer) {
    ArenaOuter2 t = TA::detail::arena_outer_init<ArenaOuter2>(
        t_outer, 1, [inner](std::size_t) { return inner; });
    std::size_t o = 0;
    for (const auto& idx : t_outer) {
      ArenaInner2& c = t.data()[o++];
      if (!c) continue;
      const double s = outer_seed2(idx);
      for (std::size_t e = 0; e < c.size(); ++e) c.data()[e] = cell_val2(s, e);
    }
    return t;
  });
  return x;
}
OwnArr2 make_own2(TA::World& w, const TA::TiledRange& tr,
                  const TA::Range& inner) {
  OwnArr2 x(w, tr);
  x.init_tiles([inner](const TA::Range& t_outer) {
    OwnOuter2 t(t_outer);
    std::size_t o = 0;
    for (const auto& idx : t_outer) {
      OwnInner2 cell(inner);
      const double s = outer_seed2(idx);
      for (std::size_t e = 0; e < cell.size(); ++e)
        cell.data()[e] = cell_val2(s, e);
      t.data()[o++] = cell;
    }
    return t;
  });
  return x;
}

template <typename ArrA, typename ArrB>
double array_max_reldiff2(const ArrA& A, const ArrB& B) {
  double d = 0.0;
  for (auto it = A.begin(); it != A.end(); ++it) {
    const auto a_tile = (*it).get();
    if (a_tile.empty()) continue;
    const auto b_tile = B.find(it.index()).get();
    if (a_tile.size() != b_tile.size()) {
      d = std::max(d, 1.0);
      continue;
    }
    for (std::size_t o = 0; o < a_tile.size(); ++o) {
      const auto& ac = a_tile[o];
      const auto& bc = b_tile[o];
      if (ac.size() != bc.size()) {
        d = std::max(d, 1.0);
        continue;
      }
      for (std::size_t e = 0; e < ac.size(); ++e) {
        const double av = double(ac.data()[e]), bv = double(bc.data()[e]);
        d = std::max(d, std::abs(av - bv) / std::max(1.0, std::abs(bv)));
      }
    }
  }
  A.world().gop.max(&d, 1);
  return d;
}

// Read the plan-active counter, summed across ranks. Returns 0 when the
// counter is compiled out (build-test defines TA_STRIDED_DGEMM_COUNT).
std::size_t plan_active_count(TA::World& w) {
#ifdef TA_STRIDED_DGEMM_COUNT
  std::size_t c = TA::detail::g_summa_plan_active_calls.load();
  w.gop.sum(&c, 1);
  return c;
#else
  (void)w;
  return 0;
#endif
}
void reset_plan_active_count() {
#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_summa_plan_active_calls.store(0);
#endif
}
}  // namespace

BOOST_AUTO_TEST_SUITE(summa_two_trange_dist_suite)

// ce+e: pure Contraction ToT x ToT (inner OUTER-product, no inner contraction).
// C(i1,i2,j1,j2;a,b) = A(i1,i2,k;a) * B(j1,j2,k;b). Inactive .retile() (empty
// and own-U targets) must be EXACTLY equal to the no-retile result, at np=1 and
// np=2; the plan-active counter must read 0.
BOOST_AUTO_TEST_CASE(retile_identity_ce_e_dist) {
  auto& w = TA::get_default_world();
  auto t = tr1d(4, 2), kk = tr1d(4, 4);
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{t, t, kk}, TA::Range{3});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{t, t, kk}, TA::Range{5});
  w.gop.fence();

  reset_plan_active_count();

  // no-retile reference.
  ArrayToT2 C_ref =
      TA::einsum(A0("i1,i2,k;a"), B0("j1,j2,k;b"), "i1,i2,j1,j2;a,b");
  w.gop.fence();

  // .retile() with EMPTY targets -> plan inactive -> bit-for-bit identical.
  ArrayToT2 C_empty;
  C_empty("i1,i2,j1,j2;a,b") =
      (A0("i1,i2,k;a") * B0("j1,j2,k;b")).retile({}, {}, {}, {});
  w.gop.fence();
  BOOST_CHECK_EQUAL(array_max_reldiff2(C_empty, C_ref), 0.0);

  // .retile() with targets == the operands' own U tilings -> identity.
  ArrayToT2 C_self;
  C_self("i1,i2,j1,j2;a,b") =
      (A0("i1,i2,k;a") * B0("j1,j2,k;b"))
          .retile(/*H=*/{}, /*M=*/{t, t}, /*N=*/{t, t}, /*K=*/{kk});
  w.gop.fence();
  BOOST_CHECK_EQUAL(array_max_reldiff2(C_self, C_ref), 0.0);

  // identity anchor: no active branch executed.
  BOOST_CHECK_EQUAL(plan_active_count(w), std::size_t{0});
}

// ce+ce: ToT x ToT with an INNER contraction (the hce_ce-family shape, minus
// the Hadamard modes to keep the np=2 grid simple). i1,i2 left externals; j1
// right external; k outer-contracted; inner c contracted, inner a spectator.
// C(i1,i2,j1;a) = A(i1,i2,k;a,c) * B(j1,k;c). Inactive .retile() (empty and
// own-U targets) must be EXACTLY equal to the no-retile result, np=1 and np=2.
BOOST_AUTO_TEST_CASE(retile_identity_ce_ce_dist) {
  auto& w = TA::get_default_world();
  auto t = tr1d(4, 2), kk = tr1d(4, 4);
  // inner a=3 (spectator), c=4 (contracted) on A; inner c=4 on B.
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{t, t, kk}, TA::Range{3, 4});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{t, kk}, TA::Range{4});
  w.gop.fence();

  reset_plan_active_count();

  ArrayToT2 C_ref = TA::einsum(A0("i1,i2,k;a,c"), B0("j1,k;c"), "i1,i2,j1;a");
  w.gop.fence();

  // EMPTY targets -> inactive.
  ArrayToT2 C_empty;
  C_empty("i1,i2,j1;a") =
      (A0("i1,i2,k;a,c") * B0("j1,k;c")).retile({}, {}, {}, {});
  w.gop.fence();
  BOOST_CHECK_EQUAL(array_max_reldiff2(C_empty, C_ref), 0.0);

  // own-U targets -> identity. M = {i1,i2} (left externals), N = {j1} (right
  // external), K = {k} (outer-contracted).
  ArrayToT2 C_self;
  C_self("i1,i2,j1;a") =
      (A0("i1,i2,k;a,c") * B0("j1,k;c"))
          .retile(/*H=*/{}, /*M=*/{t, t}, /*N=*/{t}, /*K=*/{kk});
  w.gop.fence();
  BOOST_CHECK_EQUAL(array_max_reldiff2(C_self, C_ref), 0.0);

  BOOST_CHECK_EQUAL(plan_active_count(w), std::size_t{0});
}

// Guard: an ACTIVE .retile() (here a K-coarsen collapse) is currently supported
// only at MPI world size 1. At np>1 the active inbound-coarsen path would gather
// fine U operand tiles while the SUMMA broadcast still keys/roots on the coarse
// K geometry, silently corrupting the result; the engine rejects it with a
// collective TA::Exception (coarse_K_ in cont_engine.h, gated on
// world.size()>1). At np=1 the same active K-coarsen must succeed. This suite
// runs at both world sizes, so the assertion branches on the world size.
BOOST_AUTO_TEST_CASE(retile_active_np_gt_1_rejects) {
  auto& w = TA::get_default_world();
  auto t = tr1d(4, 2);
  auto kU = tr1d(8, 2);   // 4 fine U K-tiles
  auto kT = tr1d(8, 8);   // collapse to 1 coarse K-tile (active plan)
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{t, t, kU}, TA::Range{3});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{t, t, kU}, TA::Range{5});
  w.gop.fence();

  if (w.size() > 1) {
    // The throw is symmetric (every rank evaluates plan_.active && size>1), so
    // it is collective -- safe under BOOST_REQUIRE_THROW.
    ArrayToT2 C;
    BOOST_REQUIRE_THROW(
        (C("i1,i2,j1,j2;a,b") =
             (A0("i1,i2,k;a") * B0("j1,j2,k;b"))
                 .retile(/*H=*/{}, /*M=*/{t, t}, /*N=*/{t, t}, /*K=*/{kT})),
        TA::Exception);
  } else {
    // np=1: the active K-coarsen must succeed and match the no-retile oracle.
    ArrayToT2 C_ref =
        TA::einsum(A0("i1,i2,k;a"), B0("j1,j2,k;b"), "i1,i2,j1,j2;a,b");
    w.gop.fence();
    ArrayToT2 C;
    C("i1,i2,j1,j2;a,b") = (A0("i1,i2,k;a") * B0("j1,j2,k;b"))
                               .retile(/*H=*/{}, /*M=*/{t, t}, /*N=*/{t, t},
                                       /*K=*/{kT});
    w.gop.fence();
    BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  }
}

BOOST_AUTO_TEST_SUITE_END()
