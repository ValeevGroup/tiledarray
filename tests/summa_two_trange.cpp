/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
 *
 *  Unit tests for the pure per-axis nesting plan and the
 *  T-grid -> U-result ordinal map (contraction_retile.h). No World needed:
 *  TiledRange1 / TiledRange / GemmHelper are all built directly.
 */

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

BOOST_AUTO_TEST_SUITE_END()
