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

BOOST_AUTO_TEST_SUITE_END()
