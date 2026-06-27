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
#include <TiledArray/tensor/dense_retile.h>
#include <TiledArray/tensor/arena_tensor.h>
#include <TiledArray/tensor/tensor.h>
#include <TiledArray/tiled_range.h>
#include <TiledArray/tiled_range1.h>

#include "unit_test_config.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <random>
#include <string>
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

// Dense analogue of arena_gather_nonuniform: dense_gather_block packs fine plain
// TA::Tensor tiles into one contiguous coarse tile, leaving holes (positions no
// fine tile covers) zero. This is the mixed-path operand packer (dense_retile.h).
BOOST_AUTO_TEST_CASE(dense_gather_block_basic) {
  using Tensor = TA::Tensor<double>;
  TA::Range coarse({0, 0}, {4, 6});
  std::vector<TA::Range> franges = {TA::Range({0, 0}, {2, 3}),
                                    TA::Range({0, 3}, {2, 6}),
                                    TA::Range({2, 0}, {4, 3})};  // (2..4,3..6) absent
  auto val = [](long i, long j) { return double(100 * i + j); };
  std::vector<Tensor> fine;
  for (auto& r : franges) {
    Tensor t(r);
    for (std::size_t p = 0; p < r.volume(); ++p) {
      auto ix = r.idx(p);
      t.data()[p] = val(ix[0], ix[1]);
    }
    fine.push_back(t);
  }
  Tensor c = TA::detail::dense_gather_block(fine, coarse, 1);
  for (std::size_t p = 0; p < coarse.volume(); ++p) {
    auto ix = coarse.idx(p);
    long i = ix[0], j = ix[1];
    const bool in_hole = (i >= 2 && j >= 3);
    BOOST_CHECK_EQUAL(c.data()[p], in_hole ? 0.0 : val(i, j));
  }
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

// ---------------------------------------------------------------------------
// REFINE-K (free operand split). ce_e ToT contraction
// C(i1,i2,j1,j2;a,b) = A(i1,i2,k;a) * B(j1,j2,k;b) with the OUTER contracted
// index k tiled as ONE U tile [0,8); a .retile() REFINES K into 4 T tiles
// (width 2). The externals (M={i1,i2}, N={j1,j2}) stay Identity.
//
// HAND PREDICTION: coarse grid M=2*2=4, N=2*2=4, K_coarse=4 (4 T K-tiles).
// np=1 1x1 grid. SUMMA steps over the 4 T K-cells. For each T K-cell `kc` the
// (single) U K-tile [0,8) is view-split to the T K-box [2kc,2kc+2) per operand
// outer cell, packed, and the 4 partial GEMMs accumulate over K -- exactly
// reproducing the single full-K GEMM. Result trange == U (1 K tile gone from
// the result anyway; the result is M x N = 16 U tiles). The result axes are
// Identity, so each coarse cell maps 1:1 to a distinct U tile -- the MERGE
// path must NOT engage (g_summa_result_merge_count == 0).
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(retile_refine_k_ce_e) {
  auto& w = TA::get_default_world();
  auto t = tr1(4, 2);    // 2 external tiles per axis
  auto kU = tr1(8, 8);   // ONE U K-tile [0,8)
  auto kT = tr1(8, 2);   // REFINE K -> 4 T tiles of width 2

  ArrayToT1 A0 = make_arena1(w, TA::TiledRange{t, t, kU}, TA::Range{3});
  ArrayToT1 B0 = make_arena1(w, TA::TiledRange{t, t, kU}, TA::Range{5});
  OwnArr1 A0o = make_own1(w, TA::TiledRange{t, t, kU}, TA::Range{3});
  OwnArr1 B0o = make_own1(w, TA::TiledRange{t, t, kU}, TA::Range{5});
  w.gop.fence();

  OwnArr1 ref =
      TA::einsum(A0o("i1,i2,k;a"), B0o("j1,j2,k;b"), "i1,i2,j1,j2;a,b");
  ArrayToT1 C_ref =
      TA::einsum(A0("i1,i2,k;a"), B0("j1,j2,k;b"), "i1,i2,j1,j2;a,b");
  w.gop.fence();
  BOOST_CHECK_SMALL(array_max_reldiff1(C_ref, ref), 1e-9);

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_summa_plan_active_calls.store(0);
  TA::detail::g_summa_result_merge_count.store(0);
#endif

  ArrayToT1 C_refine;
  C_refine("i1,i2,j1,j2;a,b") =
      (A0("i1,i2,k;a") * B0("j1,j2,k;b"))
          .retile(/*H=*/{}, /*M=*/{t, t}, /*N=*/{t, t}, /*K=*/{kT});
  w.gop.fence();

  // L3: result trange is the no-retile (U) trange.
  BOOST_CHECK(C_refine.trange() == C_ref.trange());
  // Values match the no-retile oracle.
  BOOST_CHECK_SMALL(array_max_reldiff1(C_refine, C_ref), 1e-9);

#ifdef TA_STRIDED_DGEMM_COUNT
  std::size_t active = TA::detail::g_summa_plan_active_calls.load();
  w.gop.sum(&active, 1);
  BOOST_CHECK_GT(active, std::size_t{0});

  // Refine on an OPERAND (K) axis is the FREE direction: NO outbound merge.
  std::size_t merged = TA::detail::g_summa_result_merge_count.load();
  w.gop.sum(&merged, 1);
  BOOST_CHECK_EQUAL(merged, std::size_t{0});
#endif
}

// ---------------------------------------------------------------------------
// REFINE a RESULT axis (SUMMA-N). ce_ce ToT contraction
// C(i1,i2,j1;a) = A(i1,i2,k;a,c) * B(j1,k;c). The SUMMA-N result axis j1 is
// ONE U tile [0,4); a .retile() REFINES it into 2 T tiles (width 2).
//
// HAND PREDICTION: coarse grid M=2*2=4 (i1,i2 identity), N_coarse=2 (2 T
// N-tiles), K=1. T result grid = 4*2 = 8 cells. U result C(i1,i2,j1) = 2*2*1
// = 4 U tiles. Each U tile (i1,i2,j1=0) is covered by the 2 T cells sharing
// (i1,i2) with N in {0,1} => 2 T result sub-pages MERGE into 1 U tile. So the
// MERGE path ENGAGES: g_summa_result_merge_count == 8 (4 U tiles x 2 T
// sub-pages each). Right operand B(j1,k) is view-split on N to the T N-box.
// Oracle match < 1e-9.
//
// witness: a PURE-COARSEN config (same shapes as retile_coarsen_m_ce_ce,
// M coarsened, no result axis refined) keeps the SAME counter at 0.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(retile_refine_n_ce_ce) {
  auto& w = TA::get_default_world();
  auto i1 = tr1(4, 2);   // 2 M-tiles
  auto i2 = tr1(4, 2);   // 2 M-tiles
  auto j1U = tr1(4, 4);  // ONE U N-tile [0,4)
  auto j1T = tr1(4, 2);  // REFINE N -> 2 T tiles of width 2
  auto kk = tr1(4, 4);   // single K tile

  ArrayToT1 A0 = make_arena1(w, TA::TiledRange{i1, i2, kk}, TA::Range{3, 4});
  ArrayToT1 B0 = make_arena1(w, TA::TiledRange{j1U, kk}, TA::Range{4});
  OwnArr1 A0o = make_own1(w, TA::TiledRange{i1, i2, kk}, TA::Range{3, 4});
  OwnArr1 B0o = make_own1(w, TA::TiledRange{j1U, kk}, TA::Range{4});
  w.gop.fence();

  OwnArr1 ref = TA::einsum(A0o("i1,i2,k;a,c"), B0o("j1,k;c"), "i1,i2,j1;a");
  ArrayToT1 C_ref =
      TA::einsum(A0("i1,i2,k;a,c"), B0("j1,k;c"), "i1,i2,j1;a");
  w.gop.fence();
  BOOST_CHECK_SMALL(array_max_reldiff1(C_ref, ref), 1e-9);

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_summa_plan_active_calls.store(0);
  TA::detail::g_summa_result_merge_count.store(0);
#endif

  ArrayToT1 C_refine;
  C_refine("i1,i2,j1;a") =
      (A0("i1,i2,k;a,c") * B0("j1,k;c"))
          .retile(/*H=*/{}, /*M=*/{i1, i2}, /*N=*/{j1T}, /*K=*/{kk});
  w.gop.fence();

  // L3: result trange is the no-retile (U) trange (N back at U: 1 tile).
  BOOST_CHECK(C_refine.trange() == C_ref.trange());
  // Values match the no-retile oracle.
  BOOST_CHECK_SMALL(array_max_reldiff1(C_refine, C_ref), 1e-9);

#ifdef TA_STRIDED_DGEMM_COUNT
  std::size_t active = TA::detail::g_summa_plan_active_calls.load();
  w.gop.sum(&active, 1);
  BOOST_CHECK_GT(active, std::size_t{0});

  // Refine on a RESULT axis ENGAGES the outbound merge: 4 U tiles each gather
  // 2 T sub-pages => 8.
  std::size_t merged = TA::detail::g_summa_result_merge_count.load();
  w.gop.sum(&merged, 1);
  BOOST_CHECK_EQUAL(merged, std::size_t{8});
#endif
}

// witness: a pure-coarsen config (no result axis refined) must keep the
// result-merge counter at 0. Reuses retile_coarsen_m_ce_ce's shapes.
BOOST_AUTO_TEST_CASE(retile_coarsen_no_merge_l5) {
  auto& w = TA::get_default_world();
  auto i1U = tr1(8, 2);  // 4 U M-tiles on the ride axis
  auto i2 = tr1(4, 2);   // 2 tiles
  auto j1 = tr1(4, 2);   // 2 tiles
  auto kk = tr1(4, 4);   // single K tile
  auto i1T = tr1(8, 8);  // coarsen i1 to ONE tile

  ArrayToT1 A0 = make_arena1(w, TA::TiledRange{i1U, i2, kk}, TA::Range{3, 4});
  ArrayToT1 B0 = make_arena1(w, TA::TiledRange{j1, kk}, TA::Range{4});
  ArrayToT1 C_ref =
      TA::einsum(A0("i1,i2,k;a,c"), B0("j1,k;c"), "i1,i2,j1;a");
  w.gop.fence();

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_summa_result_merge_count.store(0);
#endif

  ArrayToT1 C_coarsen;
  C_coarsen("i1,i2,j1;a") =
      (A0("i1,i2,k;a,c") * B0("j1,k;c"))
          .retile(/*H=*/{}, /*M=*/{i1T, i2}, /*N=*/{j1}, /*K=*/{kk});
  w.gop.fence();
  BOOST_CHECK_SMALL(array_max_reldiff1(C_coarsen, C_ref), 1e-9);

#ifdef TA_STRIDED_DGEMM_COUNT
  // L5: a pure-coarsen config never engages the outbound merge.
  std::size_t merged = TA::detail::g_summa_result_merge_count.load();
  w.gop.sum(&merged, 1);
  BOOST_CHECK_EQUAL(merged, std::size_t{0});
#endif
}

// ---------------------------------------------------------------------------
// review (cross-step operand re-fetch). The per-gather dedupe cache
// in get_col_coarsen/get_row_coarsen lives per get_col/get_row CALL (one call
// per SUMMA-K step). The reviewer flagged that with k_>1 (multiple SUMMA-K
// steps) AND a refined result axis (the outbound-merge path engaged), a shared
// U operand tile could be re-fetched ACROSS steps and re-arm the lazy-operand
// over-notify deadlock that the per-call cache fixed within a step.
//
// (A) k_=2 via K-IDENTITY (2 U K-tiles -> 2 SUMMA steps) + REFINE result axis
// N. ce_ce C(i1,i2,j1;a) = A(i1,i2,k;a,c) * B(j1,k;c). Each K step selects a
// DISTINCT U K-tile, so the right operand B(j1,k) U tiles fetched in step 0
// (k_u=0) and step 1 (k_u=1) are disjoint -- no cross-step re-fetch -- while
// the refined N within each step still shares B's N tile across the 2 N-cols
// (per-call cache covers that). Result merge engages (j1 refined): 4 U result
// tiles x 2 T N-sub-pages = 8. Must run green (no hang) + oracle match.
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(retile_refine_n_multistep_k_ce_ce) {
  auto& w = TA::get_default_world();
  auto i1 = tr1(4, 2);   // 2 M-tiles
  auto i2 = tr1(4, 2);   // 2 M-tiles
  auto j1U = tr1(4, 4);  // ONE U N-tile [0,4)
  auto j1T = tr1(4, 2);  // REFINE N -> 2 T tiles of width 2
  auto kU = tr1(8, 4);   // 2 U K-tiles
  auto kT = tr1(8, 4);   // K IDENTITY -> 2 T K-tiles => k_=2 SUMMA steps

  ArrayToT1 A0 = make_arena1(w, TA::TiledRange{i1, i2, kU}, TA::Range{3, 4});
  ArrayToT1 B0 = make_arena1(w, TA::TiledRange{j1U, kU}, TA::Range{4});
  OwnArr1 A0o = make_own1(w, TA::TiledRange{i1, i2, kU}, TA::Range{3, 4});
  OwnArr1 B0o = make_own1(w, TA::TiledRange{j1U, kU}, TA::Range{4});
  w.gop.fence();

  OwnArr1 ref = TA::einsum(A0o("i1,i2,k;a,c"), B0o("j1,k;c"), "i1,i2,j1;a");
  ArrayToT1 C_ref = TA::einsum(A0("i1,i2,k;a,c"), B0("j1,k;c"), "i1,i2,j1;a");
  w.gop.fence();
  BOOST_CHECK_SMALL(array_max_reldiff1(C_ref, ref), 1e-9);

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_summa_plan_active_calls.store(0);
  TA::detail::g_summa_result_merge_count.store(0);
#endif

  ArrayToT1 C_refine;
  C_refine("i1,i2,j1;a") =
      (A0("i1,i2,k;a,c") * B0("j1,k;c"))
          .retile(/*H=*/{}, /*M=*/{i1, i2}, /*N=*/{j1T}, /*K=*/{kT});
  w.gop.fence();

  BOOST_CHECK(C_refine.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff1(C_refine, C_ref), 1e-9);

#ifdef TA_STRIDED_DGEMM_COUNT
  std::size_t active = TA::detail::g_summa_plan_active_calls.load();
  w.gop.sum(&active, 1);
  BOOST_CHECK_GT(active, std::size_t{0});
  // 4 U result tiles each gather 2 T N-sub-pages => 8 (independent of k_).
  std::size_t merged = TA::detail::g_summa_result_merge_count.load();
  w.gop.sum(&merged, 1);
  BOOST_CHECK_EQUAL(merged, std::size_t{8});
#endif
}

// ---------------------------------------------------------------------------
// (B) The strongest cross-step re-fetch: REFINE-K (ONE U K-tile -> 2 T K-tiles
// => k_=2 SUMMA steps that BOTH map back to the SAME single U K-tile) combined
// with REFINE result axis N. ce_ce C(i1,i2,j1;a) = A(i1,i2,k;a,c) * B(j1,k;c).
// Here step 0 and step 1 of the SUMMA-K loop each view-split the SAME U K-tile
// [0,8), so the right operand B(j1,k=0) AND left operand A(.,.,k=0) U tiles are
// fetched in BOTH steps -- the genuine cross-step re-fetch. If the lazy-operand
// over-notify were re-armed across steps this would HANG; it must run green and
// match the oracle. (refine_k_ce_e already exercises the 4-step operand-axis
// re-fetch; this adds the result-merge path on top.)
// ---------------------------------------------------------------------------
BOOST_AUTO_TEST_CASE(retile_refine_nk_ce_ce) {
  auto& w = TA::get_default_world();
  auto i1 = tr1(4, 2);   // 2 M-tiles
  auto i2 = tr1(4, 2);   // 2 M-tiles
  auto j1U = tr1(4, 4);  // ONE U N-tile [0,4)
  auto j1T = tr1(4, 2);  // REFINE N -> 2 T tiles of width 2
  auto kU = tr1(8, 8);   // ONE U K-tile [0,8)
  auto kT = tr1(8, 4);   // REFINE K -> 2 T K-tiles => k_=2 steps, same U K-tile

  ArrayToT1 A0 = make_arena1(w, TA::TiledRange{i1, i2, kU}, TA::Range{3, 4});
  ArrayToT1 B0 = make_arena1(w, TA::TiledRange{j1U, kU}, TA::Range{4});
  OwnArr1 A0o = make_own1(w, TA::TiledRange{i1, i2, kU}, TA::Range{3, 4});
  OwnArr1 B0o = make_own1(w, TA::TiledRange{j1U, kU}, TA::Range{4});
  w.gop.fence();

  OwnArr1 ref = TA::einsum(A0o("i1,i2,k;a,c"), B0o("j1,k;c"), "i1,i2,j1;a");
  ArrayToT1 C_ref = TA::einsum(A0("i1,i2,k;a,c"), B0("j1,k;c"), "i1,i2,j1;a");
  w.gop.fence();
  BOOST_CHECK_SMALL(array_max_reldiff1(C_ref, ref), 1e-9);

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_summa_plan_active_calls.store(0);
  TA::detail::g_summa_result_merge_count.store(0);
#endif

  ArrayToT1 C_refine;
  C_refine("i1,i2,j1;a") =
      (A0("i1,i2,k;a,c") * B0("j1,k;c"))
          .retile(/*H=*/{}, /*M=*/{i1, i2}, /*N=*/{j1T}, /*K=*/{kT});
  w.gop.fence();

  BOOST_CHECK(C_refine.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff1(C_refine, C_ref), 1e-9);

#ifdef TA_STRIDED_DGEMM_COUNT
  std::size_t active = TA::detail::g_summa_plan_active_calls.load();
  w.gop.sum(&active, 1);
  BOOST_CHECK_GT(active, std::size_t{0});
  std::size_t merged = TA::detail::g_summa_result_merge_count.load();
  w.gop.sum(&merged, 1);
  BOOST_CHECK_EQUAL(merged, std::size_t{8});
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

// NON-UNIFORM inner-extent ToT operand: the inner cell Range varies with the
// outer index (the defining ToT property). ext_fn(outer_idx) -> inner Range.
// All captures are by value / synchronous (arena_outer_init's range callback
// runs inside this task), so there is no deferred-task dangling-capture risk.
template <typename ExtFn>
ArrayToT2 make_arena2_nu(TA::World& w, const TA::TiledRange& tr, ExtFn ext_fn) {
  ArrayToT2 x(w, tr);
  x.init_tiles([ext_fn](const TA::Range& t_outer) {
    ArenaOuter2 t = TA::detail::arena_outer_init<ArenaOuter2>(
        t_outer, 1, [&t_outer, &ext_fn](std::size_t ord) {
          const auto oix = t_outer.idx(ord);
          std::vector<std::size_t> v(oix.begin(), oix.end());
          return ext_fn(v);
        });
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

// Plain dense (non-ToT) DistArray for the MIXED kernel's plain ride operand g.
// Filled by ABSOLUTE element coordinate (outer_seed2 over the element index) so
// a coarse-packed block carries observably distinct values -- a mis-packed cell
// blows up the oracle reldiff. Mirrors strided_canonicalize.cpp::make_plain.
using PlainArr2 = TA::DistArray<TA::Tensor<double>, TA::DensePolicy>;
// [[maybe_unused]]: scaffolding for the deferred mixed-kernel active-retile fix
// (see the BLOCKED note below the sparse cases); no in-tree caller yet.
[[maybe_unused]] PlainArr2 make_plain2(TA::World& w, const TA::TiledRange& tr) {
  PlainArr2 g(w, tr);
  g.init_tiles([](const TA::Range& r) {
    TA::Tensor<double> t(r);
    for (const auto& idx : r) t(idx) = 1.0 + 1e-2 * outer_seed2(idx);
    return t;
  });
  return g;
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

// --- SparsePolicy arena-ToT harness ------------------------------
// Mirrors make_arena2/make_own2 but on a SparsePolicy array whose SparseShape
// zeros a chosen subset of OUTER tiles (the `is_zero(tile_outer_idx)` predicate
// returns true for absent tiles). Only non-zero tiles are filled. The threshold
// is std::numeric_limits<float>::min(), so any positive norm clears the bar.
using ArrayToTSparse = TA::DistArray<ArenaOuter2, TA::SparsePolicy>;

// Build a SparseShape over `tr` where `is_zero(tile_idx)` selects absent tiles.
template <typename IsZero>
TA::SparseShape<float> sparse_shape_from(const TA::TiledRange& tr,
                                         IsZero is_zero) {
  const auto& tiles = tr.tiles_range();
  TA::Tensor<float> norms(tiles, 0.0f);
  for (const auto& tidx : tiles) {
    if (!is_zero(tidx))
      norms[tiles.ordinal(tidx)] = 1.0f;  // present (unit per-element norm)
  }
  // do_not_scale=true: norms are taken as-is; exactly-zero tiles screen out.
  return TA::SparseShape<float>(norms, tr, /*do_not_scale=*/true);
}

template <typename IsZero>
ArrayToTSparse make_arena_sparse(TA::World& w, const TA::TiledRange& tr,
                                 const TA::Range& inner, IsZero is_zero) {
  ArrayToTSparse x(w, tr, sparse_shape_from(tr, is_zero));
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

// Sparse + NON-UNIFORM inner extents combined (the bench's exact domain): outer
// sparsity from is_zero, inner cell Range from ext_fn(outer_idx). All captures
// value/synchronous (no deferred-task dangling).
template <typename IsZero, typename ExtFn>
ArrayToTSparse make_arena_sparse_nu(TA::World& w, const TA::TiledRange& tr,
                                    IsZero is_zero, ExtFn ext_fn) {
  ArrayToTSparse x(w, tr, sparse_shape_from(tr, is_zero));
  x.init_tiles([ext_fn](const TA::Range& t_outer) {
    ArenaOuter2 t = TA::detail::arena_outer_init<ArenaOuter2>(
        t_outer, 1, [&t_outer, &ext_fn](std::size_t ord) {
          const auto oix = t_outer.idx(ord);
          std::vector<std::size_t> v(oix.begin(), oix.end());
          return ext_fn(v);
        });
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

// SparsePolicy sibling of make_plain2 for the mixed whole-coarse-zero step
// (the plain ride operand g must also be able to zero a coarse K block). Built
// collectively from `is_zero(tile_idx)` -- the threshold is set by the caller.
using PlainArrSparse = TA::DistArray<TA::Tensor<double>, TA::SparsePolicy>;
template <typename IsZero>
PlainArrSparse make_plain_sparse(TA::World& w, const TA::TiledRange& tr,
                                 IsZero is_zero) {
  PlainArrSparse g(w, tr, sparse_shape_from(tr, is_zero));
  g.init_tiles([](const TA::Range& r) {
    TA::Tensor<double> t(r);
    for (const auto& idx : r) t(idx) = 1.0 + 1e-2 * outer_seed2(idx);
    return t;
  });
  return g;
}

// Sparse-aware max relative diff: a tile present in exactly one of A/B (zero in
// the other) counts as a full mismatch, so the comparison cannot pass by both
// sides happening to omit the same tile.
template <typename ArrA, typename ArrB>
double array_max_reldiff_sparse(const ArrA& A, const ArrB& B) {
  double d = 0.0;
  const auto& tiles = A.trange().tiles_range();
  for (const auto& tidx : tiles) {
    const bool az = A.is_zero(tidx);
    const bool bz = B.is_zero(tidx);
    if (az && bz) continue;
    if (az != bz) {  // present on one side only
      d = std::max(d, 1.0);
      continue;
    }
    const auto a_tile = A.find(tidx).get();
    const auto b_tile = B.find(tidx).get();
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

// An ACTIVE COARSEN .retile() (here a K-coarsen collapse) must succeed and match
// the no-retile oracle at BOTH np=1 and np>1. At np>1 the engine distributes the
// operands and result on the COARSE T-grid (each U tile co-located on its
// covering coarse cell's owner), so the in-step gather stays local and the
// coarse-keyed broadcast is consistent. This suite runs at both world sizes; the
// same assertion holds at each.
BOOST_AUTO_TEST_CASE(retile_active_coarsen_np_gt_1_ok) {
  auto& w = TA::get_default_world();
  auto t = tr1d(4, 2);
  auto kU = tr1d(8, 2);   // 4 fine U K-tiles
  auto kT = tr1d(8, 8);   // collapse to 1 coarse K-tile (active plan)
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{t, t, kU}, TA::Range{3});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{t, t, kU}, TA::Range{5});
  w.gop.fence();

  reset_plan_active_count();

  ArrayToT2 C_ref =
      TA::einsum(A0("i1,i2,k;a"), B0("j1,j2,k;b"), "i1,i2,j1,j2;a,b");
  w.gop.fence();

  ArrayToT2 C;
  C("i1,i2,j1,j2;a,b") = (A0("i1,i2,k;a") * B0("j1,j2,k;b"))
                             .retile(/*H=*/{}, /*M=*/{t, t}, /*N=*/{t, t},
                                     /*K=*/{kT});
  w.gop.fence();
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
}

// Guard: an ACTIVE .retile() that REFINES an axis is still rejected at np>1
// (deferred). Here K is a single U tile refined into several T tiles. The throw
// is symmetric across ranks (every rank evaluates plan_has_refine && size>1) so
// it is collective -- safe under BOOST_REQUIRE_THROW. At np=1 the same refine
// plan must succeed and match the oracle.
BOOST_AUTO_TEST_CASE(retile_active_refine_np_gt_1_rejects) {
  auto& w = TA::get_default_world();
  auto t = tr1d(4, 2);
  auto kU = tr1d(8, 8);   // ONE U K-tile [0,8)
  auto kT = tr1d(8, 2);   // REFINE K -> 4 T tiles (active refine plan)
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{t, t, kU}, TA::Range{3});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{t, t, kU}, TA::Range{5});
  w.gop.fence();

  if (w.size() > 1) {
    ArrayToT2 C;
    BOOST_REQUIRE_THROW(
        (C("i1,i2,j1,j2;a,b") =
             (A0("i1,i2,k;a") * B0("j1,j2,k;b"))
                 .retile(/*H=*/{}, /*M=*/{t, t}, /*N=*/{t, t}, /*K=*/{kT})),
        TA::Exception);
  } else {
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

// Promoted coarsen-K (ce+e) case: runs at np=1 AND np=2 (unlabeled dist suite).
// Mirror of summa_two_trange_suite/retile_coarsen_k_ce_e: collapse the 4 fine U
// K-tiles to ONE coarse K-tile; result/operands at U. Oracle match < 1e-9 and
// the plan was active. At np=2 this exercises the COARSE T-grid operand/result
// distribution (each U tile on its covering coarse cell's owner).
BOOST_AUTO_TEST_CASE(retile_coarsen_k_ce_e_dist) {
  auto& w = TA::get_default_world();
  auto t = tr1d(4, 2);
  auto kU = tr1d(8, 2);   // 4 fine U K-tiles
  auto kT = tr1d(8, 8);   // collapse to 1 coarse K-tile
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{t, t, kU}, TA::Range{3});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{t, t, kU}, TA::Range{5});
  w.gop.fence();

  reset_plan_active_count();

  ArrayToT2 C_ref =
      TA::einsum(A0("i1,i2,k;a"), B0("j1,j2,k;b"), "i1,i2,j1,j2;a,b");
  w.gop.fence();

  ArrayToT2 C;
  C("i1,i2,j1,j2;a,b") = (A0("i1,i2,k;a") * B0("j1,j2,k;b"))
                             .retile(/*H=*/{}, /*M=*/{t, t}, /*N=*/{t, t},
                                     /*K=*/{kT});
  w.gop.fence();
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
}

// Promoted coarsen-M (ce+ce) case: runs at np=1 AND np=2. Mirror of
// summa_two_trange_suite/retile_coarsen_m_ce_ce: coarsen the SUMMA-M ride axis
// i1 (4 U tiles -> 1 T tile). The coarse grid is COARSE on M while the U result
// C(i1,i2,j1) has more tiles, so finalize carves coarse -> U. At np=2 each
// coarse cell (and its covered U result tiles) lands on one owner.
BOOST_AUTO_TEST_CASE(retile_coarsen_m_ce_ce_dist) {
  auto& w = TA::get_default_world();
  auto i1U = tr1d(8, 2);  // 4 U M-tiles on the ride axis
  auto i2 = tr1d(4, 2);   // 2 tiles
  auto j1 = tr1d(4, 2);   // 2 tiles
  auto kk = tr1d(4, 4);   // single K tile
  auto i1T = tr1d(8, 8);  // coarsen i1 to ONE tile (covers all 4 U M-tiles)
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{i1U, i2, kk}, TA::Range{3, 4});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{j1, kk}, TA::Range{4});
  w.gop.fence();

  reset_plan_active_count();

  ArrayToT2 C_ref = TA::einsum(A0("i1,i2,k;a,c"), B0("j1,k;c"), "i1,i2,j1;a");
  w.gop.fence();

  ArrayToT2 C;
  C("i1,i2,j1;a") = (A0("i1,i2,k;a,c") * B0("j1,k;c"))
                        .retile(/*H=*/{}, /*M=*/{i1T, i2}, /*N=*/{j1},
                                /*K=*/{kk});
  w.gop.fence();
  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
}

// gap-fill: ce+e RESULT-AXIS coarsen (the K-coarsen sibling
// retile_coarsen_k_ce_e_dist only coarsens the contracted axis). Same ce+e
// contraction C(i1,i2,j1,j2;a,b) = A(i1,i2,k;a) * B(j1,j2,k;b); here the SUMMA-M
// left-external i1 (4 fine U tiles) coarsens to ONE coarse tile while i2,j1,j2,K
// stay identity. A coarse M-cell now covers >1 U result tile, so finalize carves
// the coarse grid back to the U result trange (the ce+e analogue of
// retile_coarsen_m_ce_ce's >1-U-tile-per-cell carve). Oracle = no-retile einsum;
// match < 1e-9, plan active, ce_e strided DGEMM fired. Runs np=1 AND np=2.
BOOST_AUTO_TEST_CASE(retile_coarsen_m_ce_e_dist) {
  auto& w = TA::get_default_world();
  auto i1U = tr1d(8, 2);  // 4 U M-tiles on the left-external ride axis
  auto i2 = tr1d(4, 2);   // 2 tiles
  auto j = tr1d(4, 2);    // 2 right-external tiles per axis
  auto kk = tr1d(4, 4);   // single K tile
  auto i1T = tr1d(8, 8);  // coarsen i1 to ONE tile (covers all 4 U M-tiles)
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{i1U, i2, kk}, TA::Range{3});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{j, j, kk}, TA::Range{5});
  w.gop.fence();

  reset_plan_active_count();
#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_e_calls.store(0);
#endif

  ArrayToT2 C_ref =
      TA::einsum(A0("i1,i2,k;a"), B0("j1,j2,k;b"), "i1,i2,j1,j2;a,b");
  w.gop.fence();

  ArrayToT2 C;
  C("i1,i2,j1,j2;a,b") = (A0("i1,i2,k;a") * B0("j1,j2,k;b"))
                             .retile(/*H=*/{}, /*M=*/{i1T, i2}, /*N=*/{j, j},
                                     /*K=*/{kk});
  w.gop.fence();
  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
#ifdef TA_STRIDED_DGEMM_COUNT
  std::size_t fires = TA::detail::g_strided_dgemm_ce_e_calls.load();
  w.gop.sum(&fires, 1);
  BOOST_CHECK_GT(fires, std::size_t{0});
#endif
}

// Permuted-result ACTIVE carve, NO Hadamard (non-general path): M={i1,i2},
// N={j1,j2}, K coarsened (active). The requested output i1,j1,i2,j2 INTERLEAVES
// M and N -> crosses the role boundary with >=2 axes per role, the exact shape
// that crashed the carve (agent/findings/2026-06-26-multiretile-permuted-carve-*).
// Result axes stay identity (only K coarsens) -> the 1:1 carve. Oracle = stock
// einsum; match < 1e-9, plan active. Runs np=1 AND np=2.
BOOST_AUTO_TEST_CASE(retile_active_permuted_carve_ce_e_dist) {
  auto& w = TA::get_default_world();
  auto i1 = tr1d(4, 2), i2 = tr1d(4, 2);   // 2 M-axes, 2 U tiles each
  auto j1 = tr1d(4, 2), j2 = tr1d(4, 2);   // 2 N-axes, 2 U tiles each
  auto kU = tr1d(8, 2);                    // 4 fine U K-tiles
  auto kT = tr1d(8, 8);                    // coarsen K: 4 U -> 1 T (active)
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{i1, i2, kU}, TA::Range{3});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{j1, j2, kU}, TA::Range{5});
  w.gop.fence();
  reset_plan_active_count();

  // cross-boundary permuted output: i1 (M), j1 (N), i2 (M), j2 (N).
  ArrayToT2 C_ref =
      TA::einsum(A0("i1,i2,k;a"), B0("j1,j2,k;b"), "i1,j1,i2,j2;a,b");
  w.gop.fence();

  ArrayToT2 C;
  C("i1,j1,i2,j2;a,b") = (A0("i1,i2,k;a") * B0("j1,j2,k;b"))
                             .retile(/*H=*/{}, /*M=*/{i1, i2}, /*N=*/{j1, j2},
                                     /*K=*/{kT});
  w.gop.fence();
  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
}

// Permuted-result ACTIVE carve where a RESULT axis also coarsens (>1 U tiles per
// coarse page): M={i1,i2} with i1 coarsened (4 U -> 1 T), N={j1,j2}; output
// i1,j1,i2,j2 crosses the role boundary. Exercises the general (subdividing)
// carve in the permuted layout, not just the 1:1 case. Oracle = stock einsum;
// match < 1e-9, plan active. Runs np=1 AND np=2.
BOOST_AUTO_TEST_CASE(retile_active_permuted_carve_coarsenM_ce_e_dist) {
  auto& w = TA::get_default_world();
  auto i1U = tr1d(8, 2);   // 4 U M-tiles on i1
  auto i2 = tr1d(4, 2);    // 2 U tiles
  auto j1 = tr1d(4, 2), j2 = tr1d(4, 2);
  auto kk = tr1d(4, 4);    // single K tile
  auto i1T = tr1d(8, 8);   // coarsen i1: 4 U -> 1 T (a coarse page covers 4 U M-tiles)
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{i1U, i2, kk}, TA::Range{3});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{j1, j2, kk}, TA::Range{5});
  w.gop.fence();
  reset_plan_active_count();

  ArrayToT2 C_ref =
      TA::einsum(A0("i1,i2,k;a"), B0("j1,j2,k;b"), "i1,j1,i2,j2;a,b");
  w.gop.fence();

  ArrayToT2 C;
  C("i1,j1,i2,j2;a,b") = (A0("i1,i2,k;a") * B0("j1,j2,k;b"))
                             .retile(/*H=*/{}, /*M=*/{i1T, i2}, /*N=*/{j1, j2},
                                     /*K=*/{kk});
  w.gop.fence();
  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
}

// Permuted-result ACTIVE retile where a RESULT axis REFINES (target finer than
// the U operand on a result axis) -> the outbound MERGE path (MergeSetTask), not
// the carve. M={i1,i2}, N={j1,j2} with j1 refined (1 U tile [0,4) -> 2 T tiles);
// output i1,j1,i2,j2 crosses the role boundary. Exercises the MergeSetTask
// reconciliation alongside the carve fix. Active refine is rejected at np>1, so:
// np=1 -> oracle einsum match < 1e-9, plan active; np>1 -> REQUIRE_THROW.
BOOST_AUTO_TEST_CASE(retile_active_permuted_refine_n_ce_e_dist) {
  auto& w = TA::get_default_world();
  auto i1 = tr1d(4, 2), i2 = tr1d(4, 2);  // 2 M-axes
  auto j1U = tr1d(4, 4);                   // ONE U N-tile [0,4) on j1
  auto j1T = tr1d(4, 2);                   // REFINE j1 -> 2 T tiles of width 2
  auto j2 = tr1d(4, 2);                    // 2 N-tiles on j2
  auto kk = tr1d(4, 4);                    // single K tile
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{i1, i2, kk}, TA::Range{3});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{j1U, j2, kk}, TA::Range{5});
  w.gop.fence();
  reset_plan_active_count();

  if (w.size() > 1) {
    ArrayToT2 C;
    BOOST_REQUIRE_THROW(
        (C("i1,j1,i2,j2;a,b") =
             (A0("i1,i2,k;a") * B0("j1,j2,k;b"))
                 .retile(/*H=*/{}, /*M=*/{i1, i2}, /*N=*/{j1T, j2}, /*K=*/{kk})),
        TA::Exception);
  } else {
    ArrayToT2 C_ref =
        TA::einsum(A0("i1,i2,k;a"), B0("j1,j2,k;b"), "i1,j1,i2,j2;a,b");
    w.gop.fence();
    ArrayToT2 C;
    C("i1,j1,i2,j2;a,b") = (A0("i1,i2,k;a") * B0("j1,j2,k;b"))
                               .retile(/*H=*/{}, /*M=*/{i1, i2}, /*N=*/{j1T, j2},
                                       /*K=*/{kk});
    w.gop.fence();
    BOOST_CHECK(C.trange() == C_ref.trange());
    BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
    BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
  }
}

// gap-fill: ce+ce COARSEN COMBO (M ride + K together), Dense. The
// existing dense ce_ce cases coarsen M alone (retile_coarsen_m_ce_ce) or K is
// only combined with M on the SPARSE zero-step path; this is the clean DENSE M+K
// combo. C(i1,i2,j1;a) = A(i1,i2,k;a,c) * B(j1,k;c). The SUMMA-M ride i1
// coarsens 4 U -> 1 T AND the contracted k coarsens 4 U -> 1 T, so the left ride
// operand packs both a coarse M-block and a coarse K-block in one fat strided
// GEMM. Oracle = no-retile einsum; match < 1e-9, plan active, ce_ce strided
// DGEMM fired. Runs np=1 AND np=2.
BOOST_AUTO_TEST_CASE(retile_coarsen_mk_ce_ce_dist) {
  auto& w = TA::get_default_world();
  auto i1U = tr1d(8, 2);  // 4 U M-tiles on the ride axis
  auto i2 = tr1d(4, 2);   // 2 tiles
  auto j1 = tr1d(4, 2);   // 2 tiles
  auto kU = tr1d(8, 2);   // 4 fine U K-tiles on the contracted axis
  auto i1T = tr1d(8, 8);  // coarsen i1: 4 U -> 1 T
  auto kT = tr1d(8, 8);   // coarsen k:  4 U -> 1 T
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{i1U, i2, kU}, TA::Range{3, 4});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{j1, kU}, TA::Range{4});
  w.gop.fence();

  reset_plan_active_count();
#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
#endif

  ArrayToT2 C_ref = TA::einsum(A0("i1,i2,k;a,c"), B0("j1,k;c"), "i1,i2,j1;a");
  w.gop.fence();

  ArrayToT2 C;
  C("i1,i2,j1;a") = (A0("i1,i2,k;a,c") * B0("j1,k;c"))
                        .retile(/*H=*/{}, /*M=*/{i1T, i2}, /*N=*/{j1},
                                /*K=*/{kT});
  w.gop.fence();
  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
#ifdef TA_STRIDED_DGEMM_COUNT
  std::size_t fires = TA::detail::g_strided_dgemm_ce_ce_right_calls.load() +
                      TA::detail::g_strided_dgemm_ce_ce_left_calls.load();
  w.gop.sum(&fires, 1);
  BOOST_CHECK_GT(fires, std::size_t{0});
#endif
}

// gap-fill: ce+e COARSEN-N (right-external result axis), Dense. The
// existing ce+e coarsen cases coarsen K (retile_coarsen_k_ce_e_dist) or the
// LEFT-external M ride (retile_coarsen_m_ce_e_dist); the SUMMA-N (right
// external) axis was never coarsened ALONE. C(i1,i2,j1,j2;a,b) =
// A(i1,i2,k;a) * B(j1,j2,k;b); here the right-external j1 (4 fine U tiles)
// coarsens to ONE coarse tile while i1,i2,j2,K stay identity. A coarse N-cell
// covers >1 U result tile, so finalize carves the coarse N grid back to the U
// result trange (the N analogue of the M carve in retile_coarsen_m_ce_e_dist).
// Oracle = no-retile einsum; match < 1e-9, plan active, ce_e strided DGEMM
// fired. Runs np=1 AND np=2.
BOOST_AUTO_TEST_CASE(retile_coarsen_n_ce_e_dist) {
  auto& w = TA::get_default_world();
  auto i = tr1d(4, 2);    // 2 left-external M-tiles per axis
  auto j1U = tr1d(8, 2);  // 4 U N-tiles on the right-external ride axis
  auto j2 = tr1d(4, 2);   // 2 right-external tiles
  auto kk = tr1d(4, 4);   // single K tile
  auto j1T = tr1d(8, 8);  // coarsen j1 to ONE tile (covers all 4 U N-tiles)
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{i, i, kk}, TA::Range{3});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{j1U, j2, kk}, TA::Range{5});
  w.gop.fence();

  reset_plan_active_count();
#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_e_calls.store(0);
#endif

  ArrayToT2 C_ref =
      TA::einsum(A0("i1,i2,k;a"), B0("j1,j2,k;b"), "i1,i2,j1,j2;a,b");
  w.gop.fence();

  ArrayToT2 C;
  C("i1,i2,j1,j2;a,b") = (A0("i1,i2,k;a") * B0("j1,j2,k;b"))
                             .retile(/*H=*/{}, /*M=*/{i, i}, /*N=*/{j1T, j2},
                                     /*K=*/{kk});
  w.gop.fence();
  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
#ifdef TA_STRIDED_DGEMM_COUNT
  std::size_t fires = TA::detail::g_strided_dgemm_ce_e_calls.load();
  w.gop.sum(&fires, 1);
  BOOST_CHECK_GT(fires, std::size_t{0});
#endif
}

// gap-fill: ce+ce COARSEN-N (right-external result axis), Dense. The
// dense ce_ce cases coarsen the M ride (retile_coarsen_m_ce_ce_dist) or M+K
// (retile_coarsen_mk_ce_ce_dist); the SUMMA-N right-external j1 was never
// coarsened alone. C(i1,i2,j1;a) = A(i1,i2,k;a,c) * B(j1,k;c). j1 coarsens 4 U
// -> 1 T while i1,i2,K stay identity. The right operand B(j1,k) packs a coarse
// N-block; finalize carves the coarse N grid back to the U result trange.
// Oracle = no-retile einsum; match < 1e-9, plan active, ce_ce strided DGEMM
// fired. Runs np=1 AND np=2.
BOOST_AUTO_TEST_CASE(retile_coarsen_n_ce_ce_dist) {
  auto& w = TA::get_default_world();
  auto i1 = tr1d(4, 2);   // 2 left-external M-tiles
  auto i2 = tr1d(4, 2);   // 2 left-external M-tiles
  auto j1U = tr1d(8, 2);  // 4 U N-tiles on the right-external ride axis
  auto kk = tr1d(4, 4);   // single K tile
  auto j1T = tr1d(8, 8);  // coarsen j1 to ONE tile (covers all 4 U N-tiles)
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{i1, i2, kk}, TA::Range{3, 4});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{j1U, kk}, TA::Range{4});
  w.gop.fence();

  reset_plan_active_count();
#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
#endif

  ArrayToT2 C_ref = TA::einsum(A0("i1,i2,k;a,c"), B0("j1,k;c"), "i1,i2,j1;a");
  w.gop.fence();

  ArrayToT2 C;
  C("i1,i2,j1;a") = (A0("i1,i2,k;a,c") * B0("j1,k;c"))
                        .retile(/*H=*/{}, /*M=*/{i1, i2}, /*N=*/{j1T},
                                /*K=*/{kk});
  w.gop.fence();
  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
#ifdef TA_STRIDED_DGEMM_COUNT
  std::size_t fires = TA::detail::g_strided_dgemm_ce_ce_right_calls.load() +
                      TA::detail::g_strided_dgemm_ce_ce_left_calls.load();
  w.gop.sum(&fires, 1);
  BOOST_CHECK_GT(fires, std::size_t{0});
#endif
}

// gap-fill: SPARSE identity anchor (ce+e). The identity anchors so far
// (retile_identity_ce_e_dist / retile_identity_ce_ce_dist) are all DensePolicy;
// the SparsePolicy inactive-plan path -- a .retile() whose targets coincide with
// the operands' own U tilings on a SparsePolicy array -- had no bit-for-bit
// anchor. With some OUTER tiles screened out (the left operand zeros the k==1
// U K-tile, the right operand stays dense), the inactive plan must take the
// byte-for-byte STOCK SUMMA path: result EXACTLY equal (sparse reldiff == 0.0)
// to the no-retile result AND plan_active_count == 0 (no active branch ran).
// Shape built collectively => rank-stable, np=1 AND np=2.
BOOST_AUTO_TEST_CASE(retile_identity_sparse_ce_e_dist) {
  auto& w = TA::get_default_world();
  const float saved_threshold = TA::SparseShape<float>::threshold();
  w.gop.serial_invoke([] {
    TA::SparseShape<float>::threshold(std::numeric_limits<float>::min());
  });

  auto t = tr1d(4, 2);   // 2 external tiles per axis
  auto kk = tr1d(4, 2);  // 2 K-tiles (idx 0,1)
  // Left A(i1,i2,k): zero the k==1 U K-tile (a screened-out outer tile, so the
  // sparse shape is non-trivial); right B fully dense.
  ArrayToTSparse A0 = make_arena_sparse(
      w, TA::TiledRange{t, t, kk}, TA::Range{3},
      [](const auto& tidx) { return tidx[2] == 1; });
  ArrayToTSparse B0 = make_arena_sparse(w, TA::TiledRange{t, t, kk},
                                        TA::Range{5},
                                        [](const auto&) { return false; });
  w.gop.fence();

  reset_plan_active_count();

  ArrayToTSparse C_ref =
      TA::einsum(A0("i1,i2,k;a"), B0("j1,j2,k;b"), "i1,i2,j1,j2;a,b");
  w.gop.fence();

  // EMPTY targets -> inactive -> bit-for-bit stock.
  ArrayToTSparse C_empty;
  C_empty("i1,i2,j1,j2;a,b") =
      (A0("i1,i2,k;a") * B0("j1,j2,k;b")).retile({}, {}, {}, {});
  w.gop.fence();
  BOOST_CHECK_EQUAL(array_max_reldiff_sparse(C_empty, C_ref), 0.0);

  // own-U targets -> identity -> still bit-for-bit stock.
  ArrayToTSparse C_self;
  C_self("i1,i2,j1,j2;a,b") =
      (A0("i1,i2,k;a") * B0("j1,j2,k;b"))
          .retile(/*H=*/{}, /*M=*/{t, t}, /*N=*/{t, t}, /*K=*/{kk});
  w.gop.fence();
  BOOST_CHECK_EQUAL(array_max_reldiff_sparse(C_self, C_ref), 0.0);

  BOOST_CHECK_EQUAL(plan_active_count(w), std::size_t{0});

  w.gop.serial_invoke(
      [saved_threshold] { TA::SparseShape<float>::threshold(saved_threshold); });
}

// gap-fill: SPARSE identity anchor (ce+ce). DensePolicy ce_ce identity
// is retile_identity_ce_ce_dist; this is its SparsePolicy sibling -- an inactive
// .retile() (empty + own-U targets) on a sparse ce_ce contraction with screened
// outer tiles must be byte-for-byte STOCK (reldiff == 0.0) and run no active
// branch (plan_active_count == 0). C(i1,i2,j1;a) = A(i1,i2,k;a,c) * B(j1,k;c).
// Built collectively => rank-stable, np=1 AND np=2.
BOOST_AUTO_TEST_CASE(retile_identity_sparse_ce_ce_dist) {
  auto& w = TA::get_default_world();
  const float saved_threshold = TA::SparseShape<float>::threshold();
  w.gop.serial_invoke([] {
    TA::SparseShape<float>::threshold(std::numeric_limits<float>::min());
  });

  auto t = tr1d(4, 2);   // 2 external tiles per axis
  auto kk = tr1d(4, 2);  // 2 K-tiles
  // Left A zeros the k==1 U K-tile; right B fully dense.
  ArrayToTSparse A0 = make_arena_sparse(
      w, TA::TiledRange{t, t, kk}, TA::Range{3, 4},
      [](const auto& tidx) { return tidx[2] == 1; });
  ArrayToTSparse B0 = make_arena_sparse(w, TA::TiledRange{t, kk},
                                        TA::Range{4},
                                        [](const auto&) { return false; });
  w.gop.fence();

  reset_plan_active_count();

  ArrayToTSparse C_ref =
      TA::einsum(A0("i1,i2,k;a,c"), B0("j1,k;c"), "i1,i2,j1;a");
  w.gop.fence();

  ArrayToTSparse C_empty;
  C_empty("i1,i2,j1;a") =
      (A0("i1,i2,k;a,c") * B0("j1,k;c")).retile({}, {}, {}, {});
  w.gop.fence();
  BOOST_CHECK_EQUAL(array_max_reldiff_sparse(C_empty, C_ref), 0.0);

  ArrayToTSparse C_self;
  C_self("i1,i2,j1;a") =
      (A0("i1,i2,k;a,c") * B0("j1,k;c"))
          .retile(/*H=*/{}, /*M=*/{t, t}, /*N=*/{t}, /*K=*/{kk});
  w.gop.fence();
  BOOST_CHECK_EQUAL(array_max_reldiff_sparse(C_self, C_ref), 0.0);

  BOOST_CHECK_EQUAL(plan_active_count(w), std::size_t{0});

  w.gop.serial_invoke(
      [saved_threshold] { TA::SparseShape<float>::threshold(saved_threshold); });
}

// gap-fill: SPARSE ce+e coarsen-K (sparse ce+e had zero coverage; the
// sparse cases so far are all ce_ce). C(i1,i2,j1,j2;a,b) = A(i1,i2,k;a)*B(.;b)
// with the contracted k coarsened 4 U -> 1 T. The LEFT operand zeros the
// boundary U K-tile (k==3) so the coarse K pack carries a HOLE (the A1
// full-footprint single-page pack on the ce_e strided arm); the right operand
// stays dense. The single coarse K-step is still live on both operands (left
// k=0..2 present). Oracle = no-retile einsum on the SAME sparse inputs; match
// < 1e-9, plan active, no hang. Shape built collectively => rank-stable, np=1
// AND np=2.
BOOST_AUTO_TEST_CASE(retile_active_sparse_coarsen_k_ce_e_dist) {
  auto& w = TA::get_default_world();
  const float saved_threshold = TA::SparseShape<float>::threshold();
  w.gop.serial_invoke([] {
    TA::SparseShape<float>::threshold(std::numeric_limits<float>::min());
  });

  auto t = tr1d(4, 2);    // 2 external tiles per axis
  auto kU = tr1d(8, 2);   // 4 fine U K-tiles (idx 0..3)
  auto kT = tr1d(8, 8);   // coarsen K to ONE T tile (covers all 4 U K-tiles)

  // Left A(i1,i2,k): zero the boundary U K-tile k==3 everywhere (coarse K cell
  // still packs k=0,1,2 -> step live). Right B(j1,j2,k): fully dense.
  ArrayToTSparse A0 = make_arena_sparse(
      w, TA::TiledRange{t, t, kU}, TA::Range{3},
      [](const auto& tidx) { return tidx[2] == 3; });
  ArrayToTSparse B0 = make_arena_sparse(w, TA::TiledRange{t, t, kU},
                                        TA::Range{5},
                                        [](const auto&) { return false; });
  w.gop.fence();

  reset_plan_active_count();

  ArrayToTSparse C_ref =
      TA::einsum(A0("i1,i2,k;a"), B0("j1,j2,k;b"), "i1,i2,j1,j2;a,b");
  w.gop.fence();

  ArrayToTSparse C;
  C("i1,i2,j1,j2;a,b") = (A0("i1,i2,k;a") * B0("j1,j2,k;b"))
                             .retile(/*H=*/{}, /*M=*/{t, t}, /*N=*/{t, t},
                                     /*K=*/{kT});
  w.gop.fence();
  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff_sparse(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});

  w.gop.serial_invoke(
      [saved_threshold] { TA::SparseShape<float>::threshold(saved_threshold); });
}

// (crash #1): SparsePolicy arena-ToT ce_ce coarsen where a coarse cell
// packs SOME absent U sub-tiles (holes) but NO whole coarse cell/step is zero --
// isolating the full-footprint single-page pack from the A2 SUMMA-level liveness
// bug. Coarsen the contracted SUMMA-K axis k (4 fine U K-tiles -> 1 coarse T
// K-tile) and zero the BOUNDARY (last) U K-tile (k==3) on the LEFT operand only,
// leaving the right operand fully dense. Pre-fix, `PackBlockTask::run` sizes the
// packed left coarse tile's K box from the min/max of the PRESENT left U tiles,
// so it shrinks to k in [0,6); the right coarse tile keeps the full k in [0,8).
// The contraction kernel then sees a left K-extent that disagrees with the right
// K-extent (Ko from each operand differs) -> `arena_strided_dgemm_ce_ce_right`'s
// `shape_ok` TA_ASSERT fires (crash #1). Post-fix the left coarse tile is packed
// at the full k in [0,8) with a HOLE at k==3 (a null cell the kernel skips), so
// both operands agree and the result equals the oracle.
//
// The single coarse K-step still has present tiles on BOTH operands (left k=0..2
// present, right dense), so no whole coarse cell/step is zero and the result is
// fully dense -- no A2 (SUMMA-level liveness) involvement. Oracle = no-retile
// einsum on the SAME sparse inputs. The shape is built collectively (identical
// on every rank) so the pattern is rank-stable: runs at np=1 AND np=2.
BOOST_AUTO_TEST_CASE(retile_active_sparse_coarsen_k_ce_ce_dist) {
  auto& w = TA::get_default_world();
  // Threshold so any positive tile norm clears the bar (binary sparsity).
  // Save and restore the global default to avoid leaking into later tests.
  const float saved_threshold = TA::SparseShape<float>::threshold();
  w.gop.serial_invoke([] {
    TA::SparseShape<float>::threshold(std::numeric_limits<float>::min());
  });

  auto i1 = tr1d(4, 2);   // 2 left-external M-tiles
  auto i2 = tr1d(4, 2);   // 2 left-external M-tiles
  auto j1 = tr1d(4, 2);   // 2 right-external N-tiles
  auto kU = tr1d(8, 2);   // 4 fine U K-tiles (idx 0..3) on the contracted axis
  auto kT = tr1d(8, 8);   // coarsen K to ONE T tile (covers all 4 U K-tiles)

  // Left A(i1,i2,k): zero the boundary (last) U K-tile k==3 everywhere. The
  // coarse K cell still packs k=0,1,2 (non-zero, step live); the missing
  // boundary K tile is exactly what shrank the pre-fix box.
  ArrayToTSparse A0 = make_arena_sparse(
      w, TA::TiledRange{i1, i2, kU}, TA::Range{3, 4},
      [](const auto& t) { return t[2] == 3; });
  // Right B(j1,k): fully dense (full K extent, no holes).
  ArrayToTSparse B0 = make_arena_sparse(w, TA::TiledRange{j1, kU},
                                        TA::Range{4},
                                        [](const auto&) { return false; });
  w.gop.fence();

  reset_plan_active_count();

  ArrayToTSparse C_ref =
      TA::einsum(A0("i1,i2,k;a,c"), B0("j1,k;c"), "i1,i2,j1;a");
  w.gop.fence();

  ArrayToTSparse C;
  C("i1,i2,j1;a") = (A0("i1,i2,k;a,c") * B0("j1,k;c"))
                        .retile(/*H=*/{}, /*M=*/{i1, i2}, /*N=*/{j1},
                                /*K=*/{kT});
  w.gop.fence();
  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff_sparse(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});

  w.gop.serial_invoke(
      [saved_threshold] { TA::SparseShape<float>::threshold(saved_threshold); });
}

// (crash #2, SUMMA-level liveness): SparsePolicy arena-ToT ce_ce that
// coarsens the M ride axis i1 AND the multi-step contracted axis k, with a
// WHOLE coarse K-step genuinely zero on BOTH operands. C(i1,i2,j1;a) =
// A(i1,i2,k;a,c) * B(j1,k;c). i1 coarsens 4 U -> 2 T; k coarsens 4 U -> 2 coarse
// K cells {k0,k1},{k2,k3}. Zeroing every U tile with k in {2,3} on BOTH operands
// makes the SECOND coarse K step empty everywhere -- iterate_col/iterate_row must
// skip it and the coarse masks must prune its processes. Only coarse K step 0
// contributes, on the single coarse basis. Pre-A2 the fine-U-result liveness
// disagreed with the coarse operand steps and stranded contributions on null
// reduce tasks (pimpl_). Oracle = no-retile einsum on the same
// sparse inputs; match < 1e-9, plan active, no hang, at np=1 AND np=2.
BOOST_AUTO_TEST_CASE(retile_active_sparse_coarsen_mk_zerostep_ce_ce_dist) {
  auto& w = TA::get_default_world();
  const float saved_threshold = TA::SparseShape<float>::threshold();
  w.gop.serial_invoke([] {
    TA::SparseShape<float>::threshold(std::numeric_limits<float>::min());
  });

  auto i1U = tr1d(8, 2);  // 4 U M-tiles on the ride axis
  auto i2 = tr1d(4, 2);   // 2 left-external M-tiles (identity)
  auto j1 = tr1d(4, 2);   // 2 N-tiles
  auto kU = tr1d(8, 2);   // 4 fine U K-tiles (idx 0..3)
  auto i1T = tr1d(8, 4);  // coarsen i1: 4 U -> 2 T (each covers 2 U)
  auto kT = tr1d(8, 4);   // coarsen k:  4 U -> 2 coarse K cells {k0,k1},{k2,k3}

  // BOTH operands zero the entire second coarse K block (k in {2,3}); the second
  // SUMMA step is then empty on every rank.
  ArrayToTSparse A0 = make_arena_sparse(
      w, TA::TiledRange{i1U, i2, kU}, TA::Range{3, 4},
      [](const auto& t) { return t[2] >= 2; });
  ArrayToTSparse B0 = make_arena_sparse(
      w, TA::TiledRange{j1, kU}, TA::Range{4},
      [](const auto& t) { return t[1] >= 2; });
  w.gop.fence();

  reset_plan_active_count();

  ArrayToTSparse C_ref = TA::einsum(A0("i1,i2,k;a,c"), B0("j1,k;c"), "i1,i2,j1;a");
  w.gop.fence();

  ArrayToTSparse C;
  C("i1,i2,j1;a") = (A0("i1,i2,k;a,c") * B0("j1,k;c"))
                        .retile(/*H=*/{}, /*M=*/{i1T, i2}, /*N=*/{j1},
                                /*K=*/{kT});
  w.gop.fence();
  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff_sparse(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});

  w.gop.serial_invoke(
      [saved_threshold] { TA::SparseShape<float>::threshold(saved_threshold); });
}

// (crash #2, liveness/SET-count decoupling): a scattered-sparse ce_ce
// that produces a CROSS-K SPURIOUS coarse result cell -- coarse-present (the
// coarse gemm product is non-zero) yet fine-absent (every covered fine U result
// tile is zero). C(i1,i2,j1;a) = A(i1,i2,k;a,c) * B(j1,k;c) over ONE coarse K
// cell {k0,k1} (kU = 2 U -> kT = 1 T). Pattern:
//   - LEFT A present only at k==0 (zero k==1 everywhere).
//   - RIGHT B for j1==0 present at k0 AND k1 (genuine); for j1==1 present ONLY at
//     k1 (zero k0).
// Then for the coarse result cell (M, j1==1): coarse_left present (via k0),
// coarse_right present (via k1) => coarse_result_cell_nonzero == true (LIVE).
// But the FINE result C[*, j1==1] = A[*,0]*B[1,0] + A[*,1]*B[1,1] =
// present*absent + absent*present = 0 => the cell covers ZERO fine-nonzero U
// result tiles. The cell must be LIVE (so its (holes) contributions land
// safely), submitted+consumed, and SET zero U tiles -- count-neutral. The j1==0
// column is genuine and non-zero. M is coarsened so the active path fires.
// Oracle = no-retile einsum; match < 1e-9, plan active, no hang, np=1 AND np=2.
BOOST_AUTO_TEST_CASE(retile_active_sparse_coarsen_m_crossk_ce_ce_dist) {
  auto& w = TA::get_default_world();
  const float saved_threshold = TA::SparseShape<float>::threshold();
  w.gop.serial_invoke([] {
    TA::SparseShape<float>::threshold(std::numeric_limits<float>::min());
  });

  auto i1U = tr1d(4, 2);  // 2 U M-tiles on the ride axis
  auto i2 = tr1d(2, 1);   // 2 left-external M-tiles (identity)
  auto j1 = tr1d(4, 2);   // 2 N-tiles (j1 == 0 genuine, j1 == 1 cross-k spurious)
  auto kU = tr1d(4, 2);   // 2 fine U K-tiles (k0, k1)
  auto i1T = tr1d(4, 4);  // coarsen i1: 2 U -> 1 T (covers both)
  auto kT = tr1d(4, 4);   // coarsen k:  2 U -> 1 coarse K cell {k0, k1}

  // LEFT A(i1,i2,k): present only at k==0.
  ArrayToTSparse A0 = make_arena_sparse(
      w, TA::TiledRange{i1U, i2, kU}, TA::Range{3, 4},
      [](const auto& t) { return t[2] == 1; });
  // RIGHT B(j1,k): j1==0 present at k0 & k1; j1==1 present only at k1 (zero k0).
  ArrayToTSparse B0 = make_arena_sparse(
      w, TA::TiledRange{j1, kU}, TA::Range{4},
      [](const auto& t) { return t[0] == 1 && t[1] == 0; });
  w.gop.fence();

  reset_plan_active_count();

  ArrayToTSparse C_ref = TA::einsum(A0("i1,i2,k;a,c"), B0("j1,k;c"), "i1,i2,j1;a");
  w.gop.fence();

  ArrayToTSparse C;
  C("i1,i2,j1;a") = (A0("i1,i2,k;a,c") * B0("j1,k;c"))
                        .retile(/*H=*/{}, /*M=*/{i1T, i2}, /*N=*/{j1},
                                /*K=*/{kT});
  w.gop.fence();
  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff_sparse(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});

  w.gop.serial_invoke(
      [saved_threshold] { TA::SparseShape<float>::threshold(saved_threshold); });
}

// (crash #2, DEAD-placeholder path): the one A2 branch with no prior
// coverage -- a coarse result cell where coarse_result_cell_nonzero == FALSE.
// C(i1,i2,j1;a) = A(i1,i2,k;a,c) * B(j1,k;c). M ride axis i1 coarsened (active
// path fires); N axis j1 has 2 tiles. The RIGHT operand B(j1,k) is zeroed for
// the ENTIRE j1==1 column across ALL k, so coarse_right_cell_nonzero(k, j1==1)
// is false for every coarse k => coarse_result_cell_nonzero(*, j1==1) == false
// for every coarse M-row. Those (M, j1==1) coarse result cells are STRUCTURALLY
// ABSENT (dead): initialize_active allocates them as NULL placeholder
// ReducePairTasks (the `else { new placeholder }` branch), the sparse `contract`
// add-guard `if (!reduce_tasks_[idx]) continue;` skips arming them, and
// finalize_active CONSUMES them WITHOUT submit (the dead-cell `else` branch).
// The j1==0 column stays fully present (left dense, right present) so the
// contraction is non-trivial. Oracle = no-retile einsum on the SAME sparse
// inputs (its j1==1 result tiles are likewise absent); match < 1e-9, plan
// active, no hang, at np=1 AND np=2.
BOOST_AUTO_TEST_CASE(retile_active_sparse_coarsen_m_deadcol_ce_ce_dist) {
  auto& w = TA::get_default_world();
  const float saved_threshold = TA::SparseShape<float>::threshold();
  w.gop.serial_invoke([] {
    TA::SparseShape<float>::threshold(std::numeric_limits<float>::min());
  });

  auto i1U = tr1d(8, 2);  // 4 U M-tiles on the ride axis
  auto i2 = tr1d(4, 2);   // 2 left-external M-tiles (identity)
  auto j1 = tr1d(4, 2);   // 2 N-tiles (j1 == 0 present, j1 == 1 dead column)
  auto kU = tr1d(8, 2);   // 4 fine U K-tiles (idx 0..3)
  auto i1T = tr1d(8, 4);  // coarsen i1: 4 U -> 2 T (each covers 2 U)
  auto kT = tr1d(8, 4);   // coarsen k:  4 U -> 2 coarse K cells {k0,k1},{k2,k3}

  // LEFT A(i1,i2,k): fully dense (every coarse left cell present).
  ArrayToTSparse A0 = make_arena_sparse(
      w, TA::TiledRange{i1U, i2, kU}, TA::Range{3, 4},
      [](const auto&) { return false; });
  // RIGHT B(j1,k): zero the ENTIRE j1==1 column across ALL k; j1==0 dense. This
  // makes coarse_right_cell_nonzero(k, j1==1) false for every coarse k.
  ArrayToTSparse B0 = make_arena_sparse(
      w, TA::TiledRange{j1, kU}, TA::Range{4},
      [](const auto& t) { return t[0] == 1; });
  w.gop.fence();

  reset_plan_active_count();

  ArrayToTSparse C_ref = TA::einsum(A0("i1,i2,k;a,c"), B0("j1,k;c"), "i1,i2,j1;a");
  w.gop.fence();

  ArrayToTSparse C;
  C("i1,i2,j1;a") = (A0("i1,i2,k;a,c") * B0("j1,k;c"))
                        .retile(/*H=*/{}, /*M=*/{i1T, i2}, /*N=*/{j1},
                                /*K=*/{kT});
  w.gop.fence();
  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff_sparse(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});

  // The whole j1==1 result column is structurally absent (a missing dead cell
  // or a stray submitted page there would blow up the reldiff above; this is
  // the explicit witness that the dead column carries no result tiles).
  const auto& tiles = C.trange().tiles_range();
  for (const auto& tidx : tiles)
    if (tidx[2] == 1) BOOST_CHECK(C.is_zero(tidx));

  w.gop.serial_invoke(
      [saved_threshold] { TA::SparseShape<float>::threshold(saved_threshold); });
}

// ===========================================================================
// priority gap (Task-A2 carry-forward): the MIXED / scale kernel on the
// ACTIVE retile path -- BLOCKED on a production gap, see agent/plans/
// task-6.1-report.md for the full root cause and the recommended fix.
//
// The mixed contraction multiplies a PLAIN Tensor<double> operand g by an
// arena-ToT operand I to an arena-ToT result C:
//
//   C(i_3,i_1,i_2,k_; a) = g(i_3,k_,m_) * I(m_,i_1,i_2; a)
//
// m_ is the contracted SUMMA-K axis; g's externals i_3,k_ are SUMMA-M; I's
// externals i_1,i_2 are SUMMA-N; the inner a rides from I to the result -> the
// per-cell kernel is the t_x_tot scale fast-path (g_scale_strided_calls[1]).
//
// CONFIRMED PRODUCTION BUG (lldb at TiledArray::exception_break, np=1): an
// ACTIVE coarsen .retile() on this contraction aborts with
// TA_ASSERT failed: pimpl_
// from Summa::contract (the DenseShape overload).
// Root cause: the SUMMA masks / initialize_active / finalize_active gate on the
// RESULT value_type (arena-ToT) and allocate reduce tasks on the COARSE grid,
// AND get_row gates on the RIGHT operand eval type (arena-ToT) and packs coarse
// -- but get_col gates on the LEFT operand eval type,
// which for the plain g is NOT arena-ToT, so the active branch is compiled out
// and get_col emits the STOCK fine-U column. contract then indexes the coarse
// reduce_tasks_ array with a fine-U col[i].first (col.size()==64 vs coarse
// M_grid==1) and hits an unallocated (null pimpl_) slot. The intended plain-
// operand coarse pack `dense_gather_block` was specified
// but NEVER implemented -- it exists only as a name in bench comments, so there
// is no code path that packs a plain dense operand coarse. Fixing it is a
// design-level change (new dense_gather_block + re-gate get_col / the operand
// masks / iterate_col on plan_.active for a plain operand + np>1 ownership for
// the plain block) spanning arena_retile.h + contraction_eval.h -- beyond a
// test-coverage task. Deferred; the mixed cases are NOT added here (a crashing
// or weakened test would be worse than an honest BLOCKED). The make_plain2 /
// make_plain_sparse helpers above are the scaffolding the fix will consume.
// ===========================================================================

// ===========================================================================
// arena Hadamard (fused-outer) ToT contraction harness + an
// inactive-plan stock anchor at nh_>1.
//
// This is the FIRST test exercising the general/Hadamard SUMMA path with
// n_slabs_ > 1 (a fused outer axis with more than one U tile). All prior
// active-retile cases had nh_==1 (a single fused slab). The contraction
//   C("h,i,j;a") = A("h,i,k;a,c") * B("h,j,k;c")
// has a leading FUSED outer mode h (the b-style index from
// general_product.cpp::expression_general_product_tot_inner_hadamard), outer
// contracted k, outer-external i (left ride) / j (right), and an INNER
// contraction over c (a rides to the result) -- so the per-cell kernel is the
// ce_ce strided DGEMM (g_strided_dgemm_ce_ce_{left,right}). The fused h axis
// has 2 U tiles, so n_slabs_ == 2 and the SUMMA general path runs 2 slabs.
//
// What this anchor proves TODAY (stock path, nh_>1):
//   * the no-retile general product C = A*B is correct vs the einsum oracle;
//   * a `.retile()` with EMPTY targets is bit-for-bit identical to no-retile
//     (an inactive plan must not perturb the stock SUMMA path);
//   * the ce_ce strided DGEMM fires on the fused (nh_>1) contraction;
//   * the plan-active counter reads 0 (no active retile code path ran).
//
// What this anchor deliberately does NOT do: drive an ACTIVE retile plan at
// nh_>1 (e.g. coarsen K or H with explicit per-role targets). That path is
// BROKEN at plan construction today and is the subject of /c/d.
// See the investigation note immediately below for the exact failure.
//
// ---------------------------------------------------------------------------
// INVESTIGATION (, report-only -- characterization of the active-plan gap;
// NOT a committed failing test):
//
// Giving this SAME Hadamard contraction any NON-EMPTY .retile() target vector
// (identity-via-U, coarsen-K, OR coarsen-H -- all behave identically) throws at
// plan construction, BEFORE any SUMMA work:
//
//
//   TA_ASSERT failed: targets.size() == U_axes.size()
//
// Root cause: make_retile_plan partitions the FULL
// operand trange (left_.trange(), 3 outer dims here: h,i,k) using the bounds of
// op_.gemm_helper() -- but that helper is the FOLDED, fused-mode-STRIPPED outer
// helper - nh). For
// nf=1 the folded helper has left_rank=2, left_outer=[0,1), left_inner=[1,2).
// The partition loop reads left_U.dim(0)=h into hU
// (because 0 < left_outer_begin()+nf == 1) and then STOPS at left_outer_end()==1
// -- so the real M external left_U.dim(1)=i is never visited and mU is empty.
// Likewise nU is empty. retile_role_axes then asserts targetM.size()(==1) ==
// mU.size()(==0) and throws. With nf==0 (all prior active cases) the folded and
// full helpers coincide and the partition is correct; the mismatch surfaces
// only for nf>0. Fixing this dimension-offset (so make_retile_plan accounts for
// the nh leading fused modes that the trange carries but the folded helper does
// not) is exactly 's first job.
// ===========================================================================
BOOST_AUTO_TEST_CASE(retile_identity_hadamard_ce_ce_nh_gt_1_dist) {
  auto& w = TA::get_default_world();
  auto h = tr1d(4, 2);   // 2 fused H tiles  => n_slabs_ == 2 (nh_>1)
  auto i = tr1d(4, 2);   // 2 left-external M-tiles
  auto j = tr1d(4, 2);   // 2 right-external N-tiles
  auto kk = tr1d(8, 4);  // 2 contracted K-tiles
  // A(h,i,k; a,c): inner a=3 spectator (rides), c=4 contracted.
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{h, i, kk}, TA::Range{3, 4});
  // B(h,j,k; c): inner c=4 contracted.
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{h, j, kk}, TA::Range{4});
  w.gop.fence();

  reset_plan_active_count();
#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
#endif

  // No-retile reference: same inputs, Hadamard general product over fused h.
  ArrayToT2 C_ref = TA::einsum(A0("h,i,k;a,c"), B0("h,j,k;c"), "h,i,j;a");
  w.gop.fence();

  // .retile() with EMPTY targets -> inactive plan -> bit-for-bit identical to
  // the no-retile stock SUMMA path, at np=1 AND np=2, with n_slabs_ == 2.
  ArrayToT2 C;
  C("h,i,j;a") =
      (A0("h,i,k;a,c") * B0("h,j,k;c")).retile(/*H=*/{}, /*M=*/{}, /*N=*/{},
                                               /*K=*/{});
  w.gop.fence();

  // Result trange equals the no-retile (U) trange; values match the oracle.
  // (Tolerance, not bit-for-bit: at np=2 the 2-slab x 2-K-tile SUMMA reduction
  // may accumulate the contracted axis in a different order than the no-retile
  // path, differing at the last ULP -- numerically identical, not byte-equal.)
  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  // Inactive plan: the plan-active counter must read 0 (no active path ran).
  // This is the load-bearing inactive-plan invariant -- the EMPTY .retile()
  // must not engage any active retile code path on a nh_>1 contraction.
  BOOST_CHECK_EQUAL(plan_active_count(w), std::size_t{0});

#ifdef TA_STRIDED_DGEMM_COUNT
  // The ce_ce strided DGEMM fired on the fused (nh_>1) contraction.
  std::size_t fires = TA::detail::g_strided_dgemm_ce_ce_right_calls.load() +
                      TA::detail::g_strided_dgemm_ce_ce_left_calls.load();
  w.gop.sum(&fires, 1);
  BOOST_CHECK_GT(fires, std::size_t{0});
#endif
}

// ===========================================================================
// FIRST ACTIVE retile plan on a fused (nh_>1) Hadamard contraction.
//
// Same contraction as the anchor -- C("h,i,j;a") = A("h,i,k;a,c") *
// B("h,j,k;c") -- but now driven by an ACTIVE .retile():
//   * H IDENTITY  (targetH == the U h tiling): the fused axis is NOT coarsened
// (coarse H is a separate case); n_slabs_ stays U-derived == the U H tile count.
//   * K COARSENED (targetK coarser than the U k tiling): the 2 U K-tiles
//     collapse into 1 coarse SUMMA K-tile, so the SUMMA runs 1 coarse K-step
//     per slab while the operands stay tiled at the fine (U) K count.
//   * M / N externals IDENTITY (targetM/N == the U i/j tilings).
//
// This is the green gate for 's two fixes:
//   (1) make_retile_plan now partitions the FULL [H,M,K]/[H,K,N] operand
//       tranges with an explicit nf offset (the folded helper gave empty M/N
//       and threw at plan construction for ANY non-empty target when nf>0);
//   (2) get_col_coarsen / get_row_coarsen now pin the operand gather to the
//       coarse slab step_h(s) (the leading H mode), so slab h==1 gathers slab
//       1's U tiles, not slab 0's.
//
// Oracle = no-retile einsum on the SAME inputs; result must match (<1e-9), the
// plan-active counter must be > 0 (an active path ran), and the ce_ce strided
// DGEMM must fire (> 0).
//
// NP SCOPING: init_distribution_general is now coarse-aware on
// the ungrouped (proc_h_ == 1) general path -- it composes the COARSE
// co-location of the ordinary 2-d path (make_operand/result_coarse_pmap) with
// the slab replication of the general path (SlabbedPmap over n_slabs_). With
// n_slabs_ == 2 fused slabs, M_grid == N_grid == 2 externals and P == 2 the
// proc_h_ heuristic picks proc_h_ == 1 (the 2-d cap absorbs the 2 ranks), so
// this case exercises the new ACTIVE ungrouped branch at np=2. The result is
// delivered at the FINE (U) tiling and validated against the no-retile einsum
// oracle. (The grouped proc_h_ > 1 active path is, a separate case.)
BOOST_AUTO_TEST_CASE(retile_active_hadamard_identityH_coarsenK_ce_ce_dist) {
  auto& w = TA::get_default_world();

  auto h = tr1d(4, 2);   // 2 fused H tiles  => n_slabs_ == 2 (nh_>1)
  auto i = tr1d(4, 2);   // 2 left-external M-tiles
  auto j = tr1d(4, 2);   // 2 right-external N-tiles
  auto kU = tr1d(8, 4);  // 2 contracted K U-tiles
  auto kT = tr1d(8, 8);  // coarsen K: 2 U K-tiles -> 1 coarse T K-tile
  // A(h,i,k; a,c): inner a=3 spectator (rides), c=4 contracted.
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{h, i, kU}, TA::Range{3, 4});
  // B(h,j,k; c): inner c=4 contracted.
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{h, j, kU}, TA::Range{4});
  w.gop.fence();

  reset_plan_active_count();
#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
#endif

  // No-retile reference: same inputs, Hadamard general product over fused h.
  ArrayToT2 C_ref = TA::einsum(A0("h,i,k;a,c"), B0("h,j,k;c"), "h,i,j;a");
  w.gop.fence();

  // ACTIVE retile: H identity (== U h), externals identity (== U i/j), K
  // coarsened (kT coarser than kU). Active because the K role is non-identity.
  ArrayToT2 C;
  C("h,i,j;a") = (A0("h,i,k;a,c") * B0("h,j,k;c"))
                     .retile(/*H=*/{h}, /*M=*/{i}, /*N=*/{j}, /*K=*/{kT});
  w.gop.fence();

  // Result trange equals the no-retile (U) trange; values match the oracle.
  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  // np=1 green gate: an active retile path ran on the nh_>1 contraction ...
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
#ifdef TA_STRIDED_DGEMM_COUNT
  // ... and the ce_ce strided DGEMM fired on the coarse-K cells.
  std::size_t fires = TA::detail::g_strided_dgemm_ce_ce_right_calls.load() +
                      TA::detail::g_strided_dgemm_ce_ce_left_calls.load();
  w.gop.sum(&fires, 1);
  BOOST_CHECK_GT(fires, std::size_t{0});
#endif
}

// ===========================================================================
// ACTIVE retile on a fused (nh_>1) Hadamard contraction that
// RIDES the externals down to a SINGLE coarse tile, exercising the h-GROUPED
// (proc_h_ > 1) 3-d grid branch at np=2.
//
// Same contraction as / -- C("h,i,j;a") = A("h,i,k;a,c") * B("h,j,k;c")
// -- driven by an ACTIVE .retile() that:
//   * H IDENTITY  (targetH == U h tiling): the fused axis is not coarsened;
// n_slabs_ stays U-derived == 2 (coarse H is a separate case).
//   * M COARSENED to ONE coarse tile (i: 2 U tiles -> iT one T tile) and
//   * N COARSENED to ONE coarse tile (j: 2 U tiles -> jT one T tile): the
//     per-slab external SUMMA grid collapses to 1x1 (coarse M_grid==N_grid==1).
//   * K COARSENED (2 U K-tiles -> 1 coarse SUMMA K-tile).
//
// With M_grid*N_grid == 1 the 2-d cap is 1, so at np>1 the proc_h_ heuristic
// spreads the world over the slab (h) axis (proc_h_ == min(n_slabs_, P) > 1),
// engaging the h-GROUPED 3-d grid branch. re-implemented that branch's
// ACTIVE coarse co-location (group-local coarse proc_grid_ + coarse-co-located
// operand/result pmaps wrapped in the h-grouped SlabbedPmap) and fixed the
// result-ordinal base: initialize_active / finalize_active now base the U
// result tile on the GLOBAL slab index h (slab_u_base = h * (u_vol / nh_)),
// matching the reduce-task / result-owner layout (slab_base = h *
// result_slab_size_). Previously they used the GROUP-LOCAL slab_ord(h), which
// aliased every group's tiles onto slab 0's ordinals -> the closing-fence
// deadlock that blocked.
//
// Both np=1 (ungrouped 1x1 active path) and np=2 (grouped, proc_h_ == 2) must
// match the no-retile einsum oracle. At np=2 we additionally assert the grouped
// path engaged via the g_summa_proc_h_grouped_calls witness.
BOOST_AUTO_TEST_CASE(retile_active_hadamard_ride_single_tile_grouped_ce_ce_dist) {
  auto& w = TA::get_default_world();

  auto h = tr1d(4, 2);    // 2 fused H tiles  => n_slabs_ == 2 (nh_>1)
  auto i = tr1d(4, 2);    // 2 left-external M U-tiles ...
  auto iT = tr1d(4, 4);   // ... coarsened to ONE coarse M tile
  auto j = tr1d(4, 2);    // 2 right-external N U-tiles ...
  auto jT = tr1d(4, 4);   // ... coarsened to ONE coarse N tile
  auto kU = tr1d(8, 4);   // 2 contracted K U-tiles
  auto kT = tr1d(8, 8);   // coarsen K: 2 U K-tiles -> 1 coarse T K-tile
  // A(h,i,k; a,c): inner a=3 spectator (rides), c=4 contracted.
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{h, i, kU}, TA::Range{3, 4});
  // B(h,j,k; c): inner c=4 contracted.
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{h, j, kU}, TA::Range{4});
  w.gop.fence();

  reset_plan_active_count();
#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
  TA::detail::g_summa_proc_h_grouped_calls.store(0);
#endif

  // No-retile reference: same inputs, Hadamard general product over fused h.
  ArrayToT2 C_ref = TA::einsum(A0("h,i,k;a,c"), B0("h,j,k;c"), "h,i,j;a");
  w.gop.fence();

  // ACTIVE retile, ride single-tile (M/N coarsened to one coarse tile), K
  // coarsened, H identity. At np=1 the 1x1 coarse grid is the ungrouped active
  // path; at np=2 the surplus rank rides the slab axis (proc_h_ == 2, the
  // h-grouped 3-d grid). Both must match the oracle, no hang.
  ArrayToT2 C;
  C("h,i,j;a") = (A0("h,i,k;a,c") * B0("h,j,k;c"))
                     .retile(/*H=*/{h}, /*M=*/{iT}, /*N=*/{jT}, /*K=*/{kT});
  w.gop.fence();

  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
#ifdef TA_STRIDED_DGEMM_COUNT
  std::size_t fires = TA::detail::g_strided_dgemm_ce_ce_right_calls.load() +
                      TA::detail::g_strided_dgemm_ce_ce_left_calls.load();
  w.gop.sum(&fires, 1);
  BOOST_CHECK_GT(fires, std::size_t{0});
  // At np>1 the ride-single-tile regime must engage the h-grouped (proc_h_ > 1)
  // active distribution; at np=1 it stays ungrouped (counter 0).
  std::size_t grouped = TA::detail::g_summa_proc_h_grouped_calls.load();
  w.gop.sum(&grouped, 1);
  if (w.size() > 1)
    BOOST_CHECK_GT(grouped, std::size_t{0});
  else
    BOOST_CHECK_EQUAL(grouped, std::size_t{0});
#endif
}

// Result-PERMUTATION on the ACTIVE retile path (the coupling gap). Same shapes
// as retile_active_hadamard_ride_single_tile_grouped_ce_ce_dist, but the result
// is requested in a PERMUTED outer layout (j,i,h) vs the canonical (h,i,j). The
// engine evaluates the contraction in canonical layout via an inner Summa, then
// re-permutes per tile (UnaryEvalImpl<GeneralRepermuteOp>). Under the active
// coarsen-M/N ride-single plan the inner Summa carries the FINE U canonical
// trange while its result pmap must map every FINE U tile to its covering COARSE
// cell's owner -- the coupling fixes. Oracle = no-retile einsum into the
// SAME permuted layout. Must match < 1e-9 and terminate at np=1 AND np=2.
BOOST_AUTO_TEST_CASE(retile_active_hadamard_permuted_result_ride_single_ce_ce_dist) {
  auto& w = TA::get_default_world();

  auto h = tr1d(4, 2);    // 2 fused H tiles
  auto i = tr1d(4, 2);    // 2 left-external M U-tiles ...
  auto iT = tr1d(4, 4);   // ... coarsened to ONE coarse M tile
  auto j = tr1d(4, 2);    // 2 right-external N U-tiles ...
  auto jT = tr1d(4, 4);   // ... coarsened to ONE coarse N tile
  auto kU = tr1d(8, 4);   // 2 contracted K U-tiles
  auto kT = tr1d(8, 8);   // coarsen K: 2 U K-tiles -> 1 coarse T K-tile
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{h, i, kU}, TA::Range{3, 4});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{h, j, kU}, TA::Range{4});
  w.gop.fence();

  reset_plan_active_count();

  // Oracle: no-retile einsum into the PERMUTED (j,i,h) outer layout.
  ArrayToT2 C_ref = TA::einsum(A0("h,i,k;a,c"), B0("h,j,k;c"), "j,i,h;a");
  w.gop.fence();

  ArrayToT2 C;
  C("j,i,h;a") = (A0("h,i,k;a,c") * B0("h,j,k;c"))
                     .retile(/*H=*/{h}, /*M=*/{iT}, /*N=*/{jT}, /*K=*/{kT});
  w.gop.fence();

  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
}

// Permuted result + coarsen M ride AND K together (ce+ce), Hadamard h fused.
// Exercises the coarse-M carve and coarse-K step geometry together under a
// result permutation. Oracle = no-retile einsum into the permuted layout.
BOOST_AUTO_TEST_CASE(retile_active_hadamard_permuted_result_coarsen_mk_ce_ce_dist) {
  auto& w = TA::get_default_world();
  auto h = tr1d(4, 2);    // 2 fused H tiles
  auto i = tr1d(8, 2);    // 4 left-external M U-tiles ...
  auto iT = tr1d(8, 8);   // ... coarsened to ONE coarse M tile
  auto j = tr1d(4, 2);    // 2 right-external N U-tiles (identity)
  auto kU = tr1d(8, 2);   // 4 contracted K U-tiles ...
  auto kT = tr1d(8, 8);   // ... coarsened to ONE coarse K tile
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{h, i, kU}, TA::Range{3, 4});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{h, j, kU}, TA::Range{4});
  w.gop.fence();

  reset_plan_active_count();

  ArrayToT2 C_ref = TA::einsum(A0("h,i,k;a,c"), B0("h,j,k;c"), "j,i,h;a");
  w.gop.fence();

  ArrayToT2 C;
  C("j,i,h;a") = (A0("h,i,k;a,c") * B0("h,j,k;c"))
                     .retile(/*H=*/{h}, /*M=*/{iT}, /*N=*/{j}, /*K=*/{kT});
  w.gop.fence();
  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
}

// Permuted result on the ce+e kernel (no inner contraction index shared beyond
// the contracted k; spectator inner a rides). Canonical result (h,i,j1,j2);
// request the permuted (j2,j1,i,h). Coarsen the left-external M ride + K.
BOOST_AUTO_TEST_CASE(retile_active_hadamard_permuted_result_ce_e_dist) {
  auto& w = TA::get_default_world();
  auto h = tr1d(4, 2);    // 2 fused H tiles
  auto i = tr1d(8, 2);    // 4 left-external M U-tiles ...
  auto iT = tr1d(8, 8);   // ... coarsened to ONE coarse M tile
  auto j = tr1d(4, 2);    // 2 right-external tiles per N axis (identity)
  auto kU = tr1d(8, 2);   // 4 contracted K U-tiles ...
  auto kT = tr1d(8, 8);   // ... coarsened to ONE coarse K tile
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{h, i, kU}, TA::Range{3});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{h, j, j, kU}, TA::Range{5});
  w.gop.fence();

  reset_plan_active_count();

  ArrayToT2 C_ref =
      TA::einsum(A0("h,i,k;a"), B0("h,j1,j2,k;b"), "j2,j1,i,h;a,b");
  w.gop.fence();

  ArrayToT2 C;
  C("j2,j1,i,h;a,b") = (A0("h,i,k;a") * B0("h,j1,j2,k;b"))
                           .retile(/*H=*/{h}, /*M=*/{iT}, /*N=*/{j, j},
                                   /*K=*/{kT});
  w.gop.fence();
  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
}

// REQUIRED: NON-INVOLUTIVE result permutation (3-cycle). Canonical (h,i,j)
// requested as (i,j,h): target[0]=i=canon[1], [1]=j=canon[2], [2]=h=canon[0].
// Guards the ENGINE's value-permutation direction (GeneralRepermuteOp / perm_):
// an inverted VALUE-permute corrupts element placement, which an involution
// (swap/reversal) and the bench `absum` would both miss but this 3-cycle
// catches. NOTE: it does NOT guard make_repermuted_result_pmap's .inv()
// (placement-only / perf -- see). Permutation-sensitive, np=1 AND np=2.
BOOST_AUTO_TEST_CASE(retile_active_hadamard_permuted_result_3cycle_ce_ce_dist) {
  auto& w = TA::get_default_world();
  auto h = tr1d(4, 2);
  auto i = tr1d(4, 2);
  auto iT = tr1d(4, 4);
  auto j = tr1d(4, 2);
  auto jT = tr1d(4, 4);
  auto kU = tr1d(8, 4);
  auto kT = tr1d(8, 8);
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{h, i, kU}, TA::Range{3, 4});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{h, j, kU}, TA::Range{4});
  w.gop.fence();
  reset_plan_active_count();
  ArrayToT2 C_ref = TA::einsum(A0("h,i,k;a,c"), B0("h,j,k;c"), "i,j,h;a");
  w.gop.fence();
  ArrayToT2 C;
  C("i,j,h;a") = (A0("h,i,k;a,c") * B0("h,j,k;c"))
                     .retile(/*H=*/{h}, /*M=*/{iT}, /*N=*/{jT}, /*K=*/{kT});
  w.gop.fence();
  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
}

// ===========================================================================
// H-AXIS COARSEN core (np=1). The Hadamard (fused, leading-outer)
// axis is itself COARSENED -- the hard coupled core. Mechanically this is the
// Task-3 M-axis coarsen applied to the LEADING fused axis: a coarse slab covers
// a contiguous range of U fused tiles, the operand pack gathers them along the
// leading fused axis into ONE single-page tile whose fused outer extent = the
// union of the covered U fused extents (so BatchedContractReduce, which batches
// over fused_volume(tile.range()) == the product of the leading nfused-mode
// EXTENTS, automatically widens each batched-GEMM call), and the result carve
// splits the coarse slab's result tile back into its U fused tiles.
//
// -- coarsen-H ONLY (M/N/K identity): isolates the H mechanics.
// C("h,i,j;a") = A("h,i,k;a,c") * B("h,j,k;c"). 4 U fused tiles -> 1 coarse H
// slab. n_slabs_ goes from 4 (U) to 1 (coarse), so the whole H axis batches in
// ONE slab. Oracle = no-retile einsum on the SAME inputs (< 1e-9), plan active,
// ce_ce strided DGEMM fired. SCOPE: runs at BOTH np=1 and np=2. With
// n_slabs_ == 1 coarse slab and M_grid == N_grid == 2 externals, the proc_h_
// heuristic picks proc_h_ == 1 at np<=2 (the 2-d cap M_grid*N_grid==4 absorbs
// both ranks), so this is the UNGROUPED (proc_h_ == 1) coarse-H active path --
// the np=2 fallback. The grouped (proc_h_ > 1) coarse-H path is the
// ride-single-tile case below. The operand/result SlabbedPmaps
// replicate over the U slab count (n_slabs_u_ == 4) while SUMMA rides the
// single coarse slab.
BOOST_AUTO_TEST_CASE(retile_active_hadamard_coarsenH_only_ce_ce_dist) {
  auto& w = TA::get_default_world();

  auto hU = tr1d(8, 2);   // 4 fused H U-tiles  => n_slabs_u_ == 4 (nh_>1)
  auto hT = tr1d(8, 8);   // coarsen H: 4 U fused tiles -> 1 coarse H slab
  auto i = tr1d(4, 2);    // 2 left-external M-tiles
  auto j = tr1d(4, 2);    // 2 right-external N-tiles
  auto kk = tr1d(8, 4);   // 2 contracted K-tiles
  // A(h,i,k; a,c): inner a=3 spectator (rides), c=4 contracted.
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{hU, i, kk}, TA::Range{3, 4});
  // B(h,j,k; c): inner c=4 contracted.
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{hU, j, kk}, TA::Range{4});
  w.gop.fence();

  reset_plan_active_count();
#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
  TA::detail::g_summa_proc_h_grouped_calls.store(0);
#endif

  ArrayToT2 C_ref = TA::einsum(A0("h,i,k;a,c"), B0("h,j,k;c"), "h,i,j;a");
  w.gop.fence();

  // ACTIVE retile: H coarsened (4 U -> 1 T), externals/K identity.
  ArrayToT2 C;
  C("h,i,j;a") = (A0("h,i,k;a,c") * B0("h,j,k;c"))
                     .retile(/*H=*/{hT}, /*M=*/{i}, /*N=*/{j}, /*K=*/{kk});
  w.gop.fence();

  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
#ifdef TA_STRIDED_DGEMM_COUNT
  std::size_t fires = TA::detail::g_strided_dgemm_ce_ce_right_calls.load() +
                      TA::detail::g_strided_dgemm_ce_ce_left_calls.load();
  w.gop.sum(&fires, 1);
  BOOST_CHECK_GT(fires, std::size_t{0});
  // Single coarse slab: the proc_h_ heuristic stays ungrouped at np<=2.
  std::size_t grouped = TA::detail::g_summa_proc_h_grouped_calls.load();
  w.gop.sum(&grouped, 1);
  BOOST_CHECK_EQUAL(grouped, std::size_t{0});
#endif
}

// (cont.) -- coarsen-H + coarse-K (the coupled core). Same contraction;
// H coarsened (4 U fused -> 2 coarse slabs, each covering 2 U fused tiles) AND
// K coarsened (2 U K-tiles -> 1 coarse K-tile). Exercises the partial H coarsen
// (n_slabs_ == 2, not 1) together with the coarse-K pack, so a coarse slab maps
// to >1 U fused tile and the carve must split the coarse slab tile back into
// both of its U fused result tiles. Oracle < 1e-9, plan active, ce_ce fired.
// SCOPE: runs at BOTH np=1 and np=2. With 2 coarse slabs and M_grid ==
// N_grid == 2, the proc_h_ heuristic stays UNGROUPED at np<=2 (2-d cap == 4
// absorbs both ranks), so this remains the ungrouped (proc_h_ == 1) active
// path even though u_per_coarse_slab == 2 (the carve still splits each coarse
// slab into its 2 U fused result tiles). fallback.
BOOST_AUTO_TEST_CASE(retile_active_hadamard_coarsenHK_ce_ce_dist) {
  auto& w = TA::get_default_world();

  auto hU = tr1d(8, 2);   // 4 fused H U-tiles
  auto hT = tr1d(8, 4);   // coarsen H: 4 U -> 2 coarse slabs (each covers 2 U)
  auto i = tr1d(4, 2);    // 2 left-external M-tiles
  auto j = tr1d(4, 2);    // 2 right-external N-tiles
  auto kU = tr1d(8, 4);   // 2 contracted K U-tiles
  auto kT = tr1d(8, 8);   // coarsen K: 2 U K-tiles -> 1 coarse T K-tile
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{hU, i, kU}, TA::Range{3, 4});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{hU, j, kU}, TA::Range{4});
  w.gop.fence();

  reset_plan_active_count();
#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
#endif

  ArrayToT2 C_ref = TA::einsum(A0("h,i,k;a,c"), B0("h,j,k;c"), "h,i,j;a");
  w.gop.fence();

  ArrayToT2 C;
  C("h,i,j;a") = (A0("h,i,k;a,c") * B0("h,j,k;c"))
                     .retile(/*H=*/{hT}, /*M=*/{i}, /*N=*/{j}, /*K=*/{kT});
  w.gop.fence();

  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
#ifdef TA_STRIDED_DGEMM_COUNT
  std::size_t fires = TA::detail::g_strided_dgemm_ce_ce_right_calls.load() +
                      TA::detail::g_strided_dgemm_ce_ce_left_calls.load();
  w.gop.sum(&fires, 1);
  BOOST_CHECK_GT(fires, std::size_t{0});
#endif
}

// ===========================================================================
// -- coarsen-H GROUPED (proc_h_ > 1) ride-single-tile at np=2.
// The hard np=2 coarse-H path: H coarsened AND the externals (M/N) collapsed to
// ONE coarse tile each (ride single-tile), so M_grid * N_grid == 1, the 2-d cap
// is 1, and at np>1 the surplus ranks ride the slab axis -- proc_h_ ==
// min(n_slabs_coarse, P) > 1. With 4 U fused tiles coarsened to 2 coarse slabs
// (u_per_coarse_slab == 2) and P == 2, proc_h_ == 2: each rank's group owns one
// coarse slab and ALL the U tiles (operand + result) that coarse slab covers,
// via the h-grouped SlabbedPmap keyed by the COVERING coarse slab
// (coarse_slab_of). The carve's set_tile must land on the rank that ran that
// coarse slab's SUMMA (result_tile_owner delegates to the grouped result pmap).
// Both np=1 (ungrouped 1x1) and np=2 (grouped, proc_h_ == 2) must match the
// oracle with no hang; at np>1 the grouped coarse-H path is witnessed via
// g_summa_proc_h_grouped_calls.
BOOST_AUTO_TEST_CASE(
    retile_active_hadamard_coarsenH_ride_single_tile_grouped_ce_ce_dist) {
  auto& w = TA::get_default_world();

  auto hU = tr1d(8, 2);   // 4 fused H U-tiles  => n_slabs_u_ == 4
  auto hT = tr1d(8, 4);   // coarsen H: 4 U -> 2 coarse slabs (each covers 2 U)
  auto i = tr1d(4, 2);    // 2 left-external M U-tiles ...
  auto iT = tr1d(4, 4);   // ... coarsened to ONE coarse M tile (ride)
  auto j = tr1d(4, 2);    // 2 right-external N U-tiles ...
  auto jT = tr1d(4, 4);   // ... coarsened to ONE coarse N tile (ride)
  auto kU = tr1d(8, 4);   // 2 contracted K U-tiles
  auto kT = tr1d(8, 8);   // coarsen K: 2 U K-tiles -> 1 coarse T K-tile
  // A(h,i,k; a,c): inner a=3 spectator (rides), c=4 contracted.
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{hU, i, kU}, TA::Range{3, 4});
  // B(h,j,k; c): inner c=4 contracted.
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{hU, j, kU}, TA::Range{4});
  w.gop.fence();

  reset_plan_active_count();
#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
  TA::detail::g_summa_proc_h_grouped_calls.store(0);
#endif

  ArrayToT2 C_ref = TA::einsum(A0("h,i,k;a,c"), B0("h,j,k;c"), "h,i,j;a");
  w.gop.fence();

  // ACTIVE retile: H coarsened (4 U -> 2 coarse), M/N ride single-tile, K
  // coarsened. At np=2 this is the grouped (proc_h_ == 2) coarse-H path.
  ArrayToT2 C;
  C("h,i,j;a") = (A0("h,i,k;a,c") * B0("h,j,k;c"))
                     .retile(/*H=*/{hT}, /*M=*/{iT}, /*N=*/{jT}, /*K=*/{kT});
  w.gop.fence();

  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
#ifdef TA_STRIDED_DGEMM_COUNT
  std::size_t fires = TA::detail::g_strided_dgemm_ce_ce_right_calls.load() +
                      TA::detail::g_strided_dgemm_ce_ce_left_calls.load();
  w.gop.sum(&fires, 1);
  BOOST_CHECK_GT(fires, std::size_t{0});
  // At np>1 the ride-single-tile coarse-H regime must engage the h-grouped
  // (proc_h_ > 1) active distribution; at np=1 it stays ungrouped (counter 0).
  std::size_t grouped = TA::detail::g_summa_proc_h_grouped_calls.load();
  w.gop.sum(&grouped, 1);
  if (w.size() > 1)
    BOOST_CHECK_GT(grouped, std::size_t{0});
  else
    BOOST_CHECK_EQUAL(grouped, std::size_t{0});
#endif
}

// ===========================================================================
// Structurally-ABSENT SUMMA role on the active path (regression for the
// empty-role pack-box bug).
//
// A contraction need not populate every SUMMA role. hce_ce is the canonical
// example: C(h,i;a) = A(h,i,k;c) * B(h,k;a,c) has SUMMA-N = EMPTY (the right
// operand contributes no OUTER external). The active two-trange path used to
// inject a spurious degenerate axis into the operand pack box for an absent
// role (append_role_t_box's empty-role branch), over-ranking the packed tile
// vs the gathered fine tiles -> the strided kernel's shape check rejected it ->
// the GEMM never fired and the result came back SILENTLY EMPTY. The fix makes
// an absent role contribute NO pack-box axis (mirroring the guarded Hadamard
// append), so the box rank equals the operand's outer rank.
//
// The ce_ce strided kernel rides an OUTER external (never the inner indices --
// those are already the per-cell BLAS M/N/K): ce_ce_right rides the RIGHT outer
// external (N), ce_ce_left rides the LEFT outer external (M); the variant is
// fixed by which operand carries the inner external. So to also assert the fat
// GEMM FIRES, each case is built like the real workload -- the inner external
// sits on the operand whose PARTNER's outer external is present, riding the
// present axis: absent-N rides the present M (ce_ce_left), absent-M rides the
// present N (ce_ce_right). The strided fire counter is reset AFTER the oracle
// so it measures the RETILE alone (not the oracle einsum's fires).
BOOST_AUTO_TEST_CASE(retile_active_absent_N_coarsenHK_ce_ce_dist) {
  auto& w = TA::get_default_world();

  auto hU = tr1d(8, 2);   // 4 fused H U-tiles
  auto hT = tr1d(8, 4);   // coarsen H: 4 U -> 2 coarse slabs
  auto i = tr1d(4, 2);    // 2 left-external M-tiles (PRESENT, ridden)
  auto kU = tr1d(8, 4);   // 2 contracted K U-tiles
  auto kT = tr1d(8, 8);   // coarsen K: 2 U K-tiles -> 1 coarse T K-tile
  // A(h,i,k; c): M present (i), inner c=4 = pure contraction.
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{hU, i, kU}, TA::Range{4});
  // B(h,k; a,c): NO right OUTER external => SUMMA-N structurally ABSENT; inner
  // a=3 is the (right) inner external that rides into BLAS, c=4 contracted.
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{hU, kU}, TA::Range{3, 4});
  w.gop.fence();

  reset_plan_active_count();

  // Oracle: the result carries NO N mode -> C(h,i;a).
  ArrayToT2 C_ref = TA::einsum(A0("h,i,k;c"), B0("h,k;a,c"), "h,i;a");
  w.gop.fence();

#ifdef TA_STRIDED_DGEMM_COUNT
  // Reset AFTER the oracle: measure the RETILE's strided fires only.
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
#endif

  // ACTIVE retile: H coarsened, M identity, N ABSENT (empty target == no axis),
  // K coarsened. Exercises append_role_t_box's empty-N role path; the fat GEMM
  // rides the present M external (ce_ce_left).
  ArrayToT2 C;
  C("h,i;a") = (A0("h,i,k;c") * B0("h,k;a,c"))
                   .retile(/*H=*/{hT}, /*M=*/{i}, /*N=*/{}, /*K=*/{kT});
  w.gop.fence();

  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
#ifdef TA_STRIDED_DGEMM_COUNT
  std::size_t fires = TA::detail::g_strided_dgemm_ce_ce_right_calls.load() +
                      TA::detail::g_strided_dgemm_ce_ce_left_calls.load();
  w.gop.sum(&fires, 1);
  BOOST_CHECK_GT(fires, std::size_t{0});
#endif
}

// Mirror of the absent-N case with the LEFT operand carrying no OUTER external,
// so SUMMA-M is structurally absent: C(h,j;a) = A(h,k;a,c) * B(h,j,k;c). The
// inner external a sits on the LEFT, so the fat GEMM rides the present N
// external (ce_ce_right).
BOOST_AUTO_TEST_CASE(retile_active_absent_M_coarsenHK_ce_ce_dist) {
  auto& w = TA::get_default_world();

  auto hU = tr1d(8, 2);   // 4 fused H U-tiles
  auto hT = tr1d(8, 4);   // coarsen H: 4 U -> 2 coarse slabs
  auto j = tr1d(4, 2);    // 2 right-external N-tiles (PRESENT, ridden)
  auto kU = tr1d(8, 4);   // 2 contracted K U-tiles
  auto kT = tr1d(8, 8);   // coarsen K: 2 U K-tiles -> 1 coarse T K-tile
  // A(h,k; a,c): NO left OUTER external => SUMMA-M structurally ABSENT; inner
  // a=3 = (left) inner external that rides, c=4 contracted.
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{hU, kU}, TA::Range{3, 4});
  // B(h,j,k; c): N present (j), inner c=4 = pure contraction.
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{hU, j, kU}, TA::Range{4});
  w.gop.fence();

  reset_plan_active_count();

  // Oracle: the result carries NO M mode -> C(h,j;a).
  ArrayToT2 C_ref = TA::einsum(A0("h,k;a,c"), B0("h,j,k;c"), "h,j;a");
  w.gop.fence();

#ifdef TA_STRIDED_DGEMM_COUNT
  // Reset AFTER the oracle: measure the RETILE's strided fires only.
  TA::detail::g_strided_dgemm_ce_ce_right_calls.store(0);
  TA::detail::g_strided_dgemm_ce_ce_left_calls.store(0);
#endif

  // ACTIVE retile: H coarsened, M ABSENT (empty target == no axis), N identity,
  // K coarsened. Exercises append_role_t_box's empty-M role path; the fat GEMM
  // rides the present N external (ce_ce_right).
  ArrayToT2 C;
  C("h,j;a") = (A0("h,k;a,c") * B0("h,j,k;c"))
                   .retile(/*H=*/{hT}, /*M=*/{}, /*N=*/{j}, /*K=*/{kT});
  w.gop.fence();

  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
#ifdef TA_STRIDED_DGEMM_COUNT
  std::size_t fires = TA::detail::g_strided_dgemm_ce_ce_right_calls.load() +
                      TA::detail::g_strided_dgemm_ce_ce_left_calls.load();
  w.gop.sum(&fires, 1);
  BOOST_CHECK_GT(fires, std::size_t{0});
#endif
}

// Absent SUMMA-N on the ce_e (inner OUTER-PRODUCT) kernel. Unlike ce_ce (which
// rides an outer EXTERNAL), ce_e rides the outer CONTRACTION (K), which is
// always present -- so ce_e fires regardless of an absent external (the absent
// N is passed to the kernel as the degenerate extent 1, never the early-return
// 0). Confirms the empty-role pack-box fix and ce_e's K-ride together.
//   C(h,m;a,b) = A(h,m,k;a) * B(h,k;b)   (no inner contraction; N absent)
BOOST_AUTO_TEST_CASE(retile_active_absent_N_coarsenHK_ce_e_dist) {
  auto& w = TA::get_default_world();

  auto hU = tr1d(8, 2);   // 4 fused H U-tiles
  auto hT = tr1d(8, 4);   // coarsen H: 4 U -> 2 coarse slabs
  auto m = tr1d(4, 2);    // 2 left-external M-tiles
  auto kU = tr1d(8, 4);   // 2 contracted K U-tiles
  auto kT = tr1d(8, 8);   // coarsen K: 2 U K-tiles -> 1 coarse T K-tile
  // A(h,m,k; a): inner a (spectator -> result), NO inner contraction.
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{hU, m, kU}, TA::Range{3});
  // B(h,k; b): NO right external => SUMMA-N absent; inner b (spectator).
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{hU, kU}, TA::Range{2});
  w.gop.fence();

  reset_plan_active_count();

  // Oracle: ce_e outer product, result carries NO N mode -> C(h,m;a,b).
  ArrayToT2 C_ref = TA::einsum(A0("h,m,k;a"), B0("h,k;b"), "h,m;a,b");
  w.gop.fence();

#ifdef TA_STRIDED_DGEMM_COUNT
  TA::detail::g_strided_dgemm_ce_e_calls.store(0);  // retile-only
#endif

  ArrayToT2 C;
  C("h,m;a,b") = (A0("h,m,k;a") * B0("h,k;b"))
                     .retile(/*H=*/{hT}, /*M=*/{m}, /*N=*/{}, /*K=*/{kT});
  w.gop.fence();

  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
#ifdef TA_STRIDED_DGEMM_COUNT
  // ce_e rides the (present) outer contraction K, so it fires despite absent N.
  std::size_t fires = TA::detail::g_strided_dgemm_ce_e_calls.load();
  w.gop.sum(&fires, 1);
  BOOST_CHECK_GT(fires, std::size_t{0});
#endif
}

// Operand permutation (canonicalization commit e439c747) composed with an
// ACTIVE retile. The left operand is annotated non-canonically (k leads the
// externals: "h,k,i") so the engine physically permutes it to canonical
// NoTranspose layout BEFORE the coarsen pack runs (permute-then-retile). Result
// is requested in canonical order, so this isolates the OPERAND permute path
// from the result-permute path. Must match the no-retile oracle at np=1/2.
BOOST_AUTO_TEST_CASE(retile_active_operand_permuted_ce_ce_dist) {
  auto& w = TA::get_default_world();
  auto h = tr1d(4, 2);    // 2 fused H tiles
  auto i = tr1d(8, 2);    // 4 left-external M U-tiles ...
  auto iT = tr1d(8, 8);   // ... coarsened to ONE coarse M tile
  auto j = tr1d(4, 2);    // 2 right-external N U-tiles (identity)
  auto kU = tr1d(8, 2);   // 4 contracted K U-tiles ...
  auto kT = tr1d(8, 8);   // ... coarsened to ONE coarse K tile
  // Left stored as (h,k,i): k (contracted) precedes i (external) -> non-canonical
  // operand layout -> engine inserts a physical operand permute to (h,i,k).
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{h, kU, i}, TA::Range{3, 4});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{h, j, kU}, TA::Range{4});
  w.gop.fence();

  reset_plan_active_count();

  ArrayToT2 C_ref = TA::einsum(A0("h,k,i;a,c"), B0("h,j,k;c"), "h,i,j;a");
  w.gop.fence();

  ArrayToT2 C;
  C("h,i,j;a") = (A0("h,k,i;a,c") * B0("h,j,k;c"))
                     .retile(/*H=*/{h}, /*M=*/{iT}, /*N=*/{j}, /*K=*/{kT});
  w.gop.fence();
  BOOST_CHECK(C.trange() == C_ref.trange());
  BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
}

// ===========================================================================
// PERMSWEEP: exhaustive cross-product verification of the active two-trange
// retile SUMMA path for ToT contractions. For each ALLOWED coarsen-only retile
// target STRUCTURE (subset lattice over the 6 role axes {H, M1, M2, N1, N2, K},
// 64 structures incl. the all-identity inactive plan) and for EVERY permutation
// of the n=5 outer result indices {h,i1,i2,j1,j2} (120 perms via
// std::next_permutation), compare the active-retile result against the stock
// einsum oracle into the SAME permuted layout. Catches asserts / hangs / wrong
// values that single-permutation, single-structure tests miss.
//
//   ce_ce:  A("h,i1,i2,k;a,c") * B("h,j1,j2,k;c")  -> outer {h,i1,i2,j1,j2}; a
//           (inner) rides, c contracted.
//   ce_e:  A("h,i1,i2,k;a")   * B("h,j1,j2,k;b")  -> outer {h,i1,i2,j1,j2};
//           inner a/b both ride (no shared inner contraction).
//
// Coarsening = collapse the operand's 2-tile axis to ONE tile (the only coarsen
// factor available at 2 tiles). H is kept a SINGLE uniform dense axis, so every
// structure (incl. H-coarsen combined with M/N/K) is np>1-ALLOWED; no structure
// is expected to throw for these shapes. A hang reveals itself as a "trying"
// line with no matching "ok"; flush after every line.
//
// PERMSWEEP_MODE env: "prio" (DEFAULT) = the prioritized subset {identity,
// each single role alone, all-roles} = 8 structures (fast CI smoke; ~100x120
// points/run); "full" = the exhaustive 64-structure coarsen subset lattice
// (~7680 points/run, ~100s). The full sweep is opt-in so the committed
// regression stays CI-light while remaining exhaustively runnable on demand
// (PERMSWEEP_MODE=full). The full sweep was validated ALL-PASS at np=1 and
// np=2 for both kernels (see agent/.../permsweep-report.md).
namespace {
inline std::string permsweep_join(const std::vector<std::string>& v,
                                  const char* sep) {
  std::string s;
  for (std::size_t i = 0; i < v.size(); ++i) {
    if (i) s += sep;
    s += v[i];
  }
  return s;
}
// Enumerate the structure bitmasks to run, honoring PERMSWEEP_MODE.
inline std::vector<unsigned> permsweep_masks() {
  const char* m = std::getenv("PERMSWEEP_MODE");
  const bool full = (m && std::string(m) == "full");
  std::vector<unsigned> out;
  if (!full) {
    out.push_back(0u);                                        // identity
    for (unsigned b = 0; b < 6; ++b) out.push_back(1u << b);  // singles
    out.push_back(0x3Fu);                                     // all roles
  } else {
    for (unsigned mask = 0; mask < 64u; ++mask) out.push_back(mask);
  }
  return out;
}
inline std::string permsweep_struct_label(unsigned mask) {
  if (mask == 0) return "identity";
  static const char* names[6] = {"H", "M1", "M2", "N1", "N2", "K"};
  std::vector<std::string> on;
  for (unsigned b = 0; b < 6; ++b)
    if (mask & (1u << b)) on.emplace_back(names[b]);
  return permsweep_join(on, "+");
}
inline bool permsweep_full_mode() {
  const char* m = std::getenv("PERMSWEEP_MODE");
  return m && std::string(m) == "full";
}

// Two-fused (h1,h2 hadamard) permutation generality for a single kernel,
// parameterized by the per-operand inner annotations. A0/B0 are the canonical
// operands over outer (h1,h2,<A-ext>,<B-ext>,k). Runs four ADDITIVE sweeps:
//   - result layout (6! = 720): permute the 6 result modes; oracle = the
//     canonical result permute-assigned into the same order.
//   - A layout (5! = 120): present A in each physical order via
//     permute-assignment; contract with canonical B into the canonical result.
//   - B layout (5! = 120): symmetric.
//   - random joint sample (A,B,result all permuted): a bounded fixed-seed
//     sample (identical on every rank) guarding residual cross-term risk.
// PERMSWEEP_MODE=full runs every permutation + 256 joint samples; the default
// strides the large loops down (~24 each) + 16 joint so committed CI stays
// light while the exhaustive sweep stays runnable on demand. The full sweep was
// validated ALL-PASS at np=1 and np=2 for both kernels.
inline void run_two_hadamard_sweep(TA::World& w, const ArrayToT2& A0,
                                   const ArrayToT2& B0, const char* a_inner,
                                   const char* b_inner, const char* c_inner,
                                   const char* label) {
  const bool root = (w.rank() == 0);
  const bool full = permsweep_full_mode();
  const std::string Aann = std::string("h1,h2,i1,i2,k") + a_inner;
  const std::string Bann = std::string("h1,h2,j1,j2,k") + b_inner;
  const std::vector<std::string> Cmodes0{"h1", "h2", "i1", "i2", "j1", "j2"};
  const std::string canonC = permsweep_join(Cmodes0, ",") + c_inner;

  ArrayToT2 C_canon = TA::einsum(A0(Aann), B0(Bann), canonC);
  w.gop.fence();
  BOOST_REQUIRE_GT(C_canon.trange().tiles_range().volume(), 0u);

  std::size_t checks = 0;
  auto note = [&](const char* phase, const std::string& s) {
    if (root) {
      std::fprintf(stderr, "[2HAD] %s %s %s\n", label, phase, s.c_str());
      std::fflush(stderr);
    }
  };

  // ---- result-layout sweep (6!) ----
  {
    std::vector<std::string> base = Cmodes0;
    std::sort(base.begin(), base.end());
    std::size_t idx = 0;
    const std::size_t stride = full ? 1u : 30u;  // 720 -> ~24
    do {
      if ((idx++ % stride) != 0) continue;
      const std::string res = permsweep_join(base, ",") + c_inner;
      note("result", res);
      ArrayToT2 C = TA::einsum(A0(Aann), B0(Bann), res);
      w.gop.fence();
      ArrayToT2 C_oracle;
      C_oracle(res) = C_canon(canonC);
      w.gop.fence();
      BOOST_CHECK(C.trange() == C_oracle.trange());
      BOOST_CHECK_SMALL(array_max_reldiff2(C, C_oracle), 1e-9);
      ++checks;
    } while (std::next_permutation(base.begin(), base.end()));
  }

  // ---- A-layout sweep (5!) ----
  {
    std::vector<std::string> Am{"h1", "h2", "i1", "i2", "k"};
    std::sort(Am.begin(), Am.end());
    std::size_t idx = 0;
    const std::size_t stride = full ? 1u : 5u;  // 120 -> 24
    do {
      if ((idx++ % stride) != 0) continue;
      const std::string aperm = permsweep_join(Am, ",") + a_inner;
      note("A", aperm);
      ArrayToT2 A_sig;
      A_sig(aperm) = A0(Aann);
      w.gop.fence();
      ArrayToT2 C = TA::einsum(A_sig(aperm), B0(Bann), canonC);
      w.gop.fence();
      BOOST_CHECK(C.trange() == C_canon.trange());
      BOOST_CHECK_SMALL(array_max_reldiff2(C, C_canon), 1e-9);
      ++checks;
    } while (std::next_permutation(Am.begin(), Am.end()));
  }

  // ---- B-layout sweep (5!) ----
  {
    std::vector<std::string> Bm{"h1", "h2", "j1", "j2", "k"};
    std::sort(Bm.begin(), Bm.end());
    std::size_t idx = 0;
    const std::size_t stride = full ? 1u : 5u;  // 120 -> 24
    do {
      if ((idx++ % stride) != 0) continue;
      const std::string bperm = permsweep_join(Bm, ",") + b_inner;
      note("B", bperm);
      ArrayToT2 B_sig;
      B_sig(bperm) = B0(Bann);
      w.gop.fence();
      ArrayToT2 C = TA::einsum(A0(Aann), B_sig(bperm), canonC);
      w.gop.fence();
      BOOST_CHECK(C.trange() == C_canon.trange());
      BOOST_CHECK_SMALL(array_max_reldiff2(C, C_canon), 1e-9);
      ++checks;
    } while (std::next_permutation(Bm.begin(), Bm.end()));
  }

  // ---- random joint sample (A,B,result all permuted) ----
  {
    std::mt19937 rng(20260626u);  // fixed seed => identical sequence per rank
    const std::size_t N = full ? 256u : 16u;
    std::vector<std::string> Am{"h1", "h2", "i1", "i2", "k"};
    std::vector<std::string> Bm{"h1", "h2", "j1", "j2", "k"};
    std::vector<std::string> Cm = Cmodes0;
    for (std::size_t s = 0; s < N; ++s) {
      std::shuffle(Am.begin(), Am.end(), rng);
      std::shuffle(Bm.begin(), Bm.end(), rng);
      std::shuffle(Cm.begin(), Cm.end(), rng);
      const std::string aperm = permsweep_join(Am, ",") + a_inner;
      const std::string bperm = permsweep_join(Bm, ",") + b_inner;
      const std::string res = permsweep_join(Cm, ",") + c_inner;
      note("joint", aperm + " * " + bperm + " -> " + res);
      ArrayToT2 A_sig, B_sig;
      A_sig(aperm) = A0(Aann);
      B_sig(bperm) = B0(Bann);
      w.gop.fence();
      ArrayToT2 C = TA::einsum(A_sig(aperm), B_sig(bperm), res);
      w.gop.fence();
      ArrayToT2 C_oracle;
      C_oracle(res) = C_canon(canonC);
      w.gop.fence();
      BOOST_CHECK(C.trange() == C_oracle.trange());
      BOOST_CHECK_SMALL(array_max_reldiff2(C, C_oracle), 1e-9);
      ++checks;
    }
  }

  BOOST_CHECK_GT(checks, std::size_t{0});
  if (root)
    std::fprintf(stderr, "[2HAD] %s done: %zu checks (full=%d)\n", label, checks,
                 static_cast<int>(full));
}
}  // namespace

// ce_ce kernel: inner a rides, inner c contracted. n=5 outer -> 120 perms.
BOOST_AUTO_TEST_CASE(permsweep_all_perms_all_structs_ce_ce_dist) {
  auto& w = TA::get_default_world();
  const bool root = (w.rank() == 0);

  // Fine (U) operand tilings.
  auto h = tr1d(4, 2), i1 = tr1d(4, 2), i2 = tr1d(4, 2);
  auto j1 = tr1d(4, 2), j2 = tr1d(4, 2), kU = tr1d(8, 4);
  // Coarse (collapse-to-1) targets.
  auto hT = tr1d(4, 4), i1T = tr1d(4, 4), i2T = tr1d(4, 4);
  auto j1T = tr1d(4, 4), j2T = tr1d(4, 4), kT = tr1d(8, 8);

  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{h, i1, i2, kU}, TA::Range{3, 4});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{h, j1, j2, kU}, TA::Range{4});
  w.gop.fence();
  reset_plan_active_count();

  for (unsigned mask : permsweep_masks()) {
    const std::string slabel = permsweep_struct_label(mask);
    std::vector<TA::TiledRange1> Hv{(mask & 1u) ? hT : h};
    std::vector<TA::TiledRange1> Mv{(mask & 2u) ? i1T : i1,
                                    (mask & 4u) ? i2T : i2};
    std::vector<TA::TiledRange1> Nv{(mask & 8u) ? j1T : j1,
                                    (mask & 16u) ? j2T : j2};
    std::vector<TA::TiledRange1> Kv{(mask & 32u) ? kT : kU};

    std::vector<std::string> base{"h", "i1", "i2", "j1", "j2"};
    std::sort(base.begin(), base.end());
    do {
      const std::string outer = permsweep_join(base, ",");
      const std::string res = outer + ";a";
      if (root) {
        std::fprintf(stderr, "[PERMSWEEP] ce_ce struct=%s trying %s\n",
                     slabel.c_str(), outer.c_str());
        std::fflush(stderr);
      }
      ArrayToT2 C_ref =
          TA::einsum(A0("h,i1,i2,k;a,c"), B0("h,j1,j2,k;c"), res);
      w.gop.fence();
      ArrayToT2 C;
      C(res) = (A0("h,i1,i2,k;a,c") * B0("h,j1,j2,k;c")).retile(Hv, Mv, Nv, Kv);
      w.gop.fence();
      BOOST_CHECK(C.trange() == C_ref.trange());
      BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
      if (root) {
        std::fprintf(stderr, "[PERMSWEEP] ce_ce struct=%s ok %s\n",
                     slabel.c_str(), outer.c_str());
        std::fflush(stderr);
      }
    } while (std::next_permutation(base.begin(), base.end()));
  }
  // Active path must have fired at least once across the active structures.
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
}

// ce_e kernel: inner a/b both ride (no shared inner contraction). 120 perms.
BOOST_AUTO_TEST_CASE(permsweep_all_perms_all_structs_ce_e_dist) {
  auto& w = TA::get_default_world();
  const bool root = (w.rank() == 0);

  auto h = tr1d(4, 2), i1 = tr1d(4, 2), i2 = tr1d(4, 2);
  auto j1 = tr1d(4, 2), j2 = tr1d(4, 2), kU = tr1d(8, 4);
  auto hT = tr1d(4, 4), i1T = tr1d(4, 4), i2T = tr1d(4, 4);
  auto j1T = tr1d(4, 4), j2T = tr1d(4, 4), kT = tr1d(8, 8);

  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{h, i1, i2, kU}, TA::Range{3});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{h, j1, j2, kU}, TA::Range{5});
  w.gop.fence();
  reset_plan_active_count();

  for (unsigned mask : permsweep_masks()) {
    const std::string slabel = permsweep_struct_label(mask);
    std::vector<TA::TiledRange1> Hv{(mask & 1u) ? hT : h};
    std::vector<TA::TiledRange1> Mv{(mask & 2u) ? i1T : i1,
                                    (mask & 4u) ? i2T : i2};
    std::vector<TA::TiledRange1> Nv{(mask & 8u) ? j1T : j1,
                                    (mask & 16u) ? j2T : j2};
    std::vector<TA::TiledRange1> Kv{(mask & 32u) ? kT : kU};

    std::vector<std::string> base{"h", "i1", "i2", "j1", "j2"};
    std::sort(base.begin(), base.end());
    do {
      const std::string outer = permsweep_join(base, ",");
      const std::string res = outer + ";a,b";
      if (root) {
        std::fprintf(stderr, "[PERMSWEEP] ce_e struct=%s trying %s\n",
                     slabel.c_str(), outer.c_str());
        std::fflush(stderr);
      }
      ArrayToT2 C_ref = TA::einsum(A0("h,i1,i2,k;a"), B0("h,j1,j2,k;b"), res);
      w.gop.fence();
      ArrayToT2 C;
      C(res) = (A0("h,i1,i2,k;a") * B0("h,j1,j2,k;b")).retile(Hv, Mv, Nv, Kv);
      w.gop.fence();
      BOOST_CHECK(C.trange() == C_ref.trange());
      BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
      if (root) {
        std::fprintf(stderr, "[PERMSWEEP] ce_e struct=%s ok %s\n",
                     slabel.c_str(), outer.c_str());
        std::fflush(stderr);
      }
    } while (std::next_permutation(base.begin(), base.end()));
  }
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
}

// ===========================================================================
// MULTI-HADAMARD permutation generality (h1,h2 both fused). Hadamard indices do
// NOT enter the strided DGEMM (they are batch/parallelization axes the kernel
// rides over) and are NOT retileable, so this is a PURE result/operand layout
// permutation concern -- orthogonal to the retile-structure sweep above. The
// engine permutes A into canonical, B into canonical, and the result out of
// canonical as three INDEPENDENT steps, so additive single-variable sweeps
// (A alone, B alone, result alone) fully exercise permutation handling; a
// bounded random joint sample guards residual cross-term risk. Oracle = stock
// einsum; operand-layout variants are built by permute-assignment (reorders the
// canonical values; make_arena2's seed is order-dependent so a fresh build in a
// permuted trange would NOT be the same logical tensor). np=1 AND np=2.
//
// ce_ce kernel (inner a rides on A, inner c contracted): two fused indices.
BOOST_AUTO_TEST_CASE(permsweep_two_hadamard_ce_ce_dist) {
  auto& w = TA::get_default_world();
  auto h1 = tr1d(4, 2), h2 = tr1d(4, 2), i1 = tr1d(4, 2), i2 = tr1d(4, 2);
  auto j1 = tr1d(4, 2), j2 = tr1d(4, 2), kU = tr1d(8, 4);
  ArrayToT2 A0 =
      make_arena2(w, TA::TiledRange{h1, h2, i1, i2, kU}, TA::Range{3, 4});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{h1, h2, j1, j2, kU}, TA::Range{4});
  w.gop.fence();
  run_two_hadamard_sweep(w, A0, B0, ";a,c", ";c", ";a", "ce_ce");
}

// ce_e kernel (inner a/b both ride, no shared inner contraction): two fused.
BOOST_AUTO_TEST_CASE(permsweep_two_hadamard_ce_e_dist) {
  auto& w = TA::get_default_world();
  auto h1 = tr1d(4, 2), h2 = tr1d(4, 2), i1 = tr1d(4, 2), i2 = tr1d(4, 2);
  auto j1 = tr1d(4, 2), j2 = tr1d(4, 2), kU = tr1d(8, 4);
  ArrayToT2 A0 = make_arena2(w, TA::TiledRange{h1, h2, i1, i2, kU}, TA::Range{3});
  ArrayToT2 B0 = make_arena2(w, TA::TiledRange{h1, h2, j1, j2, kU}, TA::Range{5});
  w.gop.fence();
  run_two_hadamard_sweep(w, A0, B0, ";a", ";b", ";a,b", "ce_e");
}

// NON-UNIFORM inner extents (the defining ToT property) through active retile +
// result permutation -- the axis the uniform permsweep above does NOT cover and
// the only structural difference left between the (clean) test harness and the
// bench. Inner extents depend ONLY on the non-contracted (external/fused) outer
// indices so the K-fold contraction stays well-defined. Oracle = stock einsum
// into the SAME permuted layout; array_max_reldiff2 is position-sensitive. All
// 120 outer permutations x {identity, K-coarsen}, np=1 AND np=2.
BOOST_AUTO_TEST_CASE(permsweep_nonuniform_ce_e_dist) {
  auto& w = TA::get_default_world();
  const bool root = (w.rank() == 0);
  auto h = tr1d(4, 2), i1 = tr1d(4, 2), i2 = tr1d(4, 2);
  auto j1 = tr1d(4, 2), j2 = tr1d(4, 2), kU = tr1d(8, 4);
  auto kT = tr1d(8, 8);  // coarsen K (the contracted outer) to a single tile
  // a rides on A, keyed by A's external (h,i1,i2) -- NOT k; b rides on B,
  // keyed by (h,j1,j2). Extents vary 2..4, so cells have differing sizes.
  auto a_ext = [](const std::vector<std::size_t>& x) {
    return TA::Range{long(2 + (x[0] + x[1] + x[2]) % 3)};
  };
  auto b_ext = [](const std::vector<std::size_t>& x) {
    return TA::Range{long(2 + (2 * x[0] + x[1] + 3 * x[2]) % 3)};
  };
  ArrayToT2 A0 = make_arena2_nu(w, TA::TiledRange{h, i1, i2, kU}, a_ext);
  ArrayToT2 B0 = make_arena2_nu(w, TA::TiledRange{h, j1, j2, kU}, b_ext);
  w.gop.fence();
  reset_plan_active_count();
  for (unsigned mask : {0u, 32u}) {  // identity and K-coarsen
    std::vector<TA::TiledRange1> Hv{h}, Mv{i1, i2}, Nv{j1, j2};
    std::vector<TA::TiledRange1> Kv{(mask & 32u) ? kT : kU};
    std::vector<std::string> base{"h", "i1", "i2", "j1", "j2"};
    std::sort(base.begin(), base.end());
    do {
      const std::string outer = permsweep_join(base, ",");
      const std::string res = outer + ";a,b";
      if (root) {
        std::fprintf(stderr, "[PERMSWEEP-NU] ce_e mask=%u trying %s\n", mask,
                     outer.c_str());
        std::fflush(stderr);
      }
      ArrayToT2 C_ref = TA::einsum(A0("h,i1,i2,k;a"), B0("h,j1,j2,k;b"), res);
      w.gop.fence();
      ArrayToT2 C;
      C(res) = (A0("h,i1,i2,k;a") * B0("h,j1,j2,k;b")).retile(Hv, Mv, Nv, Kv);
      w.gop.fence();
      BOOST_CHECK(C.trange() == C_ref.trange());
      BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
      if (root) {
        std::fprintf(stderr, "[PERMSWEEP-NU] ce_e mask=%u ok %s\n", mask,
                     outer.c_str());
        std::fflush(stderr);
      }
    } while (std::next_permutation(base.begin(), base.end()));
  }
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
}

// ce_ce non-uniform: rider a varies with (h,i1,i2); the CONTRACTED inner c is
// kept uniform (width 4) so the inner GEMM stays well-defined. 120 perms x
// {identity, K-coarsen}, np=1 AND np=2.
BOOST_AUTO_TEST_CASE(permsweep_nonuniform_ce_ce_dist) {
  auto& w = TA::get_default_world();
  const bool root = (w.rank() == 0);
  auto h = tr1d(4, 2), i1 = tr1d(4, 2), i2 = tr1d(4, 2);
  auto j1 = tr1d(4, 2), j2 = tr1d(4, 2), kU = tr1d(8, 4);
  auto kT = tr1d(8, 8);
  auto a_ext = [](const std::vector<std::size_t>& x) {
    return TA::Range{long(2 + (x[0] + x[1] + x[2]) % 3), 4l};  // {a(var), c=4}
  };
  auto c_ext = [](const std::vector<std::size_t>&) {
    return TA::Range{4l};  // {c=4} only
  };
  ArrayToT2 A0 = make_arena2_nu(w, TA::TiledRange{h, i1, i2, kU}, a_ext);
  ArrayToT2 B0 = make_arena2_nu(w, TA::TiledRange{h, j1, j2, kU}, c_ext);
  w.gop.fence();
  reset_plan_active_count();
  for (unsigned mask : {0u, 32u}) {
    std::vector<TA::TiledRange1> Hv{h}, Mv{i1, i2}, Nv{j1, j2};
    std::vector<TA::TiledRange1> Kv{(mask & 32u) ? kT : kU};
    std::vector<std::string> base{"h", "i1", "i2", "j1", "j2"};
    std::sort(base.begin(), base.end());
    do {
      const std::string outer = permsweep_join(base, ",");
      const std::string res = outer + ";a";
      if (root) {
        std::fprintf(stderr, "[PERMSWEEP-NU] ce_ce mask=%u trying %s\n", mask,
                     outer.c_str());
        std::fflush(stderr);
      }
      ArrayToT2 C_ref =
          TA::einsum(A0("h,i1,i2,k;a,c"), B0("h,j1,j2,k;c"), res);
      w.gop.fence();
      ArrayToT2 C;
      C(res) =
          (A0("h,i1,i2,k;a,c") * B0("h,j1,j2,k;c")).retile(Hv, Mv, Nv, Kv);
      w.gop.fence();
      BOOST_CHECK(C.trange() == C_ref.trange());
      BOOST_CHECK_SMALL(array_max_reldiff2(C, C_ref), 1e-9);
      if (root) {
        std::fprintf(stderr, "[PERMSWEEP-NU] ce_ce mask=%u ok %s\n", mask,
                     outer.c_str());
        std::fflush(stderr);
      }
    } while (std::next_permutation(base.begin(), base.end()));
  }
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
}

// SPARSE permuted-result retile sweep. The bench (which crashed/mis-computed)
// and the pre-existing csv_like failure are BOTH SparsePolicy + result-permute;
// every clean test above is DensePolicy, so this closes that coverage gap with
// a clean harness. Outer-tile sparsity (a subset of outer tiles absent), uniform
// inner, K-coarsen retile, all 120 outer permutations, np=1 AND np=2. Oracle =
// stock einsum into the SAME permuted layout; array_max_reldiff_sparse is
// tile-index-matched (permutation-sensitive) and flags present-on-one-side tiles.
BOOST_AUTO_TEST_CASE(permsweep_sparse_ce_e_dist) {
  auto& w = TA::get_default_world();
  const bool root = (w.rank() == 0);
  const float saved = TA::SparseShape<float>::threshold();
  TA::SparseShape<float>::threshold(std::numeric_limits<float>::min());
  auto h = tr1d(4, 2), i1 = tr1d(4, 2), i2 = tr1d(4, 2);
  auto j1 = tr1d(4, 2), j2 = tr1d(4, 2), kU = tr1d(8, 4);
  auto kT = tr1d(8, 8);
  auto azero = [](const auto& x) {
    return ((x[0] + x[1] + x[2] + x[3]) % 4) == 0;  // ~25% of A outer absent
  };
  auto bzero = [](const auto& x) {
    return ((x[0] + x[1] + x[2] + x[3]) % 5) == 0;  // ~20% of B outer absent
  };
  ArrayToTSparse A0 =
      make_arena_sparse(w, TA::TiledRange{h, i1, i2, kU}, TA::Range{3}, azero);
  ArrayToTSparse B0 =
      make_arena_sparse(w, TA::TiledRange{h, j1, j2, kU}, TA::Range{5}, bzero);
  w.gop.fence();
  reset_plan_active_count();
  for (unsigned mask : {0u, 32u}) {
    std::vector<TA::TiledRange1> Hv{h}, Mv{i1, i2}, Nv{j1, j2};
    std::vector<TA::TiledRange1> Kv{(mask & 32u) ? kT : kU};
    std::vector<std::string> base{"h", "i1", "i2", "j1", "j2"};
    std::sort(base.begin(), base.end());
    do {
      const std::string outer = permsweep_join(base, ",");
      const std::string res = outer + ";a,b";
      if (root) {
        std::fprintf(stderr, "[PERMSWEEP-SP] ce_e mask=%u trying %s\n", mask,
                     outer.c_str());
        std::fflush(stderr);
      }
      ArrayToTSparse C_ref =
          TA::einsum(A0("h,i1,i2,k;a"), B0("h,j1,j2,k;b"), res);
      w.gop.fence();
      ArrayToTSparse C;
      C(res) = (A0("h,i1,i2,k;a") * B0("h,j1,j2,k;b")).retile(Hv, Mv, Nv, Kv);
      w.gop.fence();
      BOOST_CHECK(C.trange() == C_ref.trange());
      BOOST_CHECK_SMALL(array_max_reldiff_sparse(C, C_ref), 1e-9);
      if (root) {
        std::fprintf(stderr, "[PERMSWEEP-SP] ce_e mask=%u ok %s\n", mask,
                     outer.c_str());
        std::fflush(stderr);
      }
    } while (std::next_permutation(base.begin(), base.end()));
  }
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
  TA::SparseShape<float>::threshold(saved);
}

// SPARSE + NON-UNIFORM combined -- the bench's exact domain (the bench that
// crashed). If THIS clean harness passes, the engine's retile+repermute is
// exonerated on every axis the bench combines, isolating the bench's own
// machinery as the corruption source. K-coarsen, all 120 perms, np=1 AND np=2.
BOOST_AUTO_TEST_CASE(permsweep_sparse_nonuniform_ce_e_dist) {
  auto& w = TA::get_default_world();
  const bool root = (w.rank() == 0);
  const float saved = TA::SparseShape<float>::threshold();
  TA::SparseShape<float>::threshold(std::numeric_limits<float>::min());
  auto h = tr1d(4, 2), i1 = tr1d(4, 2), i2 = tr1d(4, 2);
  auto j1 = tr1d(4, 2), j2 = tr1d(4, 2), kU = tr1d(8, 4);
  auto kT = tr1d(8, 8);
  auto azero = [](const auto& x) { return ((x[0] + x[1] + x[2] + x[3]) % 4) == 0; };
  auto bzero = [](const auto& x) { return ((x[0] + x[1] + x[2] + x[3]) % 5) == 0; };
  auto a_ext = [](const std::vector<std::size_t>& x) {
    return TA::Range{long(2 + (x[0] + x[1] + x[2]) % 3)};  // rider, keyed by ext
  };
  auto b_ext = [](const std::vector<std::size_t>& x) {
    return TA::Range{long(2 + (2 * x[0] + x[1] + 3 * x[2]) % 3)};
  };
  ArrayToTSparse A0 =
      make_arena_sparse_nu(w, TA::TiledRange{h, i1, i2, kU}, azero, a_ext);
  ArrayToTSparse B0 =
      make_arena_sparse_nu(w, TA::TiledRange{h, j1, j2, kU}, bzero, b_ext);
  w.gop.fence();
  reset_plan_active_count();
  for (unsigned mask : {0u, 32u}) {
    std::vector<TA::TiledRange1> Hv{h}, Mv{i1, i2}, Nv{j1, j2};
    std::vector<TA::TiledRange1> Kv{(mask & 32u) ? kT : kU};
    std::vector<std::string> base{"h", "i1", "i2", "j1", "j2"};
    std::sort(base.begin(), base.end());
    do {
      const std::string outer = permsweep_join(base, ",");
      const std::string res = outer + ";a,b";
      if (root) {
        std::fprintf(stderr, "[PERMSWEEP-SPNU] ce_e mask=%u trying %s\n", mask,
                     outer.c_str());
        std::fflush(stderr);
      }
      ArrayToTSparse C_ref =
          TA::einsum(A0("h,i1,i2,k;a"), B0("h,j1,j2,k;b"), res);
      w.gop.fence();
      ArrayToTSparse C;
      C(res) = (A0("h,i1,i2,k;a") * B0("h,j1,j2,k;b")).retile(Hv, Mv, Nv, Kv);
      w.gop.fence();
      BOOST_CHECK(C.trange() == C_ref.trange());
      BOOST_CHECK_SMALL(array_max_reldiff_sparse(C, C_ref), 1e-9);
      if (root) {
        std::fprintf(stderr, "[PERMSWEEP-SPNU] ce_e mask=%u ok %s\n", mask,
                     outer.c_str());
        std::fflush(stderr);
      }
    } while (std::next_permutation(base.begin(), base.end()));
  }
  BOOST_CHECK_GT(plan_active_count(w), std::size_t{0});
  TA::SparseShape<float>::threshold(saved);
}

BOOST_AUTO_TEST_SUITE_END()


// ===== Mixed (plain x ToT -> ToT) retile ===================================
// UNTAGGED suite: runs at np=1 (run-np-1) AND np=2 (run-np-2). Validates the
// dense-operand coarse gather (dense_gather_block) co-locates in-rank and the
// strided scale GEMM fires on coarsened BLAS axes, both orientations.
namespace {
using MixInner = TA::ArenaTensor<double, TA::Range>;
using MixToT = TA::DistArray<TA::Tensor<MixInner>, TA::SparsePolicy>;
using MixT = TA::DistArray<TA::Tensor<double>, TA::SparsePolicy>;

// all-present sparse shape (every tile nonzero)
inline TA::SparseShape<float> mix_dense_shape(TA::World& w,
                                              const TA::TiledRange& tr) {
  TA::Tensor<float> n(tr.tiles_range());
  for (std::size_t o = 0; o < tr.tiles_range().volume(); ++o) n[o] = 1.0f;
  return TA::SparseShape<float>(w, n, tr);
}
// plain operand, value = 1 + 1e-3*e over the whole tile
inline MixT mix_make_plain(TA::World& w, const TA::TiledRange& tr) {
  MixT x(w, tr, mix_dense_shape(w, tr));
  x.init_tiles([](const TA::Range& r) {
    TA::Tensor<double> t(r);
    for (std::size_t e = 0; e < t.size(); ++e) t.data()[e] = 1.0 + 1e-3 * double(e);
    return t;
  });
  return x;
}
// ToT operand, inner extent a4 fixed, then compacted to one arena page
inline MixToT mix_make_tot(TA::World& w, const TA::TiledRange& tr, long a4) {
  MixToT x(w, tr, mix_dense_shape(w, tr));
  x.init_tiles_nested([a4](const auto&) { return MixInner::range_type{a4}; },
                      [](auto& cell, const auto&) {
                        for (std::size_t e = 0; e < cell.size(); ++e)
                          cell.data()[e] = 1.0 + 1e-3 * double(e);
                      });
  w.gop.fence();
  return TA::foreach (x, [](TA::Tensor<MixInner>& r,
                            const TA::Tensor<MixInner>& a) -> float {
    r = TiledArray::detail::arena_compact(a);
    return 1.0f;
  });
}
inline std::size_t mix_scale_fires() {
  return TiledArray::detail::g_scale_strided_calls[0].load() +
         TiledArray::detail::g_scale_strided_calls[1].load();
}
// Build a comma-joined annotation from an index order.
inline std::string join(const std::vector<std::string>& v) {
  std::string s;
  for (std::size_t i = 0; i < v.size(); ++i) s += (i ? "," : "") + v[i];
  return s;
}
// plain operand with a hole: tile (0,0,0) dropped (norm below threshold).
inline MixT mix_make_plain_sparse(TA::World& w, const TA::TiledRange& tr) {
  TA::Tensor<float> nrm(tr.tiles_range());
  std::size_t o = 0;
  for (auto it = tr.tiles_range().begin(); it != tr.tiles_range().end();
       ++it, ++o) {
    const auto idx = *it;
    const bool drop = (idx[0] == 0 && idx[1] == 0 && idx[2] == 0);
    nrm[o] = drop ? (std::numeric_limits<float>::min() * 0.1f) : 1.0f;
  }
  MixT x(w, tr, TA::SparseShape<float>(w, nrm, tr));
  x.init_tiles([](const TA::Range& r) {
    TA::Tensor<double> t(r);
    for (std::size_t e = 0; e < t.size(); ++e) t.data()[e] = 1.0 + 1e-3 * double(e);
    return t;
  });
  return x;
}
}  // namespace

BOOST_AUTO_TEST_SUITE(summa_two_trange_mixed_suite)

// C(i3,i1,i2,k;a4) = g(i3,k,m) * I(m,i1,i2;a4). M=(i3,k) [plain g], K=(m)
// [shared], N=(i1,i2) [ToT I]. Coarsen M and K to single tiles (both touch the
// PLAIN operand); leave N intact. Retiled result == non-retiled einsum; plan
// active; strided scale GEMM fires AND fires fewer times than the fine path.
BOOST_AUTO_TEST_CASE(mixed_T_x_ToT_coarsen_MK) {
  auto& w = TA::get_default_world();
  const int E = 8, T = 4, A4 = 6;
  auto e = tr1(E, T);
  MixT G = mix_make_plain(w, TA::TiledRange{e, e, e});      // (i3,k,m)
  MixToT I = mix_make_tot(w, TA::TiledRange{e, e, e}, A4);  // (m,i1,i2)
  w.gop.fence();

  const auto f0 = mix_scale_fires();
  MixToT C0 = TA::einsum(G("i3,k,m"), I("m,i1,i2;a4i1i2"), "i3,i1,i2,k;a4i1i2");
  w.gop.fence();
  const auto fine_fires = mix_scale_fires() - f0;

  std::vector<TA::TiledRange1> tH, tM, tN, tK;
  tM.push_back(tr1(E, E));
  tM.push_back(tr1(E, E));  // i3, k -> single tile
  tK.push_back(tr1(E, E));  // m    -> single tile
  // tN empty => i1,i2 intact

  const auto p1 = TiledArray::detail::g_summa_plan_active_calls.load();
  const auto f1 = mix_scale_fires();
  MixToT C1;
  C1("i3,i1,i2,k;a4i1i2") =
      (G("i3,k,m") * I("m,i1,i2;a4i1i2")).retile(tH, tM, tN, tK);
  w.gop.fence();
  const auto coarse_fires = mix_scale_fires() - f1;

  BOOST_CHECK_GE(TiledArray::detail::g_summa_plan_active_calls.load() - p1, 1ul);
  BOOST_CHECK_GT(coarse_fires, 0ul);              // strided kernel fired
  BOOST_CHECK_LT(coarse_fires, fine_fires);       // and on coarser (fewer) GEMMs
  BOOST_CHECK_SMALL(array_max_reldiff2(C1, C0), 1e-10);  // element-wise, order-sensitive
}

// C(i1,i2,i3,k;a4) = I(i1,i2,m;a4) * g(m,i3,k). M=(i1,i2) [ToT I], N=(i3,k)
// [plain g], K=(m) [shared]. Coarsen K and N (both touch the PLAIN right
// operand g); leave M intact. Drives the get_row dense gather (right-operand
// path). Same invariants as the T*ToT case.
BOOST_AUTO_TEST_CASE(mixed_ToT_x_T_coarsen_NK) {
  auto& w = TA::get_default_world();
  const int E = 8, T = 4, A4 = 6;
  auto e = tr1(E, T);
  MixToT I = mix_make_tot(w, TA::TiledRange{e, e, e}, A4);  // (i1,i2,m)
  MixT G = mix_make_plain(w, TA::TiledRange{e, e, e});      // (m,i3,k)
  w.gop.fence();

  const auto f0 = mix_scale_fires();
  MixToT C0 = TA::einsum(I("i1,i2,m;a4i1i2"), G("m,i3,k"), "i1,i2,i3,k;a4i1i2");
  w.gop.fence();
  const auto fine_fires = mix_scale_fires() - f0;

  std::vector<TA::TiledRange1> tH, tM, tN, tK;
  tN.push_back(tr1(E, E));
  tN.push_back(tr1(E, E));  // i3, k (on plain g) -> single tile
  tK.push_back(tr1(E, E));  // m -> single tile
  // tM empty => i1,i2 intact

  const auto p1 = TiledArray::detail::g_summa_plan_active_calls.load();
  const auto f1 = mix_scale_fires();
  MixToT C1;
  C1("i1,i2,i3,k;a4i1i2") =
      (I("i1,i2,m;a4i1i2") * G("m,i3,k")).retile(tH, tM, tN, tK);
  w.gop.fence();
  const auto coarse_fires = mix_scale_fires() - f1;

  BOOST_CHECK_GE(TiledArray::detail::g_summa_plan_active_calls.load() - p1, 1ul);
  BOOST_CHECK_GT(coarse_fires, 0ul);              // strided kernel fired
  BOOST_CHECK_LT(coarse_fires, fine_fires);       // and on coarser (fewer) GEMMs
  BOOST_CHECK_SMALL(array_max_reldiff2(C1, C0), 1e-10);  // element-wise, order-sensitive
}

// Permutation sweep (untagged -> runs np=1 AND np=2). For the T*ToT mixed
// product, two slices whose union covers the goal:
//   (i)  all 6x6 operand-permutation pairs, fixed canonical result; and
//   (ii) all 24 result permutations, fixed canonical operands.
// In EVERY combination: the retiled result equals the non-retiled einsum
// (element-wise), the plan is active, and the strided scale GEMM fires
// (canonicalization makes every operand layout fire). Role targets are derived
// from index SETS (M={i3,k}, K={m}, N={i1,i2}) -> always 2 M-axes and 1 K-axis,
// so [full,full]/[full] is positionally correct for every permutation.
BOOST_AUTO_TEST_CASE(mixed_perm_sweep_fires_and_matches) {
  auto& w = TA::get_default_world();
  const int E = 4, T = 2, A4 = 4;  // tiny extents: 2 tiles/axis
  auto e = tr1(E, T);
  MixT G = mix_make_plain(w, TA::TiledRange{e, e, e});      // stored (i3,k,m)
  MixToT I = mix_make_tot(w, TA::TiledRange{e, e, e}, A4);  // stored (m,i1,i2)
  w.gop.fence();
  const auto full = tr1(E, E);  // single-tile (full-extent) coarsen target

  // one (g-annot, I-annot, C-annot) combination: assert fires + matches.
  std::size_t combos = 0, fired = 0;
  auto run_combo = [&](const std::string& ga, const std::string& ia,
                       const std::string& ca) {
    MixToT Cref = TA::einsum(G(ga), I(ia), ca);
    w.gop.fence();
    std::vector<TA::TiledRange1> tH, tM, tN, tK;
    tM.push_back(full); tM.push_back(full);  // i3,k
    tK.push_back(full);                       // m
    const auto f1 = mix_scale_fires();
    const auto p1 = TiledArray::detail::g_summa_plan_active_calls.load();
    MixToT C1;
    C1(ca) = (G(ga) * I(ia)).retile(tH, tM, tN, tK);
    w.gop.fence();
    const auto df = mix_scale_fires() - f1;
    ++combos;
    if (df > 0ul) ++fired;
    BOOST_CHECK_GE(TiledArray::detail::g_summa_plan_active_calls.load() - p1, 1ul);
    BOOST_CHECK_GT(df, 0ul);  // the "always triggers" invariant
    // ELEMENT-WISE, order-sensitive (a scalar checksum would miss a wrong perm).
    BOOST_CHECK_SMALL(array_max_reldiff2(C1, Cref), 1e-10);
  };

  // slice (i): all operand-perm pairs, canonical result
  std::vector<std::string> g0 = {"i3", "k", "m"}, i0 = {"i1", "i2", "m"};
  std::sort(g0.begin(), g0.end());
  do {
    auto ip = i0;
    std::sort(ip.begin(), ip.end());
    do {
      run_combo(join(g0), join(ip) + ";a4i1i2", "i3,i1,i2,k;a4i1i2");
    } while (std::next_permutation(ip.begin(), ip.end()));
  } while (std::next_permutation(g0.begin(), g0.end()));

  // slice (ii): all result perms, canonical operands
  std::vector<std::string> c0 = {"i1", "i2", "i3", "k"};
  std::sort(c0.begin(), c0.end());
  do {
    run_combo("i3,k,m", "m,i1,i2;a4i1i2", join(c0) + ";a4i1i2");
  } while (std::next_permutation(c0.begin(), c0.end()));

  BOOST_TEST_MESSAGE("mixed perm sweep: " << fired << "/" << combos
                                          << " combinations fired the kernel");
  BOOST_CHECK_EQUAL(fired, combos);
}

// Distinct per-axis targets within a role (exercises positional role->axis
// target assignment, which the uniform-full sweep above cannot). M=(i3,k):
// coarsen i3 to a single tile but leave k at its own (identity) tiling -> the
// two M-axis targets DIFFER, so a positional mix-up would assert/misnest in
// make_retile_plan/append_role_t_box. The result (delivered at U) must still
// match the einsum, plan active, kernel fired.
BOOST_AUTO_TEST_CASE(mixed_T_x_ToT_distinct_M_targets) {
  auto& w = TA::get_default_world();
  const int E = 8, T = 4, A4 = 6;
  auto e = tr1(E, T);
  MixT G = mix_make_plain(w, TA::TiledRange{e, e, e});      // (i3,k,m)
  MixToT I = mix_make_tot(w, TA::TiledRange{e, e, e}, A4);  // (m,i1,i2)
  w.gop.fence();
  MixToT C0 = TA::einsum(G("i3,k,m"), I("m,i1,i2;a4i1i2"), "i3,i1,i2,k;a4i1i2");
  w.gop.fence();

  std::vector<TA::TiledRange1> tH, tM, tN, tK;
  tM.push_back(tr1(E, E));  // i3 -> single tile (full)
  tM.push_back(tr1(E, T));  // k  -> identity (own tiling) -- DISTINCT from i3
  tK.push_back(tr1(E, E));  // m  -> single tile
  const auto p1 = TiledArray::detail::g_summa_plan_active_calls.load();
  const auto f1 = mix_scale_fires();
  MixToT C1;
  C1("i3,i1,i2,k;a4i1i2") =
      (G("i3,k,m") * I("m,i1,i2;a4i1i2")).retile(tH, tM, tN, tK);
  w.gop.fence();
  BOOST_CHECK_GE(TiledArray::detail::g_summa_plan_active_calls.load() - p1, 1ul);
  BOOST_CHECK_GT(mix_scale_fires() - f1, 0ul);
  BOOST_CHECK_SMALL(array_max_reldiff2(C1, C0), 1e-10);
}

// Sparse plain operand: a dropped g tile -> hole(zero) in the coarse pack ->
// contributes nothing. Retiled result still equals the non-retiled einsum, at
// np=1 and np=2 (the np>1 alignment uses the dense shape's block_has_nonzero).
BOOST_AUTO_TEST_CASE(mixed_T_x_ToT_sparse_plain) {
  auto& w = TA::get_default_world();
  const int E = 8, T = 4, A4 = 6;
  auto e = tr1(E, T);
  MixT G = mix_make_plain_sparse(w, TA::TiledRange{e, e, e});
  MixToT I = mix_make_tot(w, TA::TiledRange{e, e, e}, A4);
  w.gop.fence();

  MixToT C0 = TA::einsum(G("i3,k,m"), I("m,i1,i2;a4i1i2"), "i3,i1,i2,k;a4i1i2");
  w.gop.fence();

  std::vector<TA::TiledRange1> tH, tM, tN, tK;
  tM.push_back(tr1(E, E)); tM.push_back(tr1(E, E));
  tK.push_back(tr1(E, E));
  MixToT C1;
  C1("i3,i1,i2,k;a4i1i2") =
      (G("i3,k,m") * I("m,i1,i2;a4i1i2")).retile(tH, tM, tN, tK);
  w.gop.fence();
  BOOST_CHECK_SMALL(array_max_reldiff2(C1, C0), 1e-10);
}

// Regression: fully-arena ce_e (ToT*ToT->ToT) retile still coarsens K and
// still fires the ce_e strided kernel under the value_type gate. (Guards the
// gate flip in.)
BOOST_AUTO_TEST_CASE(arena_ce_e_retile_regression) {
  auto& w = TA::get_default_world();
  const int E = 8, T = 4, AN = 5;
  auto e = tr1(E, T);
  MixToT A = mix_make_tot(w, TA::TiledRange{e, e, e}, AN);  // A(i1,i3,K)
  MixToT B = mix_make_tot(w, TA::TiledRange{e, e, e}, AN);  // B(i2,i4,K)
  w.gop.fence();
  const auto c0 = TA::detail::g_strided_dgemm_ce_e_calls.load();
  MixToT C0 = TA::einsum(A("i1,i3,K;a3i1i3"), B("i2,i4,K;a4i2i4"),
                         "i1,i3,i2,i4;a3i1i3,a4i2i4");
  w.gop.fence();
  const auto fine = TA::detail::g_strided_dgemm_ce_e_calls.load() - c0;

  std::vector<TA::TiledRange1> tH, tM, tN, tK;
  tK.push_back(tr1(E, E));  // coarsen K -> 1 tile
  const auto c1 = TA::detail::g_strided_dgemm_ce_e_calls.load();
  MixToT C1;
  C1("i1,i3,i2,i4;a3i1i3,a4i2i4") =
      (A("i1,i3,K;a3i1i3") * B("i2,i4,K;a4i2i4")).retile(tH, tM, tN, tK);
  w.gop.fence();
  const auto coarse = TA::detail::g_strided_dgemm_ce_e_calls.load() - c1;
  BOOST_CHECK_GT(coarse, 0ul);
  BOOST_CHECK_LT(coarse, fine);
  BOOST_CHECK_SMALL(array_max_reldiff2(C1, C0), 1e-10);
}

// ===== Env-gated mixed auto-retile (TA_SUMMA_AUTO_RETILE) ===================
// These cases drive the PRODUCTION path: a bare TA::einsum(...) (no .retile())
// that, when the gate is on, auto-coarsens the plain operand's external + K
// inside ContEngine::maybe_make_retile_plan_. The gate is flipped in-process
// via the settable mixed_auto_retile() accessor; each case restores the prior
// value (RAII) so it leaks no state to later tests.
namespace {
// RAII guard for the in-process auto-retile gate; restores the prior value.
struct MixedGateGuard {
  bool& flag;
  bool prev;
  explicit MixedGateGuard(bool on)
      : flag(TiledArray::expressions::detail::mixed_auto_retile()),
        prev(flag) {
    flag = on;
  }
  ~MixedGateGuard() { flag = prev; }
};
}  // namespace

// Gate OFF: a bare TA::einsum(...) (no .retile()) must be a STRICT no-op --
// plan never activates (g_summa_plan_active_calls delta == 0) and the result
// matches a plain einsum reference. Byte-for-byte identical to pre-gate TA.
BOOST_AUTO_TEST_CASE(mixed_auto_retile_off_is_noop) {
  auto& w = TA::get_default_world();
  MixedGateGuard guard(false);  // gate OFF
  const int E = 8, T = 4, A4 = 6;
  auto e = tr1(E, T);
  MixT G = mix_make_plain(w, TA::TiledRange{e, e, e});      // (i3,k,m)
  MixToT I = mix_make_tot(w, TA::TiledRange{e, e, e}, A4);  // (m,i1,i2)
  w.gop.fence();

  // reference (gate off): plain einsum, no auto-retile.
  MixToT Cref = TA::einsum(G("i3,k,m"), I("m,i1,i2;a4i1i2"), "i3,i1,i2,k;a4i1i2");
  w.gop.fence();

  const auto p0 = TiledArray::detail::g_summa_plan_active_calls.load();
  MixToT Coff = TA::einsum(G("i3,k,m"), I("m,i1,i2;a4i1i2"), "i3,i1,i2,k;a4i1i2");
  w.gop.fence();

  // strict no-op: the retile plan never activates with the gate off.
  BOOST_CHECK_EQUAL(TiledArray::detail::g_summa_plan_active_calls.load() - p0,
                    0ul);
  BOOST_CHECK_SMALL(array_max_reldiff2(Coff, Cref), 1e-10);
}

// Gate ON, plain-LEFT (T*ToT): the bare einsum auto-coarsens M+K (mirrors
// mixed_T_x_ToT_coarsen_MK but with NO .retile()). Plan activates, the strided
// scale GEMM fires on coarser/fewer GEMMs than the fine path, result matches.
BOOST_AUTO_TEST_CASE(mixed_auto_retile_on_T_x_ToT) {
  auto& w = TA::get_default_world();
  const int E = 8, T = 4, A4 = 6;
  auto e = tr1(E, T);
  MixT G = mix_make_plain(w, TA::TiledRange{e, e, e});      // (i3,k,m)
  MixToT I = mix_make_tot(w, TA::TiledRange{e, e, e}, A4);  // (m,i1,i2)
  w.gop.fence();

  // fine reference + fine fire-count: gate OFF.
  MixToT Cfine;
  std::size_t fine_fires = 0;
  {
    MixedGateGuard off(false);
    const auto f0 = mix_scale_fires();
    Cfine = TA::einsum(G("i3,k,m"), I("m,i1,i2;a4i1i2"), "i3,i1,i2,k;a4i1i2");
    w.gop.fence();
    fine_fires = mix_scale_fires() - f0;
  }

  // auto path: gate ON, SAME bare einsum (no .retile()).
  MixedGateGuard on(true);
  const auto p1 = TiledArray::detail::g_summa_plan_active_calls.load();
  const auto f1 = mix_scale_fires();
  MixToT Cauto =
      TA::einsum(G("i3,k,m"), I("m,i1,i2;a4i1i2"), "i3,i1,i2,k;a4i1i2");
  w.gop.fence();
  const auto coarse_fires = mix_scale_fires() - f1;

  BOOST_CHECK_GE(TiledArray::detail::g_summa_plan_active_calls.load() - p1, 1ul);
  BOOST_CHECK_GT(coarse_fires, 0ul);          // strided kernel fired
  BOOST_CHECK_LT(coarse_fires, fine_fires);   // on coarser (fewer) GEMMs
  BOOST_CHECK_SMALL(array_max_reldiff2(Cauto, Cfine), 1e-10);
}

// Gate ON, plain-RIGHT (ToT*T): the bare einsum auto-coarsens N+K (mirrors
// mixed_ToT_x_T_coarsen_NK but with NO .retile()). Drives the right-operand
// (get_row) dense gather. Same invariants.
BOOST_AUTO_TEST_CASE(mixed_auto_retile_on_ToT_x_T) {
  auto& w = TA::get_default_world();
  const int E = 8, T = 4, A4 = 6;
  auto e = tr1(E, T);
  MixToT I = mix_make_tot(w, TA::TiledRange{e, e, e}, A4);  // (i1,i2,m)
  MixT G = mix_make_plain(w, TA::TiledRange{e, e, e});      // (m,i3,k)
  w.gop.fence();

  MixToT Cfine;
  std::size_t fine_fires = 0;
  {
    MixedGateGuard off(false);
    const auto f0 = mix_scale_fires();
    Cfine = TA::einsum(I("i1,i2,m;a4i1i2"), G("m,i3,k"), "i1,i2,i3,k;a4i1i2");
    w.gop.fence();
    fine_fires = mix_scale_fires() - f0;
  }

  MixedGateGuard on(true);
  const auto p1 = TiledArray::detail::g_summa_plan_active_calls.load();
  const auto f1 = mix_scale_fires();
  MixToT Cauto =
      TA::einsum(I("i1,i2,m;a4i1i2"), G("m,i3,k"), "i1,i2,i3,k;a4i1i2");
  w.gop.fence();
  const auto coarse_fires = mix_scale_fires() - f1;

  BOOST_CHECK_GE(TiledArray::detail::g_summa_plan_active_calls.load() - p1, 1ul);
  BOOST_CHECK_GT(coarse_fires, 0ul);
  BOOST_CHECK_LT(coarse_fires, fine_fires);
  BOOST_CHECK_SMALL(array_max_reldiff2(Cauto, Cfine), 1e-10);
}

BOOST_AUTO_TEST_SUITE_END()
