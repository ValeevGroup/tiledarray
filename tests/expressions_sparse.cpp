/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2018  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Chong Peng
 *  Department of Chemistry, Virginia Tech
 *
 *  expressions_sparse.cpp
 *  May 4, 2018
 *
 */

#include "expressions_fixture.h"
#include "sparse_threshold_guard.h"

typedef ExpressionsFixture<TiledArray::Tensor<int>, TA::SparsePolicy>
    EF_TAspTensorI;
typedef boost::mpl::vector<EF_TAspTensorI> Fixtures;

BOOST_AUTO_TEST_SUITE(expressions_sparse_suite)
#include "expressions_impl.h"

BOOST_AUTO_TEST_SUITE(expressions_sparse_block_assign_suite)

// Regression: sub-block assignment `dest.block(lo,hi) = src` must not abort
// when the RHS (`src`) shape carries a LOWER screening threshold than the
// destination. Expr::eval_to(BlkTsrExpr) forms the result block shape via
// SparseShape::update_block, which keeps the DESTINATION's threshold, but
// writes every tile the RHS shape kept. A tile the RHS keeps yet the (stricter)
// destination threshold screens to zero must be DROPPED -- letting the
// destination shape be authoritative -- rather than forced into a shape-zero
// slot (which trips ArrayImpl::set's !is_zero assertion).
BOOST_AUTO_TEST_CASE(block_assign_threshold_mismatch) {
  using Shape = SparseShape<float>;
  auto& world = *GlobalFixture::world;
  // the global (static) sparse threshold must be changed on every rank at once
  const TA::test::ScopedSparseThreshold<float> threshold(1.0e-8f, world);

  // dest: 4x2 tiles; the assigned sub-block is tiles [2,4) x [0,2).
  TiledRange dest_tr{{0, 2, 4, 6, 8}, {0, 3, 6}};
  TiledRange blk_tr{{4, 6, 8}, {0, 3, 6}};  // sub-block, lobounds preserved

  // src (built under a LOW threshold) keeps two tiles: a small-norm one that
  // the destination's higher threshold will screen, and a large-norm one it
  // will not.
  Tensor<float> src_norms(blk_tr.tiles_range(), 0.0f);
  src_norms(0, 0) = 1.0e-3f;  // below dest threshold -> must be dropped
  src_norms(1, 1) = 1.0f;     // above dest threshold -> must be kept
  Shape src_shape(src_norms, blk_tr, /*do_not_scale=*/true);
  TSpArrayD src(world, blk_tr, src_shape);
  src.fill(1.0);
  BOOST_CHECK(!src.is_zero({0, 0}));  // src keeps the small-norm tile
  world.gop.fence();

  // dest: all-ones sparse array pre-sized under a HIGHER threshold.
  threshold.set(1.0e-2f);
  TSpArrayD dest(world, dest_tr);  // SparseShape(1, trange): every tile nonzero
  dest.fill(0.0);
  world.gop.fence();

  // The offending assignment: must not throw/abort with the fix in place.
  BOOST_CHECK_NO_THROW({
    dest("i,j").block({2, 0}, {4, 2}, preserve_lobound) = src("i,j");
    world.gop.fence();
  });

  // Destination threshold is authoritative: the small-norm tile is dropped, the
  // large-norm tile survives.
  BOOST_CHECK(dest.is_zero({2, 0}));   // 1e-3 < 1e-2 -> screened out
  BOOST_CHECK(!dest.is_zero({3, 1}));  // 1.0  >= 1e-2 -> kept

  // The dropped tile must also be zero in the shape's norms, not just below
  // its threshold: scaling by 100 would otherwise lift the leftover 1e-3 norm
  // over 1e-2, and the scaled expression would ask dest for a tile it does
  // not hold (ArrayImpl::find_local asserts !is_zero).
  BOOST_CHECK_EQUAL(dest.shape().data()(2, 0), 0.0f);
  TSpArrayD scaled;
  BOOST_CHECK_NO_THROW({
    scaled("i,j") = 100.0 * dest("i,j");
    world.gop.fence();
  });
  BOOST_CHECK(scaled.is_zero({2, 0}));
  BOOST_CHECK(!scaled.is_zero({3, 1}));
}

// Regression for threshold 0: a shape screened at threshold 0 cannot mark a
// zero-norm tile zero (0 < 0 is false), yet nothing produces such a tile. The
// tests below mix operands screened at a positive threshold with results made
// at threshold 0; each must neither abort nor hang. Where the shape can be
// checked before the read that would hang, a BOOST_REQUIRE on it makes a
// regression fail by name instead of wedging the suite; the contraction cannot
// be checked that way, and a regression there aborts in a task (ContractReduce
// asserts its reduction is nonempty), taking down the whole test binary.

// contraction: every tile pair feeding c(0,0) has a zero operand tile, so its
// reduction would receive no pairs
BOOST_AUTO_TEST_CASE(contraction_zero_threshold_unfed_tile) {
  using Shape = SparseShape<float>;
  auto& world = *GlobalFixture::world;
  const TA::test::ScopedSparseThreshold<float> threshold(1.0e-8f, world);

  // c(0,0) = a(0,0) b(0,0) + a(0,1) b(1,0); each pair has one zero tile
  TiledRange a_tr{{0, 2}, {0, 2, 4}};
  TiledRange b_tr{{0, 2, 4}, {0, 2}};
  Tensor<float> a_norms(a_tr.tiles_range(), 0.0f);
  Tensor<float> b_norms(b_tr.tiles_range(), 0.0f);
  a_norms(0, 0) = 1.0f;
  b_norms(1, 0) = 1.0f;
  TSpArrayD a(world, a_tr, Shape(a_norms, a_tr));
  TSpArrayD b(world, b_tr, Shape(b_norms, b_tr));
  a.fill(1.0);
  b.fill(1.0);
  world.gop.fence();

  threshold.set(0.0f);
  TSpArrayD c;
  BOOST_CHECK_NO_THROW({
    c("i,j") = a("i,k") * b("k,j");
    world.gop.fence();
  });
  BOOST_CHECK(c.is_zero(0));
}

// block assignment: src's zero tile is never written into dest
BOOST_AUTO_TEST_CASE(block_assign_zero_threshold_unwritten_tile) {
  using Shape = SparseShape<float>;
  auto& world = *GlobalFixture::world;
  const TA::test::ScopedSparseThreshold<float> threshold(1.0e-8f, world);

  TiledRange src_tr{{2, 4}, {0, 2, 4}};  // dest's tile row 1, lobound preserved
  TiledRange dest_tr{{0, 2, 4}, {0, 2, 4}};
  Tensor<float> src_norms(src_tr.tiles_range(), 0.0f);
  src_norms(0, 1) = 1.0f;  // src(0,0) zero
  TSpArrayD src(world, src_tr, Shape(src_norms, src_tr));
  src.fill(1.0);
  world.gop.fence();

  threshold.set(0.0f);
  TSpArrayD dest(world, dest_tr);  // every tile nonzero
  dest.fill(1.0);
  world.gop.fence();
  dest("i,j").block({1, 0}, {2, 2}, preserve_lobound) = src("i,j");
  world.gop.fence();

  // dest(1,0) is never written; if its shape says nonzero, the copy hangs
  BOOST_REQUIRE(dest.is_zero({1, 0}));
  TSpArrayD copy;
  BOOST_CHECK_NO_THROW({
    copy("i,j") = dest("i,j");
    world.gop.fence();
  });
}

// fill(0): a sparse array filled with zero holds no tiles
BOOST_AUTO_TEST_CASE(fill_zero_at_zero_threshold) {
  auto& world = *GlobalFixture::world;
  const TA::test::ScopedSparseThreshold<float> threshold(0.0f, world);

  TSpArrayD a(world, TiledRange{{0, 2, 4}, {0, 2, 4}});
  a.fill(0.0);
  world.gop.fence();

  // no tile is written; if the shape says nonzero, the copy hangs
  for (const auto& idx : a.trange().tiles_range())
    BOOST_REQUIRE(a.is_zero(idx));
  TSpArrayD copy;
  BOOST_CHECK_NO_THROW({
    copy("i,j") = a("i,j");
    world.gop.fence();
  });
}

BOOST_AUTO_TEST_SUITE_END()
