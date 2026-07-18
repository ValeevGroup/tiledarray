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
  const float saved_threshold = Shape::threshold();

  // The global (static) sparse threshold must be changed consistently on all
  // ranks; use the documented gop.serial_invoke idiom (see
  // SparseShape::threshold and conversions/truncate.h) rather than a bare
  // setter call.
  auto set_threshold = [&world](float t) {
    world.gop.serial_invoke([t] { Shape::threshold(t); });
  };

  // dest: 4x2 tiles; the assigned sub-block is tiles [2,4) x [0,2).
  TiledRange dest_tr{{0, 2, 4, 6, 8}, {0, 3, 6}};
  TiledRange blk_tr{{4, 6, 8}, {0, 3, 6}};  // sub-block, lobounds preserved

  // src (built under a LOW threshold) keeps two tiles: a small-norm one that
  // the destination's higher threshold will screen, and a large-norm one it
  // will not.
  set_threshold(1.0e-8f);
  Tensor<float> src_norms(blk_tr.tiles_range(), 0.0f);
  src_norms(0, 0) = 1.0e-3f;  // below dest threshold -> must be dropped
  src_norms(1, 1) = 1.0f;     // above dest threshold -> must be kept
  Shape src_shape(src_norms, blk_tr, /*do_not_scale=*/true);
  TSpArrayD src(world, blk_tr, src_shape);
  src.fill(1.0);
  BOOST_CHECK(!src.is_zero({0, 0}));  // src keeps the small-norm tile
  world.gop.fence();

  // dest: all-ones sparse array pre-sized under a HIGHER threshold.
  set_threshold(1.0e-2f);
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

  set_threshold(saved_threshold);  // restore global (static) threshold
}

BOOST_AUTO_TEST_SUITE_END()
