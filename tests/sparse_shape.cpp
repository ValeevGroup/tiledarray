/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  sparse_shape.cpp
 *  Jul 18, 2013
 *
 */

#include "TiledArray/sparse_shape.h"
#include "TiledArray/pmap/blocked_pmap.h"
#include "TiledArray/eigen.h"
#include "unit_test_config.h"
#include "sparse_shape_fixture.h"

using namespace TiledArray;

BOOST_FIXTURE_TEST_SUITE( sparse_shape_suite, SparseShapeFixture )

BOOST_AUTO_TEST_CASE( default_constructor )
{
  BOOST_CHECK_NO_THROW(SparseShape<float> x);
  SparseShape<float> x, y;
  Permutation perm;
  math::GemmHelper gemm_helper(madness::cblas::NoTrans, madness::cblas::NoTrans,
      2u, 2u, 2u);

  BOOST_CHECK(x.empty());
  BOOST_CHECK(! x.is_dense());
  BOOST_CHECK(! x.validate(tr.tiles()));

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(x[0], Exception);

  BOOST_CHECK_THROW(x.perm(perm), Exception);

  BOOST_CHECK_THROW(x.scale(2.0), Exception);
  BOOST_CHECK_THROW(x.scale(2.0, perm), Exception);

  BOOST_CHECK_THROW(x.add(y), Exception);
  BOOST_CHECK_THROW(x.add(y, 2.0), Exception);
  BOOST_CHECK_THROW(x.add(y, perm), Exception);
  BOOST_CHECK_THROW(x.add(y, 2.0, perm), Exception);
  BOOST_CHECK_THROW(x.add(2.0), Exception);
  BOOST_CHECK_THROW(x.add(2.0, perm), Exception);

  BOOST_CHECK_THROW(x.subt(y), Exception);
  BOOST_CHECK_THROW(x.subt(y, 2.0), Exception);
  BOOST_CHECK_THROW(x.subt(y, perm), Exception);
  BOOST_CHECK_THROW(x.subt(y, 2.0, perm), Exception);
  BOOST_CHECK_THROW(x.subt(2.0), Exception);
  BOOST_CHECK_THROW(x.subt(2.0, perm), Exception);

  BOOST_CHECK_THROW(x.mult(y), Exception);
  BOOST_CHECK_THROW(x.mult(y, 2.0), Exception);
  BOOST_CHECK_THROW(x.mult(y, perm), Exception);
  BOOST_CHECK_THROW(x.mult(y, 2.0, perm), Exception);

  BOOST_CHECK_THROW(x.gemm(y, 2.0, gemm_helper), Exception);
  BOOST_CHECK_THROW(x.gemm(y, 2.0, gemm_helper, perm), Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( non_comm_constructor )
{
  // Construct test tile norms
  Tensor<float> tile_norms = make_norm_tensor(tr, 42);

  // Construct the shape
  BOOST_CHECK_NO_THROW(SparseShape<float> x(tile_norms, tr));
  SparseShape<float> x(tile_norms, tr);

  // Check that the shape has been initialized
  BOOST_CHECK(! x.empty());
  BOOST_CHECK(! x.is_dense());
  BOOST_CHECK(x.validate(tr.tiles()));

  for(Tensor<float>::size_type i = 0ul; i < tile_norms.size(); ++i) {
    // Compute the expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = tile_norms[i] / float(range.volume());
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    // Check that the tile has been normalized correctly
    BOOST_CHECK_CLOSE(x[i], expected, tolerance);

    // Check zero threshold
    if(x[i] < SparseShape<float>::threshold())
      BOOST_CHECK(x.is_zero(i));
    else
      BOOST_CHECK(! x.is_zero(i));
  }

}


BOOST_AUTO_TEST_CASE( comm_constructor )
{
  // Construct test tile norms
  Tensor<float> tile_norms = make_norm_tensor(tr, 98);
  Tensor<float> tile_norms_ref = tile_norms.clone();

  // Zero non-local tiles
  TiledArray::detail::BlockedPmap pmap(*GlobalFixture::world, tr.tiles().volume());
  for(Tensor<float>::size_type i = 0ul; i < tile_norms.size(); ++i)
    if(! pmap.is_local(i))
      tile_norms[i] = 0.0f;

  // Construct the shape
  BOOST_CHECK_NO_THROW(SparseShape<float> x(*GlobalFixture::world, tile_norms, tr));
  SparseShape<float> x(*GlobalFixture::world, tile_norms, tr);

  // Check that the shape has been initialized
  BOOST_CHECK(! x.empty());
  BOOST_CHECK(! x.is_dense());
  BOOST_CHECK(x.validate(tr.tiles()));

  for(Tensor<float>::size_type i = 0ul; i < tile_norms.size(); ++i) {
    // Compute the expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = tile_norms_ref[i] / float(range.volume());
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    // Check that the tile has been normalized correctly
    BOOST_CHECK_CLOSE(x[i], expected, 0.001);

    // Check zero threshold
    if(x[i] < SparseShape<float>::threshold())
      BOOST_CHECK(x.is_zero(i));
    else
      BOOST_CHECK(! x.is_zero(i));
  }

}


BOOST_AUTO_TEST_CASE( copy_constructor )
{
  // Construct the shape
  BOOST_CHECK_NO_THROW(SparseShape<float> y(sparse_shape));
  SparseShape<float> y(sparse_shape);

  // Check that the shape has been initialized
  BOOST_CHECK(! y.empty());
  BOOST_CHECK(! y.is_dense());
  BOOST_CHECK(y.validate(tr.tiles()));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Check that the tile data has been copied correctly
    BOOST_CHECK_CLOSE(y[i], sparse_shape[i], tolerance);
  }
}

BOOST_AUTO_TEST_CASE( permute )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.perm(perm));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    BOOST_CHECK_CLOSE(result[perm ^ tr.tiles().idx(i)], sparse_shape[i], tolerance);
  }
}

BOOST_AUTO_TEST_CASE( scale )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.scale(-4.1));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const float expected = sparse_shape[i] * 4.1;

    BOOST_CHECK_CLOSE(result[i], expected, tolerance);
  }
}

BOOST_AUTO_TEST_CASE( scale_perm )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.scale(-5.4, perm));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const float expected = sparse_shape[i] * 5.4;

    BOOST_CHECK_CLOSE(result[perm ^ tr.tiles().idx(i)], expected, tolerance);
  }
}

BOOST_AUTO_TEST_CASE( add )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.add(right));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const float expected = left[i] + right[i];

    BOOST_CHECK_CLOSE(result[i], expected, tolerance);
  }
}

BOOST_AUTO_TEST_CASE( add_scale )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.add(right, -2.2f));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const float expected = (left[i] + right[i]) * 2.2f;

    BOOST_CHECK_CLOSE(result[i], expected, tolerance);
  }
}

BOOST_AUTO_TEST_CASE( add_perm )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.add(right, perm));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const float expected = left[i] + right[i];

    BOOST_CHECK_CLOSE(result[perm ^ tr.tiles().idx(i)], expected, tolerance);
  }
}

BOOST_AUTO_TEST_CASE( add_scale_perm )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.add(right, -2.3f, perm));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const float expected = (left[i] + right[i]) * 2.3f;

    BOOST_CHECK_CLOSE(result[perm ^ tr.tiles().idx(i)], expected, tolerance);
  }
}

BOOST_AUTO_TEST_CASE( add_const )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.add(-8.8f));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    const float expected = sparse_shape[i] + std::sqrt((8.8f * 8.8f) *
        float(range.volume())) / float(range.volume());

    BOOST_CHECK_CLOSE(result[i], expected, tolerance);
  }
}

BOOST_AUTO_TEST_CASE( add_const_perm )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.add(-1.7, perm));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    const float expected = sparse_shape[i] + std::sqrt((1.7f * 1.7f) *
        range.volume()) / range.volume();

    BOOST_CHECK_CLOSE(result[perm ^ tr.tiles().idx(i)], expected, tolerance);
  }
}

BOOST_AUTO_TEST_CASE( subt )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.subt(right));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const float expected = left[i] + right[i];

    BOOST_CHECK_CLOSE(result[i], expected, tolerance);
  }
}

BOOST_AUTO_TEST_CASE( subt_scale )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.subt(right, -2.2f));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const float expected = (left[i] + right[i]) * 2.2f;

    BOOST_CHECK_CLOSE(result[i], expected, tolerance);
  }
}

BOOST_AUTO_TEST_CASE( subt_perm )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.subt(right, perm));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const float expected = left[i] + right[i];

    BOOST_CHECK_CLOSE(result[perm ^ tr.tiles().idx(i)], expected, tolerance);
  }
}

BOOST_AUTO_TEST_CASE( subt_scale_perm )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.subt(right, -2.3f, perm));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const float expected = (left[i] + right[i]) * 2.3f;

    BOOST_CHECK_CLOSE(result[perm ^ tr.tiles().idx(i)], expected, tolerance);
  }
}

BOOST_AUTO_TEST_CASE( subt_const )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.subt(-8.8f));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    const float expected = sparse_shape[i] + std::sqrt((8.8f * 8.8f) *
        float(range.volume())) / float(range.volume());

    BOOST_CHECK_CLOSE(result[i], expected, tolerance);
  }
}

BOOST_AUTO_TEST_CASE( subt_const_perm )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.subt(-1.7, perm));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    const float expected = sparse_shape[i] + std::sqrt((1.7f * 1.7f) *
        float(range.volume())) / float(range.volume());

    BOOST_CHECK_CLOSE(result[perm ^ tr.tiles().idx(i)], expected, tolerance);
  }
}

BOOST_AUTO_TEST_CASE( mult )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.mult(right));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    const float expected = left[i] * right[i] * float(range.volume());

    BOOST_CHECK_CLOSE(result[i], expected, tolerance);
  }
}

BOOST_AUTO_TEST_CASE( mult_scale )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.mult(right, -2.2f));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    const float expected = (left[i] * right[i]) * 2.2f * float(range.volume());

    BOOST_CHECK_CLOSE(result[i], expected, tolerance);
  }
}

BOOST_AUTO_TEST_CASE( mult_perm )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.mult(right, perm));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    const float expected = left[i] * right[i] * float(range.volume());

    BOOST_CHECK_CLOSE(result[perm ^ tr.tiles().idx(i)], expected, tolerance);
  }
}

BOOST_AUTO_TEST_CASE( mult_scale_perm )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.mult(right, -2.3f, perm));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    const float expected = (left[i] * right[i]) * 2.3f * float(range.volume());

    BOOST_CHECK_CLOSE(result[perm ^ tr.tiles().idx(i)], expected, tolerance);
  }
}

BOOST_AUTO_TEST_CASE( gemm )
{
  // Create a matrix with the expected output
  const std::size_t m = left.data().range().size().front();
  const std::size_t n = right.data().range().size().back();
  const std::size_t k = left.data().size() / m;

  madness::ScopedArray<float> left_value(new float[m]);
  madness::ScopedArray<float> right_value(new float[n]);

  std::array<std::size_t, 2> i = {{ 0, 0 }};

  // Compute the left reduced value
  for(i[0] = 0ul; i[0] < m; ++i[0]) {
    left_value[i[0]] = 0.0f;
    for(i[1] = 0ul; i[1] < k; ++i[1]) {
      // Compute expected value
      std::size_t index = i[0] * k + i[1];
      const TiledRange::range_type range = tr.make_tile_range(index);
      const float temp = left[index] * float(range.volume());
      left_value[i[0]] +=  temp * temp;
    }

    const TiledRange1::range_type r = tr.data()[0].tile(i[0]);
    left_value[i[0]] = std::sqrt(left_value[i[0]]) / (r.second - r.first);
  }

  // Compute the right reduced value
  for(i[1] = 0ul; i[1] < n; ++i[1]) {
    right_value[i[1]] = 0.0f;

    for(i[0] = 0ul; i[0] < k; ++i[0]) {
      std::size_t index = i[0] * n + i[1];
      const TiledRange::range_type range = tr.make_tile_range(index);
      const float temp = right[index] * float(range.volume());
      right_value[i[1]] +=  temp * temp;
    }

    const TiledRange1::range_type r = tr.data()[2].tile(i[1]);
    right_value[i[1]] = std::sqrt(right_value[i[1]]) / (r.second - r.first);
  }

  // Evaluate the contraction of sparse shapes
  math::GemmHelper gemm_helper(madness::cblas::NoTrans, madness::cblas::NoTrans,
      2u, left.data().range().dim(), right.data().range().dim());
  SparseShape<float> result = left.gemm(right, -7.2, gemm_helper);

  // Check that the result is correct
  for(i[0] = 0ul; i[0] < m; ++i[0]) {
    for(i[1] = 0ul; i[1] < n; ++i[1]) {
      // Compute expected value
      float expected = left_value[i[0]] * right_value[i[1]] * 7.2f;
      if(expected < SparseShape<float>::threshold())
        expected = 0.0f;

      BOOST_CHECK_CLOSE(result[i], expected, tolerance);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
