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
#include "tiledarray.h"
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
  Tensor<float> tile_norms = make_norm_tensor(tr, 1, 42);

  // Construct the shape
  BOOST_CHECK_NO_THROW(SparseShape<float> x(tile_norms, tr));
  SparseShape<float> x(tile_norms, tr);

  // Check that the shape has been initialized
  BOOST_CHECK(! x.empty());
  BOOST_CHECK(! x.is_dense());
  BOOST_CHECK(x.validate(tr.tiles()));

  size_type zero_tile_count = 0ul;

  for(Tensor<float>::size_type i = 0ul; i < tile_norms.size(); ++i) {
    // Compute the expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = tile_norms[i] / float(range.volume());
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    // Check that the tile has been normalized correctly
    BOOST_CHECK_CLOSE(x[i], expected, tolerance);

    // Check zero threshold
    if(x[i] < SparseShape<float>::threshold()) {
      BOOST_CHECK(x.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! x.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(x.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}


BOOST_AUTO_TEST_CASE( comm_constructor )
{
  // Construct test tile norms
  Tensor<float> tile_norms = make_norm_tensor(tr, 1, 98);
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

  size_type zero_tile_count = 0ul;

  for(Tensor<float>::size_type i = 0ul; i < tile_norms.size(); ++i) {
    // Compute the expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = tile_norms_ref[i] / float(range.volume());
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    // Check that the tile has been normalized correctly
    BOOST_CHECK_CLOSE(x[i], expected, tolerance);

    // Check zero threshold
    if(x[i] < SparseShape<float>::threshold()) {
      BOOST_CHECK(x.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! x.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(x.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
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

  BOOST_CHECK_EQUAL(y.sparsity(), sparse_shape.sparsity());
}

BOOST_AUTO_TEST_CASE( permute )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.perm(perm));

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    BOOST_CHECK_CLOSE(result[perm ^ tr.tiles().idx(i)], sparse_shape[i], tolerance);
  }

  BOOST_CHECK_EQUAL(result.sparsity(), sparse_shape.sparsity());
}

BOOST_AUTO_TEST_CASE( scale )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.scale(-4.1));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    float expected = sparse_shape[i] * 4.1;
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    const float result_i = result[i];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if(result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( scale_perm )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.scale(-5.4, perm));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    float expected = sparse_shape[i] * 5.4;
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    const std::size_t pi = perm_index(i);
    const float result_i = result[pi];
    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if(result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(pi));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! result.is_zero(pi));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( add )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.add(right));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    float expected = left[i] + right[i];
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    const float result_i = result[i];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if(result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( add_scale )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.add(right, -2.2f));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    float expected = (left[i] + right[i]) * 2.2f;
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    const float result_i = result[i];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if(result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( add_perm )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.add(right, perm));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    float expected = left[i] + right[i];
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    const std::size_t pi = perm_index(i);
    const float result_i = result[pi];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if(result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(pi));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! result.is_zero(pi));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( add_scale_perm )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.add(right, -2.3f, perm));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    float expected = (left[i] + right[i]) * 2.3f;
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    const std::size_t pi = perm_index(i);
    const float result_i = result[pi];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if(result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(pi));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! result.is_zero(pi));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( add_const )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.add(-8.8f));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = sparse_shape[i] + std::sqrt((8.8f * 8.8f) *
        float(range.volume())) / float(range.volume());
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    const float result_i = result[i];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if(result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( add_const_perm )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.add(-1.7, perm));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = sparse_shape[i] + std::sqrt((1.7f * 1.7f) *
        range.volume()) / range.volume();
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    const std::size_t pi = perm_index(i);
    const float result_i = result[pi];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if(result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(pi));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! result.is_zero(pi));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( subt )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.subt(right));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    float expected = left[i] + right[i];
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    const float result_i = result[i];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if(result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( subt_scale )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.subt(right, -2.2f));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    float expected = (left[i] + right[i]) * 2.2f;
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    const float result_i = result[i];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if(result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( subt_perm )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.subt(right, perm));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    float expected = left[i] + right[i];
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    const std::size_t pi = perm_index(i);
    const float result_i = result[pi];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if(result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(pi));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! result.is_zero(pi));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( subt_scale_perm )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.subt(right, -2.3f, perm));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    float expected = (left[i] + right[i]) * 2.3f;
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    const std::size_t pi = perm_index(i);
    const float result_i = result[pi];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if(result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(pi));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! result.is_zero(pi));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( subt_const )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.subt(-8.8f));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = sparse_shape[i] + std::sqrt((8.8f * 8.8f) *
        float(range.volume())) / float(range.volume());
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    BOOST_CHECK_CLOSE(result[i], expected, tolerance);

    // Check zero threshold
    if(result[i] < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( subt_const_perm )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.subt(-1.7, perm));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = sparse_shape[i] + std::sqrt((1.7f * 1.7f) *
        float(range.volume())) / float(range.volume());
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    const std::size_t pi = perm_index(i);
    const float result_i = result[pi];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if(result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(pi));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! result.is_zero(pi));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( mult )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.mult(right));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = left[i] * right[i] * float(range.volume());
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    BOOST_CHECK_CLOSE(result[i], expected, tolerance);

    // Check zero threshold
    if(result[i] < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( mult_scale )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.mult(right, -2.2f));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = (left[i] * right[i]) * 2.2f * float(range.volume());
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    BOOST_CHECK_CLOSE(result[i], expected, tolerance);

    // Check zero threshold
    if(result[i] < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( mult_perm )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.mult(right, perm));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = left[i] * right[i] * float(range.volume());
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    const std::size_t pi = perm_index(i);
    const float result_i = result[pi];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if(result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(pi));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! result.is_zero(pi));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( mult_scale_perm )
{
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.mult(right, -2.3f, perm));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for(Tensor<float>::size_type i = 0ul; i < tr.tiles().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = (left[i] * right[i]) * 2.3f * float(range.volume());
    if(expected < SparseShape<float>::threshold())
      expected = 0.0f;

    const std::size_t pi = perm_index(i);
    const float result_i = result[pi];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if(result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(pi));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! result.is_zero(pi));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( gemm )
{
  // Create a matrix with the expected output
  const std::size_t m = left.data().range().size().front();
  const std::size_t n = right.data().range().size().back();
//  const std::size_t k = left.data().size() / m;

  size_type zero_tile_count = 0ul;

  // Evaluate the contraction of sparse shapes
  math::GemmHelper gemm_helper(madness::cblas::NoTrans, madness::cblas::NoTrans,
      2u, left.data().range().dim(), right.data().range().dim());
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.gemm(right, -7.2, gemm_helper));

  // Create volumes tensors for the arguments
  Tensor<float> volumes(tr.tiles(), 0.0f);
  for(std::size_t i = 0ul; i < tr.tiles().volume(); ++i) {
    const float volume = tr.make_tile_range(i).volume();
    volumes[i] = volume;
  }

  Tensor<float> result_norms =
      left.data().mult(volumes).gemm(right.data().mult(volumes), 7.2, gemm_helper);

  // Check that the result is correct
  std::array<std::size_t, 2> i = {{ 0, 0 }};
  for(i[0] = 0ul; i[0] < m; ++i[0]) {

    const TiledRange1::range_type r_0 = tr.data()[0].tile(i[0]);
    const float size_0 = r_0.second - r_0.first;

    for(i[1] = 0ul; i[1] < n; ++i[1]) {

      const TiledRange1::range_type r_1 = tr.data()[2].tile(i[1]);
      const float size_1 = r_1.second - r_1.first;

      // Compute expected value
      float expected = result_norms[i] / (size_0 * size_1);
      if(expected < SparseShape<float>::threshold())
        expected = 0.0f;

      BOOST_CHECK_CLOSE(result[i], expected, tolerance);

      // Check zero threshold
      if(result[i] < SparseShape<float>::threshold()) {
        BOOST_CHECK(result.is_zero(i));
        ++zero_tile_count;
      } else {
        BOOST_CHECK(! result.is_zero(i));
      }
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_CASE( gemm_perm )
{
  const Permutation perm(1,0);

  // Create a matrix with the expected output
  const std::size_t m = left.data().range().size().front();
  const std::size_t n = right.data().range().size().back();
//  const std::size_t k = left.data().size() / m;

  size_type zero_tile_count = 0ul;

  // Evaluate the contraction of sparse shapes
  math::GemmHelper gemm_helper(madness::cblas::NoTrans, madness::cblas::NoTrans,
      2u, left.data().range().dim(), right.data().range().dim());
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.gemm(right, -7.2, gemm_helper, perm));

  // Create volumes tensors for the arguments
  Tensor<float> volumes(tr.tiles(), 0.0f);
  for(std::size_t i = 0ul; i < tr.tiles().volume(); ++i) {
    const float volume = tr.make_tile_range(i).volume();
    volumes[i] = volume;
  }

  Tensor<float> result_norms =
      left.data().mult(volumes).gemm(right.data().mult(volumes), 7.2, gemm_helper).permute(perm);

  // Check that the result is correct
  std::array<std::size_t, 2> i = {{ 0, 0 }};
  for(i[0] = 0ul; i[0] < n; ++i[0]) {

    const TiledRange1::range_type r_0 = tr.data()[2].tile(i[0]);
    const float size_0 = r_0.second - r_0.first;

    for(i[1] = 0ul; i[1] < m; ++i[1]) {

      const TiledRange1::range_type r_1 = tr.data()[0].tile(i[1]);
      const float size_1 = r_1.second - r_1.first;

      // Compute expected value
      float expected = result_norms[i] / (size_0 * size_1);
      if(expected < SparseShape<float>::threshold())
        expected = 0.0f;

      BOOST_CHECK_CLOSE(result[i], expected, tolerance);

      // Check zero threshold
      if(result[i] < SparseShape<float>::threshold()) {
        BOOST_CHECK(result.is_zero(i));
        ++zero_tile_count;
      } else {
        BOOST_CHECK(! result.is_zero(i));
      }
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(), float(zero_tile_count) / float(tr.tiles().volume()), tolerance);
}

BOOST_AUTO_TEST_SUITE_END()
