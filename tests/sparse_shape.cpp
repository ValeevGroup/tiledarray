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

#include <boost/range/combine.hpp>
#ifdef TILEDARRAY_HAS_RANGEV3
#include <range/v3/view/zip.hpp>
#endif

#include "TiledArray/sparse_shape.h"
#include "sparse_shape_fixture.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;

BOOST_FIXTURE_TEST_SUITE(sparse_shape_suite, SparseShapeFixture)

BOOST_AUTO_TEST_CASE(default_constructor) {
  BOOST_CHECK_NO_THROW(SparseShape<float> x);
  SparseShape<float> x, y;
  Permutation perm;
  math::GemmHelper gemm_helper(TiledArray::math::blas::Op::NoTrans,
                               TiledArray::math::blas::Op::NoTrans, 2u, 2u, 2u);

  BOOST_CHECK(x.empty());
  BOOST_CHECK(!x.is_dense());
  BOOST_CHECK(!x.validate(tr.tiles_range()));
  BOOST_CHECK_EQUAL(x.init_threshold(), SparseShape<float>::threshold());

  BOOST_CHECK_TA_ASSERT(x.nnz(), Exception);

  BOOST_CHECK_TA_ASSERT(x[0], Exception);

  BOOST_CHECK_TA_ASSERT(x.perm(perm), Exception);

  BOOST_CHECK_TA_ASSERT(x.scale(2.0), Exception);
  BOOST_CHECK_TA_ASSERT(x.scale(2.0, perm), Exception);

  BOOST_CHECK_TA_ASSERT(x.add(y), Exception);
  BOOST_CHECK_TA_ASSERT(x.add(y, 2.0), Exception);
  BOOST_CHECK_TA_ASSERT(x.add(y, perm), Exception);
  BOOST_CHECK_TA_ASSERT(x.add(y, 2.0, perm), Exception);
  BOOST_CHECK_TA_ASSERT(x.add(2.0), Exception);
  BOOST_CHECK_TA_ASSERT(x.add(2.0, perm), Exception);

  BOOST_CHECK_TA_ASSERT(x.subt(y), Exception);
  BOOST_CHECK_TA_ASSERT(x.subt(y, 2.0), Exception);
  BOOST_CHECK_TA_ASSERT(x.subt(y, perm), Exception);
  BOOST_CHECK_TA_ASSERT(x.subt(y, 2.0, perm), Exception);
  BOOST_CHECK_TA_ASSERT(x.subt(2.0), Exception);
  BOOST_CHECK_TA_ASSERT(x.subt(2.0, perm), Exception);

  BOOST_CHECK_TA_ASSERT(x.mult(y), Exception);
  BOOST_CHECK_TA_ASSERT(x.mult(y, 2.0), Exception);
  BOOST_CHECK_TA_ASSERT(x.mult(y, perm), Exception);
  BOOST_CHECK_TA_ASSERT(x.mult(y, 2.0, perm), Exception);

  BOOST_CHECK_TA_ASSERT(x.gemm(y, 2.0, gemm_helper), Exception);
  BOOST_CHECK_TA_ASSERT(x.gemm(y, 2.0, gemm_helper, perm), Exception);
}

BOOST_AUTO_TEST_CASE(non_comm_constructor) {
  // Construct test tile norms
  Tensor<float> tile_norms = make_norm_tensor(tr, 1, 42);

  // Construct the shape using dense ctor
  BOOST_CHECK_NO_THROW(SparseShape<float> x(tile_norms, tr));
  SparseShape<float> x(tile_norms, tr);

  // Check that the shape has been initialized
  BOOST_CHECK(!x.empty());
  BOOST_CHECK(!x.is_dense());
  BOOST_CHECK(x.validate(tr.tiles_range()));
  BOOST_CHECK_EQUAL(x.init_threshold(), SparseShape<float>::threshold());

  size_type zero_tile_count = 0ul;

  for (Tensor<float>::size_type i = 0ul; i < tile_norms.size(); ++i) {
    // Compute the expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = tile_norms[i] / float(range.volume());
    if (expected < SparseShape<float>::threshold()) expected = 0.0f;

    // Check that the tile norm has been scaled correctly
    BOOST_CHECK_CLOSE(x[i], expected, tolerance);

    // Check that the unscaled tile norm is correct
    BOOST_CHECK_CLOSE(x.tile_norms()[i], tile_norms[i], tolerance);

    // Check zero threshold
    if (x[i] < SparseShape<float>::threshold()) {
      BOOST_CHECK(x.is_zero(i));
      // "zero" tile norms are set to hard 0
      BOOST_CHECK(x[i] == 0.0f);
      BOOST_CHECK(x.tile_norms()[i] == 0.0f);
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!x.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(x.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
  BOOST_CHECK(x.nnz() == x.data().size() - zero_tile_count);

  // use the sparse ctor
  {
    std::vector<std::pair<Range::index, float>> sparse_tile_norms;

    for (Tensor<float>::size_type i = 0ul; i < tile_norms.size(); ++i) {
      auto tiles_range = tr.tiles_range();
      auto idx = tiles_range.idx(i);
      if (tile_norms[i] > 0.0) {
        sparse_tile_norms.push_back(std::make_pair(idx, tile_norms[i]));
      }
    }

    // Construct the shape using sparse ctor
    BOOST_CHECK_NO_THROW(SparseShape<float> x_sp(sparse_tile_norms, tr));
    SparseShape<float> x_sp(sparse_tile_norms, tr);

    // Check that the dense and sparse ctors produced same data
    for (Tensor<float>::size_type i = 0ul; i < tile_norms.size(); ++i) {
      BOOST_CHECK_CLOSE(x[i], x_sp[i], tolerance);
      BOOST_CHECK_CLOSE(x.tile_norms()[i], x_sp.tile_norms()[i], tolerance);
    }

    BOOST_CHECK_EQUAL(x_sp.nnz(), x_sp.data().size() - zero_tile_count);
  }
}

BOOST_AUTO_TEST_CASE(comm_constructor) {
  // Construct test tile norms
  Tensor<float> tile_norms = make_norm_tensor(tr, 1, 98);
  Tensor<float> tile_norms_ref = tile_norms.clone();

  // Zero non-local tiles
  TiledArray::detail::BlockedPmap pmap(*GlobalFixture::world,
                                       tr.tiles_range().volume());
  for (Tensor<float>::size_type i = 0ul; i < tile_norms.size(); ++i)
    if (!pmap.is_local(i)) tile_norms[i] = 0.0f;

  // Construct the shape
  BOOST_CHECK_NO_THROW(
      SparseShape<float> x(*GlobalFixture::world, tile_norms, tr));
  SparseShape<float> x(*GlobalFixture::world, tile_norms, tr);

  // Check that the shape has been initialized
  BOOST_CHECK(!x.empty());
  BOOST_CHECK(!x.is_dense());
  BOOST_CHECK(x.validate(tr.tiles_range()));
  BOOST_CHECK_EQUAL(x.init_threshold(), SparseShape<float>::threshold());

  size_type zero_tile_count = 0ul;

  for (Tensor<float>::size_type i = 0ul; i < tile_norms.size(); ++i) {
    // Compute the expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = tile_norms_ref[i] / float(range.volume());
    if (expected < SparseShape<float>::threshold()) expected = 0.0f;

    // Check that the tile has been normalized correctly
    BOOST_CHECK_CLOSE(x[i], expected, tolerance);

    // Check zero threshold
    if (x[i] < SparseShape<float>::threshold()) {
      BOOST_CHECK(x.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!x.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(x.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
  BOOST_CHECK_EQUAL(x.nnz(), x.data().size() - zero_tile_count);

  // use the sparse ctor
  {
    std::vector<std::pair<Range::index, float>> sparse_tile_norms;

    for (Tensor<float>::size_type i = 0ul; i < tile_norms.size(); ++i) {
      auto tiles_range = tr.tiles_range();
      auto idx = tiles_range.idx(i);
      if (tile_norms[i] > 0.0) {
        sparse_tile_norms.push_back(std::make_pair(idx, tile_norms[i]));
      }
    }

    // Construct the shape using sparse ctor
    BOOST_CHECK_NO_THROW(
        SparseShape<float> x_sp(*GlobalFixture::world, sparse_tile_norms, tr));
    SparseShape<float> x_sp(*GlobalFixture::world, sparse_tile_norms, tr);

    // Check that the dense and sparse ctors produced same data
    for (Tensor<float>::size_type i = 0ul; i < tile_norms.size(); ++i) {
      BOOST_CHECK_CLOSE(x[i], x_sp[i], tolerance);
    }
    BOOST_CHECK_EQUAL(x_sp.nnz(), x.nnz());
  }
}

BOOST_AUTO_TEST_CASE(copy_constructor) {
  // change default threshold to make sure it's not inherited
  auto resetter = set_threshold_to_max();

  // Construct the shape
  BOOST_CHECK_NO_THROW(SparseShape<float> y(sparse_shape));
  SparseShape<float> y(sparse_shape);

  // Check that the shape has been initialized
  BOOST_CHECK(!y.empty());
  BOOST_CHECK(!y.is_dense());
  BOOST_CHECK(y.validate(tr.tiles_range()));

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Check that the tile data has been copied correctly
    BOOST_CHECK_CLOSE(y[i], sparse_shape[i], tolerance);
  }

  BOOST_CHECK_EQUAL(y.sparsity(), sparse_shape.sparsity());
  BOOST_CHECK_EQUAL(y.init_threshold(), sparse_shape.init_threshold());
  BOOST_CHECK_EQUAL(y.nnz(), sparse_shape.nnz());
}

BOOST_AUTO_TEST_CASE(permute) {
  // change default threshold to make sure it's not inherited
  auto resetter = set_threshold_to_max();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.perm(perm));

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    BOOST_CHECK_CLOSE(result[perm * tr.tiles_range().idx(i)], sparse_shape[i],
                      tolerance);
  }

  BOOST_CHECK_EQUAL(result.sparsity(), sparse_shape.sparsity());
  BOOST_CHECK_EQUAL(result.init_threshold(), sparse_shape.init_threshold());
  BOOST_CHECK_EQUAL(result.nnz(), sparse_shape.nnz());
}

BOOST_AUTO_TEST_CASE(block) {
  // change default threshold to make sure it's not inherited
  auto resetter = set_threshold_to_max();

  auto less = std::less<std::size_t>();

  for (auto lower_it = tr.tiles_range().begin();
       lower_it != tr.tiles_range().end(); ++lower_it) {
    const auto& lower = *lower_it;

    for (auto upper_it = tr.tiles_range().begin();
         upper_it != tr.tiles_range().end(); ++upper_it) {
      auto upper = *upper_it;
      for (auto it = upper.begin(); it != upper.end(); ++it) *it += 1;

      if (std::equal(lower.begin(), lower.end(), upper.begin(), less)) {
        // Check that the block function does not throw an exception
        SparseShape<float> result;
        BOOST_REQUIRE_NO_THROW(result = sparse_shape.block(lower, upper));

        BOOST_CHECK_EQUAL(result.init_threshold(),
                          sparse_shape.init_threshold());

        // Check that the block range data is correct
        std::size_t volume = 1ul;
        for (int i = int(tr.tiles_range().rank()) - 1u; i >= 0; --i) {
          auto size_i = upper[i] - lower[i];
          BOOST_CHECK_EQUAL(result.data().range().lobound(i), 0);
          BOOST_CHECK_EQUAL(result.data().range().upbound(i), size_i);
          BOOST_CHECK_EQUAL(result.data().range().extent(i), size_i);
          BOOST_CHECK_EQUAL(result.data().range().stride(i), volume);
          volume *= size_i;
        }
        BOOST_CHECK_EQUAL(result.data().range().volume(), volume);

        // Check that the data was copied and scaled correctly
        unsigned long i = 0ul;
        unsigned long zero_tile_count = 0ul;
        Range::index arg_index(sparse_shape.data().range().rank(), 0ul);
        for (auto it = result.data().range().begin();
             it != result.data().range().end(); ++it, ++i) {
          // Construct the coordinate index for the argument element
          for (unsigned int j = 0u; j < sparse_shape.data().range().rank(); ++j)
            arg_index[j] = (*it)[j] + lower[j];

          // Check the result elements
          BOOST_CHECK_CLOSE(result.data()(*it), sparse_shape.data()(arg_index),
                            tolerance);
          BOOST_CHECK_CLOSE(result.data()[i], sparse_shape.data()(arg_index),
                            tolerance);
          if (result.data()[i] < sparse_shape.init_threshold())
            ++zero_tile_count;
        }
        BOOST_CHECK_CLOSE(
            result.sparsity(),
            float(zero_tile_count) / float(result.data().range().volume()),
            tolerance);

        // validate other block functions
#if TEST_DIM == 3u
        BOOST_REQUIRE_NO_THROW(sparse_shape.block(
            {lower[0], lower[1], lower[2]}, {upper[0], upper[1], upper[2]}));
        auto result1 = sparse_shape.block({lower[0], lower[1], lower[2]},
                                          {upper[0], upper[1], upper[2]});
        BOOST_CHECK_EQUAL(result, result1);
        BOOST_REQUIRE_NO_THROW(sparse_shape.block({{lower[0], upper[0]},
                                                   {lower[1], upper[1]},
                                                   {lower[2], upper[2]}}));
        auto result2 = sparse_shape.block(
            {{lower[0], upper[0]}, {lower[1], upper[1]}, {lower[2], upper[2]}});
        BOOST_CHECK_EQUAL(result, result2);
#endif
        BOOST_REQUIRE_NO_THROW(
            sparse_shape.block(boost::combine(lower, upper)));
        auto result3 = sparse_shape.block(boost::combine(lower, upper));
        BOOST_CHECK_EQUAL(result, result3);
#ifdef TILEDARRAY_HAS_RANGEV3
        BOOST_REQUIRE_NO_THROW(
            sparse_shape.block(ranges::views::zip(lower, upper)));
        auto result4 = sparse_shape.block(ranges::views::zip(lower, upper));
        BOOST_CHECK_EQUAL(result, result4);
#endif
      } else {
        // Check that block throws an exception with a bad block range
        BOOST_CHECK_TA_ASSERT(sparse_shape.block(lower, upper),
                              TiledArray::Exception);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(block_scale) {
  // change default threshold to make sure it's not inherited
  auto resetter = set_threshold_to_max();

  auto less = std::less<std::size_t>();
  const float factor = 3.3;

  for (auto lower_it = tr.tiles_range().begin();
       lower_it != tr.tiles_range().end(); ++lower_it) {
    const auto& lower = *lower_it;

    for (auto upper_it = tr.tiles_range().begin();
         upper_it != tr.tiles_range().end(); ++upper_it) {
      auto upper = *upper_it;
      for (auto it = upper.begin(); it != upper.end(); ++it) *it += 1;

      if (std::equal(lower.begin(), lower.end(), upper.begin(), less)) {
        // Check that the block function does not throw an exception
        SparseShape<float> result;
        BOOST_REQUIRE_NO_THROW(result =
                                   sparse_shape.block(lower, upper, factor));
        BOOST_CHECK_EQUAL(result.init_threshold(),
                          sparse_shape.init_threshold());

        // Check that the block range data is correct
        std::size_t volume = 1ul;
        for (int i = int(tr.tiles_range().rank()) - 1u; i >= 0; --i) {
          auto size_i = upper[i] - lower[i];
          BOOST_CHECK_EQUAL(result.data().range().lobound(i), 0);
          BOOST_CHECK_EQUAL(result.data().range().upbound(i), size_i);
          BOOST_CHECK_EQUAL(result.data().range().extent(i), size_i);
          BOOST_CHECK_EQUAL(result.data().range().stride(i), volume);
          volume *= size_i;
        }
        BOOST_CHECK_EQUAL(result.data().range().volume(), volume);

        unsigned long i = 0ul;
        unsigned long zero_tile_count = 0ul;
        Range::index arg_index(sparse_shape.data().range().rank(), 0ul);
        for (auto it = result.data().range().begin();
             it != result.data().range().end(); ++it, ++i) {
          // Construct the coordinate index for the argument element
          for (unsigned int j = 0u; j < sparse_shape.data().range().rank(); ++j)
            arg_index[j] = (*it)[j] + lower[j];

          // Compute the expected value
          const auto expected = sparse_shape.data()(arg_index) * factor;

          // Check the result elements
          BOOST_CHECK_CLOSE(result.data()(*it), expected, tolerance);
          BOOST_CHECK_CLOSE(result.data()[i], expected, tolerance);
          if (result.data()[i] < sparse_shape.init_threshold())
            ++zero_tile_count;
        }
        BOOST_CHECK_CLOSE(
            result.sparsity(),
            float(zero_tile_count) / float(result.data().range().volume()),
            tolerance);

        // validate other block functions
#if TEST_DIM == 3u
        BOOST_REQUIRE_NO_THROW(
            sparse_shape.block({lower[0], lower[1], lower[2]},
                               {upper[0], upper[1], upper[2]}, factor));
        auto result1 =
            sparse_shape.block({lower[0], lower[1], lower[2]},
                               {upper[0], upper[1], upper[2]}, factor);
        BOOST_CHECK_EQUAL(result, result1);
        BOOST_REQUIRE_NO_THROW(sparse_shape.block(
            {{lower[0], upper[0]}, {lower[1], upper[1]}, {lower[2], upper[2]}},
            factor));
        auto result2 = sparse_shape.block(
            {{lower[0], upper[0]}, {lower[1], upper[1]}, {lower[2], upper[2]}},
            factor);
        BOOST_CHECK_EQUAL(result, result2);
#endif
        BOOST_REQUIRE_NO_THROW(
            sparse_shape.block(boost::combine(lower, upper), factor));
        auto result3 = sparse_shape.block(boost::combine(lower, upper), factor);
        BOOST_CHECK_EQUAL(result, result3);
#ifdef TILEDARRAY_HAS_RANGEV3
        BOOST_REQUIRE_NO_THROW(
            sparse_shape.block(ranges::views::zip(lower, upper), factor));
        auto result4 =
            sparse_shape.block(ranges::views::zip(lower, upper), factor);
        BOOST_CHECK_EQUAL(result, result4);
#endif

      } else {
        // Check that block throws an exception with a bad block range
        BOOST_CHECK_TA_ASSERT(sparse_shape.block(lower, upper),
                              TiledArray::Exception);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(block_perm) {
  // change default threshold to make sure it's not inherited
  auto resetter = set_threshold_to_max();

  auto less = std::less<std::size_t>();
  const auto inv_perm = perm.inv();

  for (auto lower_it = tr.tiles_range().begin();
       lower_it != tr.tiles_range().end(); ++lower_it) {
    const auto& lower = *lower_it;

    for (auto upper_it = tr.tiles_range().begin();
         upper_it != tr.tiles_range().end(); ++upper_it) {
      auto upper = *upper_it;
      for (auto it = upper.begin(); it != upper.end(); ++it) *it += 1;

      if (std::equal(lower.begin(), lower.end(), upper.begin(), less)) {
        // Check that the block function does not throw an exception
        SparseShape<float> result;
        BOOST_REQUIRE_NO_THROW(result = sparse_shape.block(lower, upper, perm));
        BOOST_CHECK_EQUAL(result.init_threshold(),
                          sparse_shape.init_threshold());

        // Check that the block range data is correct
        std::size_t volume = 1ul;
        for (int i = int(tr.tiles_range().rank()) - 1u; i >= 0; --i) {
          const auto inv_perm_i = inv_perm[i];
          const auto size_i = upper[inv_perm_i] - lower[inv_perm_i];
          BOOST_CHECK_EQUAL(result.data().range().lobound(i), 0);
          BOOST_CHECK_EQUAL(result.data().range().upbound(i), size_i);
          BOOST_CHECK_EQUAL(result.data().range().extent(i), size_i);
          BOOST_CHECK_EQUAL(result.data().range().stride(i), volume);
          volume *= size_i;
        }
        BOOST_CHECK_EQUAL(result.data().range().volume(), volume);

        // Check that the data was copied and scaled correctly
        unsigned long i = 0ul;
        unsigned long zero_tile_count = 0ul;
        Range::index arg_index(sparse_shape.data().range().rank(), 0ul);
        for (auto it = result.data().range().begin();
             it != result.data().range().end(); ++it, ++i) {
          // Construct the coordinate index for the argument element
          for (unsigned int j = 0u; j < sparse_shape.data().range().rank();
               ++j) {
            const auto perm_i = perm[j];
            arg_index[j] = (*it)[perm_i] + lower[j];
          }

          // Check the result elements
          BOOST_CHECK_CLOSE(result.data()(*it), sparse_shape.data()(arg_index),
                            tolerance);
          BOOST_CHECK_CLOSE(result.data()[i], sparse_shape.data()(arg_index),
                            tolerance);
          if (result.data()[i] < sparse_shape.init_threshold())
            ++zero_tile_count;
        }
        BOOST_CHECK_CLOSE(
            result.sparsity(),
            float(zero_tile_count) / float(result.data().range().volume()),
            tolerance);

        // validate other block functions
#if TEST_DIM == 3u
        BOOST_REQUIRE_NO_THROW(
            sparse_shape.block({lower[0], lower[1], lower[2]},
                               {upper[0], upper[1], upper[2]}, perm));
        auto result1 = sparse_shape.block({lower[0], lower[1], lower[2]},
                                          {upper[0], upper[1], upper[2]}, perm);
        BOOST_CHECK_EQUAL(result, result1);
        BOOST_REQUIRE_NO_THROW(sparse_shape.block(
            {{lower[0], upper[0]}, {lower[1], upper[1]}, {lower[2], upper[2]}},
            perm));
        auto result2 = sparse_shape.block(
            {{lower[0], upper[0]}, {lower[1], upper[1]}, {lower[2], upper[2]}},
            perm);
        BOOST_CHECK_EQUAL(result, result2);
#endif
        BOOST_REQUIRE_NO_THROW(
            sparse_shape.block(boost::combine(lower, upper), perm));
        auto result3 = sparse_shape.block(boost::combine(lower, upper), perm);
        BOOST_CHECK_EQUAL(result, result3);
#ifdef TILEDARRAY_HAS_RANGEV3
        BOOST_REQUIRE_NO_THROW(
            sparse_shape.block(ranges::views::zip(lower, upper), perm));
        auto result4 =
            sparse_shape.block(ranges::views::zip(lower, upper), perm);
        BOOST_CHECK_EQUAL(result, result4);
#endif

      } else {
        // Check that block throws an exception with a bad block range
        BOOST_CHECK_TA_ASSERT(sparse_shape.block(lower, upper),
                              TiledArray::Exception);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(block_scale_perm) {
  // change default threshold to make sure it's not inherited
  auto resetter = set_threshold_to_max();

  auto less = std::less<std::size_t>();
  const float factor = 3.3;
  const auto inv_perm = perm.inv();

  for (auto lower_it = tr.tiles_range().begin();
       lower_it != tr.tiles_range().end(); ++lower_it) {
    const auto& lower = *lower_it;

    for (auto upper_it = tr.tiles_range().begin();
         upper_it != tr.tiles_range().end(); ++upper_it) {
      auto upper = *upper_it;
      for (auto it = upper.begin(); it != upper.end(); ++it) *it += 1;

      if (std::equal(lower.begin(), lower.end(), upper.begin(), less)) {
        // Check that the block function does not throw an exception
        SparseShape<float> result;
        BOOST_REQUIRE_NO_THROW(
            result = sparse_shape.block(lower, upper, factor, perm));
        BOOST_CHECK_EQUAL(result.init_threshold(),
                          sparse_shape.init_threshold());

        // Check that the block range data is correct
        std::size_t volume = 1ul;
        for (int i = int(tr.tiles_range().rank()) - 1u; i >= 0; --i) {
          const auto inv_perm_i = inv_perm[i];
          const auto size_i = upper[inv_perm_i] - lower[inv_perm_i];
          BOOST_CHECK_EQUAL(result.data().range().lobound(i), 0);
          BOOST_CHECK_EQUAL(result.data().range().upbound(i), size_i);
          BOOST_CHECK_EQUAL(result.data().range().extent(i), size_i);
          BOOST_CHECK_EQUAL(result.data().range().stride(i), volume);
          volume *= size_i;
        }
        BOOST_CHECK_EQUAL(result.data().range().volume(), volume);

        unsigned long i = 0ul;
        unsigned long zero_tile_count = 0ul;
        Range::index arg_index(sparse_shape.data().range().rank(), 0ul);
        for (auto it = result.data().range().begin();
             it != result.data().range().end(); ++it, ++i) {
          // Construct the coordinate index for the argument element
          for (unsigned int j = 0u; j < sparse_shape.data().range().rank();
               ++j) {
            const auto perm_i = perm[j];
            arg_index[j] = (*it)[perm_i] + lower[j];
          }

          // Compute the expected value
          const auto expected = sparse_shape.data()(arg_index) * factor;

          // Check the result elements
          BOOST_CHECK_CLOSE(result.data()(*it), expected, tolerance);
          BOOST_CHECK_CLOSE(result.data()[i], expected, tolerance);
          if (result.data()[i] < sparse_shape.init_threshold())
            ++zero_tile_count;
        }
        BOOST_CHECK_CLOSE(
            result.sparsity(),
            float(zero_tile_count) / float(result.data().range().volume()),
            tolerance);

        // validate other block functions
#if TEST_DIM == 3u
        BOOST_REQUIRE_NO_THROW(
            sparse_shape.block({lower[0], lower[1], lower[2]},
                               {upper[0], upper[1], upper[2]}, factor, perm));
        auto result1 =
            sparse_shape.block({lower[0], lower[1], lower[2]},
                               {upper[0], upper[1], upper[2]}, factor, perm);
        BOOST_CHECK_EQUAL(result, result1);
        BOOST_REQUIRE_NO_THROW(sparse_shape.block(
            {{lower[0], upper[0]}, {lower[1], upper[1]}, {lower[2], upper[2]}},
            factor, perm));
        auto result2 = sparse_shape.block(
            {{lower[0], upper[0]}, {lower[1], upper[1]}, {lower[2], upper[2]}},
            factor, perm);
        BOOST_CHECK_EQUAL(result, result2);
#endif
        BOOST_REQUIRE_NO_THROW(
            sparse_shape.block(boost::combine(lower, upper), factor, perm));
        auto result3 =
            sparse_shape.block(boost::combine(lower, upper), factor, perm);
        BOOST_CHECK_EQUAL(result, result3);
#ifdef TILEDARRAY_HAS_RANGEV3
        BOOST_REQUIRE_NO_THROW(
            sparse_shape.block(ranges::views::zip(lower, upper), factor, perm));
        auto result4 =
            sparse_shape.block(ranges::views::zip(lower, upper), factor, perm);
        BOOST_CHECK_EQUAL(result, result4);
#endif

      } else {
        // Check that block throws an exception with a bad block range
        BOOST_CHECK_TA_ASSERT(sparse_shape.block(lower, upper),
                              TiledArray::Exception);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE(transform) {
  // change default threshold to make sure it's not inherited
  auto resetter = set_threshold_to_max();

  auto op = [](Tensor<float> const& t) {
    Tensor<float> new_t = t.clone();

    const auto size = new_t.range().volume();
    for (auto i = 0ul; i < size; ++i) {
      if (i % 2 == 0) {
        new_t[i] *= 2;
      } else {
        new_t[i] /= 2;
      }
    }

    return new_t;
  };

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.transform(op));
  BOOST_CHECK_EQUAL(result.init_threshold(), sparse_shape.init_threshold());

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value

    float expected;
    if (i % 2 == 0) {
      expected = sparse_shape[i] * 2;
    } else {
      expected = sparse_shape[i] / 2;
    }
    if (expected < sparse_shape.init_threshold()) expected = 0.0f;

    const float result_i = result[i];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if (result_i < sparse_shape.init_threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(mask) {
  // change default threshold to make sure it's not inherited
  auto resetter = set_threshold_to_max();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.mask(right));
  BOOST_CHECK_EQUAL(result.init_threshold(), sparse_shape.init_threshold());

  size_type zero_tile_count = 0ul;
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    if (left.is_zero(i)) {
      ++zero_tile_count;
    }
  }

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    auto threshold = sparse_shape.init_threshold();
    float expected = left[i];
    if (left[i] >= threshold && right[i] < threshold) {
      expected = 0.f;
      ++zero_tile_count;
    }

    BOOST_CHECK_CLOSE(result[i], expected, tolerance);

    // Check zero threshold
    if (result[i] < sparse_shape.init_threshold()) {
      BOOST_CHECK(result.is_zero(i));
    } else {
      BOOST_CHECK(!result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(scale) {
  // change default threshold to make sure it's not inherited
  auto resetter = set_threshold_to_max();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.scale(-4.1));

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value
    float expected = sparse_shape[i] * 4.1;
    if (expected < sparse_shape.init_threshold()) expected = 0.0f;

    const float result_i = result[i];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if (result_i < sparse_shape.init_threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(scale_perm) {
  // change default threshold to make sure it's not inherited
  auto resetter = set_threshold_to_max();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.scale(-5.4, perm));
  BOOST_CHECK_EQUAL(result.init_threshold(), sparse_shape.init_threshold());

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value
    float expected = sparse_shape[i] * 5.4;
    if (expected < sparse_shape.init_threshold()) expected = 0.0f;

    const std::size_t pi = perm_index(i);
    const float result_i = result[pi];
    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if (result_i < sparse_shape.init_threshold()) {
      BOOST_CHECK(result.is_zero(pi));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(pi));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(add) {
  // tweak threshold to make sure result inherits default threshold
  auto resetter = tweak_threshold();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.add(right));
  BOOST_CHECK_EQUAL(result.init_threshold(), SparseShape<float>::threshold());
  BOOST_CHECK_NE(result.init_threshold(), left.init_threshold());
  BOOST_CHECK_NE(result.init_threshold(), right.init_threshold());

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value
    float expected = left[i] + right[i];
    if (expected < SparseShape<float>::threshold()) expected = 0.0f;

    const float result_i = result[i];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if (result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
  BOOST_CHECK_EQUAL(result.nnz(), result.data().size() - zero_tile_count);
}

BOOST_AUTO_TEST_CASE(add_scale) {
  // tweak threshold to make sure result inherits default threshold
  auto resetter = tweak_threshold();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.add(right, -2.2f));
  BOOST_CHECK_EQUAL(result.init_threshold(), SparseShape<float>::threshold());
  BOOST_CHECK_NE(result.init_threshold(), left.init_threshold());
  BOOST_CHECK_NE(result.init_threshold(), right.init_threshold());

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value
    float expected = (left[i] + right[i]) * 2.2f;
    if (expected < SparseShape<float>::threshold()) expected = 0.0f;

    const float result_i = result[i];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if (result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(add_perm) {
  // tweak threshold to make sure result inherits default threshold
  auto resetter = tweak_threshold();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.add(right, perm));
  BOOST_CHECK_EQUAL(result.init_threshold(), SparseShape<float>::threshold());
  BOOST_CHECK_NE(result.init_threshold(), left.init_threshold());
  BOOST_CHECK_NE(result.init_threshold(), right.init_threshold());

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value
    float expected = left[i] + right[i];
    if (expected < SparseShape<float>::threshold()) expected = 0.0f;

    const std::size_t pi = perm_index(i);
    const float result_i = result[pi];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if (result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(pi));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(pi));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(add_scale_perm) {
  // tweak threshold to make sure result inherits default threshold
  auto resetter = tweak_threshold();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.add(right, -2.3f, perm));
  BOOST_CHECK_EQUAL(result.init_threshold(), SparseShape<float>::threshold());
  BOOST_CHECK_NE(result.init_threshold(), left.init_threshold());
  BOOST_CHECK_NE(result.init_threshold(), right.init_threshold());

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value
    float expected = (left[i] + right[i]) * 2.3f;
    if (expected < SparseShape<float>::threshold()) expected = 0.0f;

    const std::size_t pi = perm_index(i);
    const float result_i = result[pi];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if (result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(pi));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(pi));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(add_const) {
  // tweak threshold to make sure result inherits default threshold
  auto resetter = tweak_threshold();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.add(-8.8f));
  BOOST_CHECK_EQUAL(result.init_threshold(), SparseShape<float>::threshold());
  BOOST_CHECK_NE(result.init_threshold(), left.init_threshold());
  BOOST_CHECK_NE(result.init_threshold(), right.init_threshold());

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected =
        sparse_shape[i] + std::sqrt((8.8f * 8.8f) * float(range.volume())) /
                              float(range.volume());
    if (expected < SparseShape<float>::threshold()) expected = 0.0f;

    const float result_i = result[i];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if (result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(add_const_perm) {
  // tweak threshold to make sure result inherits default threshold
  auto resetter = tweak_threshold();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.add(-1.7, perm));
  BOOST_CHECK_EQUAL(result.init_threshold(), SparseShape<float>::threshold());
  BOOST_CHECK_NE(result.init_threshold(), left.init_threshold());
  BOOST_CHECK_NE(result.init_threshold(), right.init_threshold());

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = sparse_shape[i] +
                     std::sqrt((1.7f * 1.7f) * range.volume()) / range.volume();
    if (expected < SparseShape<float>::threshold()) expected = 0.0f;

    const std::size_t pi = perm_index(i);
    const float result_i = result[pi];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if (result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(pi));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(pi));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(subt) {
  // tweak threshold to make sure result inherits default threshold
  auto resetter = tweak_threshold();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.subt(right));
  BOOST_CHECK_EQUAL(result.init_threshold(), SparseShape<float>::threshold());
  BOOST_CHECK_NE(result.init_threshold(), left.init_threshold());
  BOOST_CHECK_NE(result.init_threshold(), right.init_threshold());

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value
    float expected = left[i] + right[i];
    if (expected < SparseShape<float>::threshold()) expected = 0.0f;

    const float result_i = result[i];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if (result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(subt_scale) {
  // tweak threshold to make sure result inherits default threshold
  auto resetter = tweak_threshold();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.subt(right, -2.2f));
  BOOST_CHECK_EQUAL(result.init_threshold(), SparseShape<float>::threshold());
  BOOST_CHECK_NE(result.init_threshold(), left.init_threshold());
  BOOST_CHECK_NE(result.init_threshold(), right.init_threshold());

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value
    float expected = (left[i] + right[i]) * 2.2f;
    if (expected < SparseShape<float>::threshold()) expected = 0.0f;

    const float result_i = result[i];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if (result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(subt_perm) {
  // tweak threshold to make sure result inherits default threshold
  auto resetter = tweak_threshold();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.subt(right, perm));
  BOOST_CHECK_EQUAL(result.init_threshold(), SparseShape<float>::threshold());
  BOOST_CHECK_NE(result.init_threshold(), left.init_threshold());
  BOOST_CHECK_NE(result.init_threshold(), right.init_threshold());

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value
    float expected = left[i] + right[i];
    if (expected < SparseShape<float>::threshold()) expected = 0.0f;

    const std::size_t pi = perm_index(i);
    const float result_i = result[pi];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if (result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(pi));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(pi));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(subt_scale_perm) {
  // tweak threshold to make sure result inherits default threshold
  auto resetter = tweak_threshold();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.subt(right, -2.3f, perm));
  BOOST_CHECK_EQUAL(result.init_threshold(), SparseShape<float>::threshold());
  BOOST_CHECK_NE(result.init_threshold(), left.init_threshold());
  BOOST_CHECK_NE(result.init_threshold(), right.init_threshold());

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value
    float expected = (left[i] + right[i]) * 2.3f;
    if (expected < SparseShape<float>::threshold()) expected = 0.0f;

    const std::size_t pi = perm_index(i);
    const float result_i = result[pi];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if (result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(pi));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(pi));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(subt_const) {
  // tweak threshold to make sure result inherits default threshold
  auto resetter = tweak_threshold();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.subt(-8.8f));
  BOOST_CHECK_EQUAL(result.init_threshold(), SparseShape<float>::threshold());
  BOOST_CHECK_NE(result.init_threshold(), left.init_threshold());
  BOOST_CHECK_NE(result.init_threshold(), right.init_threshold());

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected =
        sparse_shape[i] + std::sqrt((8.8f * 8.8f) * float(range.volume())) /
                              float(range.volume());
    if (expected < SparseShape<float>::threshold()) expected = 0.0f;

    BOOST_CHECK_CLOSE(result[i], expected, tolerance);

    // Check zero threshold
    if (result[i] < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(subt_const_perm) {
  // tweak threshold to make sure result inherits default threshold
  auto resetter = tweak_threshold();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = sparse_shape.subt(-1.7, perm));
  BOOST_CHECK_EQUAL(result.init_threshold(), SparseShape<float>::threshold());
  BOOST_CHECK_NE(result.init_threshold(), left.init_threshold());
  BOOST_CHECK_NE(result.init_threshold(), right.init_threshold());

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected =
        sparse_shape[i] + std::sqrt((1.7f * 1.7f) * float(range.volume())) /
                              float(range.volume());
    if (expected < SparseShape<float>::threshold()) expected = 0.0f;

    const std::size_t pi = perm_index(i);
    const float result_i = result[pi];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if (result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(pi));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(pi));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(mult) {
  // tweak threshold to make sure result inherits default threshold
  auto resetter = tweak_threshold();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.mult(right));
  BOOST_CHECK_EQUAL(result.init_threshold(), SparseShape<float>::threshold());
  BOOST_CHECK_NE(result.init_threshold(), left.init_threshold());
  BOOST_CHECK_NE(result.init_threshold(), right.init_threshold());

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = left[i] * right[i] * float(range.volume());
    if (expected < SparseShape<float>::threshold()) expected = 0.0f;

    BOOST_CHECK_CLOSE(result[i], expected, tolerance);

    // Check zero threshold
    if (result[i] < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(mult_scale) {
  // tweak threshold to make sure result inherits default threshold
  auto resetter = tweak_threshold();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.mult(right, -2.2f));
  BOOST_CHECK_EQUAL(result.init_threshold(), SparseShape<float>::threshold());
  BOOST_CHECK_NE(result.init_threshold(), left.init_threshold());
  BOOST_CHECK_NE(result.init_threshold(), right.init_threshold());

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = (left[i] * right[i]) * 2.2f * float(range.volume());
    if (expected < SparseShape<float>::threshold()) expected = 0.0f;

    BOOST_CHECK_CLOSE(result[i], expected, tolerance);

    // Check zero threshold
    if (result[i] < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(mult_perm) {
  // tweak threshold to make sure result inherits default threshold
  auto resetter = tweak_threshold();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.mult(right, perm));
  BOOST_CHECK_EQUAL(result.init_threshold(), SparseShape<float>::threshold());
  BOOST_CHECK_NE(result.init_threshold(), left.init_threshold());
  BOOST_CHECK_NE(result.init_threshold(), right.init_threshold());

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = left[i] * right[i] * float(range.volume());
    if (expected < SparseShape<float>::threshold()) expected = 0.0f;

    const std::size_t pi = perm_index(i);
    const float result_i = result[pi];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if (result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(pi));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(pi));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(mult_scale_perm) {
  // tweak threshold to make sure result inherits default threshold
  auto resetter = tweak_threshold();

  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.mult(right, -2.3f, perm));
  BOOST_CHECK_EQUAL(result.init_threshold(), SparseShape<float>::threshold());
  BOOST_CHECK_NE(result.init_threshold(), left.init_threshold());
  BOOST_CHECK_NE(result.init_threshold(), right.init_threshold());

  size_type zero_tile_count = 0ul;

  // Check that all the tiles have been normalized correctly
  for (Tensor<float>::size_type i = 0ul; i < tr.tiles_range().volume(); ++i) {
    // Compute expected value
    const TiledRange::range_type range = tr.make_tile_range(i);
    float expected = (left[i] * right[i]) * 2.3f * float(range.volume());
    if (expected < SparseShape<float>::threshold()) expected = 0.0f;

    const std::size_t pi = perm_index(i);
    const float result_i = result[pi];

    BOOST_CHECK_CLOSE(result_i, expected, tolerance);

    // Check zero threshold
    if (result_i < SparseShape<float>::threshold()) {
      BOOST_CHECK(result.is_zero(pi));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(!result.is_zero(pi));
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(tr.tiles_range().volume()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(gemm) {
  // tweak threshold to make sure result inherits default threshold
  auto resetter = tweak_threshold();

  // Create a matrix with the expected output
  const std::size_t m = left.data().range().extent(0);
  const std::size_t n =
      right.data().range().extent(right.data().range().rank() - 1);
  //  const std::size_t k = left.data().size() / m;

  size_type zero_tile_count = 0ul;

  // Evaluate the contraction of sparse shapes
  math::GemmHelper gemm_helper(
      TiledArray::math::blas::Op::NoTrans, TiledArray::math::blas::Op::NoTrans,
      2u, left.data().range().rank(), right.data().range().rank());
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.gemm(right, -7.2, gemm_helper));
  BOOST_CHECK_EQUAL(result.init_threshold(), SparseShape<float>::threshold());
  BOOST_CHECK_NE(result.init_threshold(), left.init_threshold());
  BOOST_CHECK_NE(result.init_threshold(), right.init_threshold());

  // Create volumes tensors for the arguments
  Tensor<float> volumes(tr.tiles_range(), 0.0f);
  for (std::size_t i = 0ul; i < tr.tiles_range().volume(); ++i) {
    const float volume = tr.make_tile_range(i).volume();
    volumes[i] = volume;
  }

  Tensor<float> result_norms = left.data().mult(volumes).gemm(
      right.data().mult(volumes), 7.2, gemm_helper);

  // Check that the result is correct
  std::array<std::size_t, 2> i = {{0, 0}};
  for (i[0] = 0ul; i[0] < m; ++i[0]) {
    const TiledRange1::range_type r_0 = tr.data()[0].tile(i[0]);
    const float size_0 = r_0.second - r_0.first;

    for (i[1] = 0ul; i[1] < n; ++i[1]) {
      const TiledRange1::range_type r_1 = tr.data()[2].tile(i[1]);
      const float size_1 = r_1.second - r_1.first;

      // Compute expected value
      float expected = result_norms[i] / (size_0 * size_1);
      if (expected < SparseShape<float>::threshold()) expected = 0.0f;

      BOOST_CHECK_CLOSE(result[i], expected, tolerance);

      // Check zero threshold
      if (result[i] < SparseShape<float>::threshold()) {
        BOOST_CHECK(result.is_zero(i));
        ++zero_tile_count;
      } else {
        BOOST_CHECK(!result.is_zero(i));
      }
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(result_norms.size()),
                    tolerance);
}

BOOST_AUTO_TEST_CASE(gemm_perm) {
  // tweak threshold to make sure result inherits default threshold
  auto resetter = tweak_threshold();

  const Permutation perm({1, 0});

  // Create a matrix with the expected output
  const std::size_t m = left.data().range().extent(0);
  const std::size_t n =
      right.data().range().extent(right.data().range().rank() - 1);
  //  const std::size_t k = left.data().size() / m;

  size_type zero_tile_count = 0ul;

  // Evaluate the contraction of sparse shapes
  math::GemmHelper gemm_helper(
      TiledArray::math::blas::Op::NoTrans, TiledArray::math::blas::Op::NoTrans,
      2u, left.data().range().rank(), right.data().range().rank());
  SparseShape<float> result;
  BOOST_REQUIRE_NO_THROW(result = left.gemm(right, -7.2, gemm_helper, perm));
  BOOST_CHECK_EQUAL(result.init_threshold(), SparseShape<float>::threshold());
  BOOST_CHECK_NE(result.init_threshold(), left.init_threshold());
  BOOST_CHECK_NE(result.init_threshold(), right.init_threshold());

  // Create volumes tensors for the arguments
  Tensor<float> volumes(tr.tiles_range(), 0.0f);
  for (std::size_t i = 0ul; i < tr.tiles_range().volume(); ++i) {
    const float volume = tr.make_tile_range(i).volume();
    volumes[i] = volume;
  }

  Tensor<float> result_norms =
      left.data()
          .mult(volumes)
          .gemm(right.data().mult(volumes), 7.2, gemm_helper)
          .permute(perm);

  // Check that the result is correct
  std::array<std::size_t, 2> i = {{0, 0}};
  for (i[0] = 0ul; i[0] < n; ++i[0]) {
    const TiledRange1::range_type r_0 = tr.data()[2].tile(i[0]);
    const float size_0 = r_0.second - r_0.first;

    for (i[1] = 0ul; i[1] < m; ++i[1]) {
      const TiledRange1::range_type r_1 = tr.data()[0].tile(i[1]);
      const float size_1 = r_1.second - r_1.first;

      // Compute expected value
      float expected = result_norms[i] / (size_0 * size_1);
      if (expected < SparseShape<float>::threshold()) expected = 0.0f;

      BOOST_CHECK_CLOSE(result[i], expected, tolerance);

      // Check zero threshold
      if (result[i] < SparseShape<float>::threshold()) {
        BOOST_CHECK(result.is_zero(i));
        ++zero_tile_count;
      } else {
        BOOST_CHECK(!result.is_zero(i));
      }
    }
  }

  BOOST_CHECK_CLOSE(result.sparsity(),
                    float(zero_tile_count) / float(result_norms.size()),
                    tolerance);
}

BOOST_AUTO_TEST_SUITE_END()
