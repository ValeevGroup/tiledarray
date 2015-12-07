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

#include "TiledArray/shape/sparse_shape.h"
#include "TiledArray/shape/sparse_shape_expt.h"
#include "tiledarray.h"
#include "unit_test_config.h"
#include "range_fixture.h"

#include <boost/test/test_case_template.hpp>
#include <boost/mpl/list.hpp>

using namespace TiledArray;
namespace expt = TiledArray::experimental;

typedef boost::mpl::list<expt::SparseShape<float>,expt::SparseShape<bool>> stypes;
typedef boost::mpl::list<expt::SparseShape<float>> stypes0;
typedef boost::mpl::list<expt::SparseShape<bool>> stypes1;

template <typename T>
struct TSparseShapeExptFixture : public TiledRangeFixture{
    typedef std::vector<std::size_t> vec_type;

    TSparseShapeExptFixture() :
      sparse_shape(make_shape(tr, 0.5, 42)),
      left(make_shape(tr, 0.1, 23)),
      right(make_shape(tr, 0.1, 82)),
      perm(make_perm()),
      perm_index(tr.tiles(), perm),
      real_check_tolerance(0.0001)

    {
      expt::SparseShape<T>::default_threshold(default_threshold_);
    }

    ~TSparseShapeExptFixture() { }

    static Tensor<T> make_norm_tensor(const TiledRange& trange,
                                      const float fill_fraction,
                                      const int seed) {
      assert(fill_fraction <= 1.0 && fill_fraction >= 0.0);
      GlobalFixture::world->srand(seed);
      Tensor<T> norms(trange.tiles());
      for(typename Tensor<T>::size_type i = 0ul; i < norms.size(); ++i) {
        const Range range = trange.make_tile_range(i);
        // make sure nonzero tile norms are MUCH greater than threshold since SparseShape scales norms by 1/volume
        norms[i] = expt::SparseShape<T>::default_threshold() * (100 + (GlobalFixture::world->rand() % 1000));
      }

      const std::size_t target_num_zeroes = std::min(norms.size(), static_cast<std::size_t>(double(norms.size()) * (1.0 - fill_fraction)));
      std::size_t num_zeroes = 0ul;
      while (num_zeroes != target_num_zeroes) {
        const auto zero_norm = expt::SparseShape<T>::default_threshold() / 2;
        const size_t rand_idx = GlobalFixture::world->rand() % norms.size();
        if (norms[rand_idx] != zero_norm) {
          norms[rand_idx] = zero_norm;
          ++num_zeroes;
        }
      }
      return norms;

    }

    static expt::SparseShape<T> make_shape(const TiledRange& trange,
                                           const float fill_percent,
                                           const int seed) {
      Tensor<T> tile_norms = make_norm_tensor(trange, fill_percent, seed);
      auto result = expt::SparseShape<T>(tile_norms);
      return result;
    }

    static Permutation make_perm() {
      std::array<unsigned int, GlobalFixture::dim> temp;
      for(std::size_t i = 0; i < temp.size(); ++i)
        temp[i] = i + 1;

      temp.back() = 0;

      return Permutation(temp.begin(), temp.end());
    }

    expt::SparseShape<T> sparse_shape;
    expt::SparseShape<T> left;
    expt::SparseShape<T> right;
    Permutation perm;
    TiledArray::detail::PermIndex perm_index;
    const double real_check_tolerance;
    static T default_threshold_;
}; // TSparseShapeExptFixture

template<> float TSparseShapeExptFixture<float>::default_threshold_(1e-3);
template<> double TSparseShapeExptFixture<double>::default_threshold_(1e-3);
template<> bool TSparseShapeExptFixture<bool>::default_threshold_(true);

struct SparseShapeExptFixture {

    SparseShapeExptFixture() {}
    ~SparseShapeExptFixture() {}

    template <typename T> // shape value_type
    TSparseShapeExptFixture<T>& fixture() {
      static TSparseShapeExptFixture<T> fixture_;
      return fixture_;
    }
}; // struct SparseShapeExptFixture

BOOST_FIXTURE_TEST_SUITE( sparse_shape_expt_suite, SparseShapeExptFixture )

BOOST_AUTO_TEST_CASE_TEMPLATE( default_constructor, Shape, stypes )
{
  using value_type = typename Shape::value_type;
  const auto& f = fixture<value_type>();

  BOOST_CHECK_NO_THROW(Shape x);
  Shape x, y;
  Permutation perm;
  math::GemmHelper gemm_helper(madness::cblas::NoTrans, madness::cblas::NoTrans,
      2u, 2u, 2u);

  BOOST_CHECK(x.empty());
  BOOST_CHECK(! x.is_dense());
  BOOST_CHECK(! x.validate(f.tr.tiles()));

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(x[0], Exception);

  BOOST_CHECK_THROW(x.perm(perm), Exception);

  BOOST_CHECK_THROW(x.scale(2.0), Exception);
  BOOST_CHECK_THROW(x.scale(2.0, perm), Exception);

  BOOST_CHECK_THROW(x.add(y), Exception);
  BOOST_CHECK_THROW(x.add(y, 2.0), Exception);
  BOOST_CHECK_THROW(x.add(y, perm), Exception);
  BOOST_CHECK_THROW(x.add(y, 2.0, perm), Exception);

  BOOST_CHECK_THROW(x.subt(y), Exception);
  BOOST_CHECK_THROW(x.subt(y, 2.0), Exception);
  BOOST_CHECK_THROW(x.subt(y, perm), Exception);
  BOOST_CHECK_THROW(x.subt(y, 2.0, perm), Exception);

  BOOST_CHECK_THROW(x.mult(y), Exception);
  BOOST_CHECK_THROW(x.mult(y, 2.0), Exception);
  BOOST_CHECK_THROW(x.mult(y, perm), Exception);
  BOOST_CHECK_THROW(x.mult(y, 2.0, perm), Exception);

  BOOST_CHECK_THROW(x.gemm(y, 2.0, gemm_helper), Exception);
  BOOST_CHECK_THROW(x.gemm(y, 2.0, gemm_helper, perm), Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE_TEMPLATE( non_comm_constructor, Shape, stypes )
{
  using value_type = typename Shape::value_type;
  const auto& f = fixture<value_type>();

  // Construct test tile norms
  auto tile_norms = TSparseShapeExptFixture<value_type>::make_norm_tensor(f.tr, 1, 42);

  // Construct the shape
  BOOST_CHECK_NO_THROW(Shape x(tile_norms));
  Shape x(tile_norms);

  // Check that the shape has been initialized
  BOOST_CHECK(! x.empty());
  BOOST_CHECK(! x.is_dense());
  BOOST_CHECK(x.validate(f.tr.tiles()));

  size_t zero_tile_count = 0ul;

  for(auto i = 0ul; i < tile_norms.size(); ++i) {
    // Check zero threshold
    if(x[i] < x.threshold()) {
      BOOST_CHECK(x.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! x.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(x.sparsity(),
                    float(zero_tile_count) / float(f.tr.tiles().volume()),
                    f.real_check_tolerance);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( comm_constructor, Shape, stypes )
{
  using value_type = typename Shape::value_type;
  const auto& f = fixture<value_type>();

  // Construct test tile norms
  auto tile_norms = TSparseShapeExptFixture<value_type>::make_norm_tensor(f.tr, 1, 98);
  auto tile_norms_ref = tile_norms.clone();

  // Zero non-local tiles
  TiledArray::detail::BlockedPmap pmap(*GlobalFixture::world, f.tr.tiles().volume());
  for(Tensor<float>::size_type i = 0ul; i < tile_norms.size(); ++i)
    if(! pmap.is_local(i))
      tile_norms[i] = 0.0f;

  // Construct the shape
  BOOST_CHECK_NO_THROW(SparseShape<float> x(*GlobalFixture::world, tile_norms, f.tr));
  SparseShape<float> x(*GlobalFixture::world, tile_norms, f.tr);

  // Check that the shape has been initialized
  BOOST_CHECK(! x.empty());
  BOOST_CHECK(! x.is_dense());
  BOOST_CHECK(x.validate(f.tr.tiles()));

  auto zero_tile_count = 0ul;

  for(auto i = 0ul; i < tile_norms.size(); ++i) {
    // Check zero threshold
    if(x[i] < x.threshold()) {
      BOOST_CHECK(x.is_zero(i));
      ++zero_tile_count;
    } else {
      BOOST_CHECK(! x.is_zero(i));
    }
  }

  BOOST_CHECK_CLOSE(x.sparsity(),
                    float(zero_tile_count) / float(f.tr.tiles().volume()),
                    f.real_check_tolerance);
}

BOOST_AUTO_TEST_CASE_TEMPLATE( copy_constructor, Shape, stypes )
{
  using value_type = typename Shape::value_type;
  const auto& f = fixture<value_type>();

  // Construct the shape
  BOOST_CHECK_NO_THROW(Shape y(f.sparse_shape));
  Shape y(f.sparse_shape);

  // Check that the shape has been initialized
  BOOST_CHECK(! y.empty());
  BOOST_CHECK(! y.is_dense());
  BOOST_CHECK(y.validate(f.tr.tiles()));

  // Check that all the tiles have been normalized correctly
  for(auto i = 0ul; i < f.tr.tiles().volume(); ++i) {
    // Check that the tile data has been copied correctly
    BOOST_CHECK_EQUAL(y[i], f.sparse_shape[i]);
  }

  BOOST_CHECK_EQUAL(y.sparsity(), f.sparse_shape.sparsity());
}

BOOST_AUTO_TEST_CASE_TEMPLATE( permute, Shape, stypes )
{
  using value_type = typename Shape::value_type;
  const auto& f = fixture<value_type>();

  Shape result;
  BOOST_REQUIRE_NO_THROW(result = f.sparse_shape.perm(f.perm));

  // Check that all the tiles have been normalized correctly
  for(auto i = 0ul; i < f.tr.tiles().volume(); ++i) {
    BOOST_CHECK_EQUAL(result[f.perm * f.tr.tiles().idx(i)], f.sparse_shape[i]);
  }

  BOOST_CHECK_EQUAL(result.sparsity(), f.sparse_shape.sparsity());
}

BOOST_AUTO_TEST_CASE_TEMPLATE( block, Shape, stypes )
{
  using value_type = typename Shape::value_type;
  const auto& f = fixture<value_type>();

  auto less = std::less<std::size_t>();

  for(auto lower_it = f.tr.tiles().begin(); lower_it != f.tr.tiles().end(); ++lower_it) {
    const auto& lower = *lower_it;

    for(auto upper_it = f.tr.tiles().begin(); upper_it != f.tr.tiles().end(); ++upper_it) {
      std::vector<std::size_t> upper = *upper_it;
      for(auto it = upper.begin(); it != upper.end(); ++it)
        *it += 1;

      if(std::equal(lower.begin(), lower.end(), upper.begin(), less)) {
        // Check that the block function does not throw an exception
        Shape result;
        BOOST_REQUIRE_NO_THROW(result = f.sparse_shape.block(lower, upper));

        // Check that the block range data is correct
        std::size_t volume = 1ul;
        for(int i = int(f.tr.tiles().rank()) - 1u; i >= 0; --i) {
          auto size_i = upper[i] - lower[i];
          BOOST_CHECK_EQUAL(result.data().range().lobound_data()[i], 0);
          BOOST_CHECK_EQUAL(result.data().range().upbound_data()[i], size_i);
          BOOST_CHECK_EQUAL(result.data().range().extent_data()[i], size_i);
          BOOST_CHECK_EQUAL(result.data().range().stride_data()[i], volume);
          volume *= size_i;
        }
        BOOST_CHECK_EQUAL(result.data().range().volume(), volume);

        // Check that the data was copied and scaled correctly
        unsigned long i = 0ul;
        unsigned long zero_tile_count = 0ul;
        std::vector<std::size_t> arg_index(f.sparse_shape.data().range().rank(), 0ul);
        for(auto it = result.data().range().begin(); it != result.data().range().end(); ++it, ++i) {
          // Construct the coordinate index for the argument element
          for(unsigned int i = 0u; i < f.sparse_shape.data().range().rank(); ++i)
            arg_index[i] = (*it)[i] + lower[i];

          // Check the result elements
          BOOST_CHECK_EQUAL(result.data()(*it), f.sparse_shape.data()(arg_index));
          BOOST_CHECK_EQUAL(result.data()[i], f.sparse_shape.data()(arg_index));
          if(result.data()[i] < result.threshold())
            ++zero_tile_count;
        }
        BOOST_CHECK_CLOSE(result.sparsity(),
                          float(zero_tile_count)/float(result.data().range().volume()),
                          f.real_check_tolerance);
      }
#ifdef TA_EXCEPTION_ERROR
      else {
        // Check that block throws an exception with a bad block range
        BOOST_CHECK_THROW(f.sparse_shape.block(lower, upper), TiledArray::Exception);
      }
#endif // TA_EXCEPTION_ERROR
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( block_scale, Shape, stypes )
{
  using value_type = typename Shape::value_type;
  const auto& f = fixture<value_type>();

  auto less = std::less<std::size_t>();
  const float factor = 3.3;

  for(auto lower_it = f.tr.tiles().begin(); lower_it != f.tr.tiles().end(); ++lower_it) {
    const auto& lower = *lower_it;

    for(auto upper_it = f.tr.tiles().begin(); upper_it != f.tr.tiles().end(); ++upper_it) {
      std::vector<std::size_t> upper = *upper_it;
      for(auto it = upper.begin(); it != upper.end(); ++it)
        *it += 1;

      if(std::equal(lower.begin(), lower.end(), upper.begin(), less)) {

        // Check that the block function does not throw an exception
        Shape result;
        BOOST_REQUIRE_NO_THROW(result = f.sparse_shape.block(lower, upper, factor));

        // Check that the block range data is correct
        std::size_t volume = 1ul;
        for(int i = int(f.tr.tiles().rank()) - 1u; i >= 0; --i) {
          auto size_i = upper[i] - lower[i];
          BOOST_CHECK_EQUAL(result.data().range().lobound_data()[i], 0);
          BOOST_CHECK_EQUAL(result.data().range().upbound_data()[i], size_i);
          BOOST_CHECK_EQUAL(result.data().range().extent_data()[i], size_i);
          BOOST_CHECK_EQUAL(result.data().range().stride_data()[i], volume);
          volume *= size_i;
        }
        BOOST_CHECK_EQUAL(result.data().range().volume(), volume);

        unsigned long i = 0ul;
        unsigned long zero_tile_count = 0ul;
        std::vector<std::size_t> arg_index(f.sparse_shape.data().range().rank(), 0ul);
        for(auto it = result.data().range().begin(); it != result.data().range().end(); ++it, ++i) {
          // Construct the coordinate index for the argument element
          for(unsigned int i = 0u; i < f.sparse_shape.data().range().rank(); ++i)
            arg_index[i] = (*it)[i] + lower[i];

          // Compute the expected value
          const value_type expected = f.sparse_shape.data()(arg_index) * factor;

          // Check the result elements
          BOOST_CHECK_EQUAL(result.data()(*it), expected);
          BOOST_CHECK_EQUAL(result.data()[i], expected);
          if(result.data()[i] < result.threshold())
            ++zero_tile_count;
        }
        BOOST_CHECK_CLOSE(result.sparsity(),
                          float(zero_tile_count)/float(result.data().range().volume()),
                          f.real_check_tolerance);
      }
#ifdef TA_EXCEPTION_ERROR
      else {
        // Check that block throws an exception with a bad block range
        BOOST_CHECK_THROW(f.sparse_shape.block(lower, upper), TiledArray::Exception);
      }
#endif // TA_EXCEPTION_ERROR
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( block_perm, Shape, stypes )
{
  using value_type = typename Shape::value_type;
  const auto& f = fixture<value_type>();

  auto less = std::less<std::size_t>();
  const auto inv_perm = f.perm.inv();

  for(auto lower_it = f.tr.tiles().begin(); lower_it != f.tr.tiles().end(); ++lower_it) {
    const auto& lower = *lower_it;

    for(auto upper_it = f.tr.tiles().begin(); upper_it != f.tr.tiles().end(); ++upper_it) {
      std::vector<std::size_t> upper = *upper_it;
      for(auto it = upper.begin(); it != upper.end(); ++it)
        *it += 1;

      if(std::equal(lower.begin(), lower.end(), upper.begin(), less)) {

        // Check that the block function does not throw an exception
        Shape result;
        BOOST_REQUIRE_NO_THROW(result = f.sparse_shape.block(lower, upper, f.perm));

        // Check that the block range data is correct
        std::size_t volume = 1ul;
        for(int i = int(f.tr.tiles().rank()) - 1u; i >= 0; --i) {
          const auto inv_perm_i = inv_perm[i];
          const auto size_i = upper[inv_perm_i] - lower[inv_perm_i];
          BOOST_CHECK_EQUAL(result.data().range().lobound_data()[i], 0);
          BOOST_CHECK_EQUAL(result.data().range().upbound_data()[i], size_i);
          BOOST_CHECK_EQUAL(result.data().range().extent_data()[i], size_i);
          BOOST_CHECK_EQUAL(result.data().range().stride_data()[i], volume);
          volume *= size_i;
        }
        BOOST_CHECK_EQUAL(result.data().range().volume(), volume);

        // Check that the data was copied and scaled correctly
        unsigned long i = 0ul;
        unsigned long zero_tile_count = 0ul;
        std::vector<std::size_t> arg_index(f.sparse_shape.data().range().rank(), 0ul);
        for(auto it = result.data().range().begin(); it != result.data().range().end(); ++it, ++i) {
          // Construct the coordinate index for the argument element
          for(unsigned int i = 0u; i < f.sparse_shape.data().range().rank(); ++i) {
            const auto perm_i = f.perm[i];
            arg_index[i] = (*it)[perm_i] + lower[i];
          }

          // Check the result elements
          BOOST_CHECK_EQUAL(result.data()(*it), f.sparse_shape.data()(arg_index));
          BOOST_CHECK_EQUAL(result.data()[i], f.sparse_shape.data()(arg_index));
          if(result.data()[i] < result.threshold())
            ++zero_tile_count;
        }
        BOOST_CHECK_CLOSE(result.sparsity(),
                          float(zero_tile_count)/float(result.data().range().volume()),
                          f.real_check_tolerance);
      }
#ifdef TA_EXCEPTION_ERROR
      else {
        // Check that block throws an exception with a bad block range
        BOOST_CHECK_THROW(f.sparse_shape.block(lower, upper), TiledArray::Exception);
      }
#endif // TA_EXCEPTION_ERROR
    }
  }
}

BOOST_AUTO_TEST_CASE_TEMPLATE( block_scale_perm, Shape, stypes )
{
  using value_type = typename Shape::value_type;
  const auto& f = fixture<value_type>();

  auto less = std::less<std::size_t>();
  const float factor = 3.3;
  const auto inv_perm = f.perm.inv();

  for(auto lower_it = f.tr.tiles().begin(); lower_it != f.tr.tiles().end(); ++lower_it) {
    const auto& lower = *lower_it;

    for(auto upper_it = f.tr.tiles().begin(); upper_it != f.tr.tiles().end(); ++upper_it) {
      std::vector<std::size_t> upper = *upper_it;
      for(auto it = upper.begin(); it != upper.end(); ++it)
        *it += 1;

      if(std::equal(lower.begin(), lower.end(), upper.begin(), less)) {

        // Check that the block function does not throw an exception
        Shape result;
        BOOST_REQUIRE_NO_THROW(result = f.sparse_shape.block(lower, upper, factor, f.perm));

        // Check that the block range data is correct
        std::size_t volume = 1ul;
        for(int i = int(f.tr.tiles().rank()) - 1u; i >= 0; --i) {
          const auto inv_perm_i = inv_perm[i];
          const auto size_i = upper[inv_perm_i] - lower[inv_perm_i];
          BOOST_CHECK_EQUAL(result.data().range().lobound_data()[i], 0);
          BOOST_CHECK_EQUAL(result.data().range().upbound_data()[i], size_i);
          BOOST_CHECK_EQUAL(result.data().range().extent_data()[i], size_i);
          BOOST_CHECK_EQUAL(result.data().range().stride_data()[i], volume);
          volume *= size_i;
        }
        BOOST_CHECK_EQUAL(result.data().range().volume(), volume);

        unsigned long i = 0ul;
        unsigned long zero_tile_count = 0ul;
        std::vector<std::size_t> arg_index(f.sparse_shape.data().range().rank(), 0ul);
        for(auto it = result.data().range().begin(); it != result.data().range().end(); ++it, ++i) {
          // Construct the coordinate index for the argument element
          for(unsigned int i = 0u; i < f.sparse_shape.data().range().rank(); ++i) {
            const auto perm_i = f.perm[i];
            arg_index[i] = (*it)[perm_i] + lower[i];
          }

          // Compute the expected value
          const value_type expected = f.sparse_shape.data()(arg_index) * factor;

          // Check the result elements
          BOOST_CHECK_EQUAL(result.data()(*it), expected);
          BOOST_CHECK_EQUAL(result.data()[i], expected);
          if(result.data()[i] < result.threshold())
            ++zero_tile_count;
        }
        BOOST_CHECK_CLOSE(result.sparsity(),
                          float(zero_tile_count)/float(result.data().range().volume()),
                          f.real_check_tolerance);
      }
#ifdef TA_EXCEPTION_ERROR
      else {
        // Check that block throws an exception with a bad block range
        BOOST_CHECK_THROW(f.sparse_shape.block(lower, upper), TiledArray::Exception);
      }
#endif // TA_EXCEPTION_ERROR
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
