/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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
 *  tensor_view.cpp
 *  May 29, 2015
 *
 */

#include "TiledArray/tensor/tensor_view.h"
#include "tiledarray.h"
#include "unit_test_config.h"
#include <random>

using TiledArray::Tensor;
using TiledArray::TensorView;
using TiledArray::TensorConstView;
using TiledArray::Range;
using TiledArray::BlockRange;

struct TensorViewFixture {

  TensorViewFixture() : t(random_tensor(Range(std::vector<int>{0,1,2}, std::vector<int>{5,7,11})))
  { }

  ~TensorViewFixture() { }

  static Tensor<int> random_tensor(const Range& range) {
    Tensor<int> result(range);

    std::default_random_engine generator;
    std::uniform_int_distribution<int> distribution(0,100);

    for(auto& value : result)
      value = distribution(generator);

    return result;
  }

  Tensor<int> t;
}; // TensorViewFixture

BOOST_FIXTURE_TEST_SUITE( tensor_view_suite, TensorViewFixture )

BOOST_AUTO_TEST_CASE( non_const_view )
{

  for(auto lower_it = t.range().begin(); lower_it != t.range().end(); ++lower_it) {
    const auto lower = *lower_it;
    for(auto upper_it = t.range().begin(); upper_it != t.range().end(); ++upper_it) {
      std::vector<std::size_t> upper = *upper_it;
      for(unsigned int i = 0u; i < upper.size(); ++i)
        ++(upper[i]);


      if(std::equal(lower.begin(), lower.end(), upper.begin(),
          [] (std::size_t l, std::size_t r) { return l < r; })) {

        // Check that the sub-block is constructed without exceptions
        BOOST_CHECK_NO_THROW(TensorView<int> view(t, lower,upper));
        TensorView<int> view(t, lower,upper);

        // Check that the data of the block range is correct
        std::size_t volume = 1ul;
        for(unsigned int i = 0u; i < t.range().rank(); ++i) {
          BOOST_CHECK_EQUAL(view.range().start()[i], lower[i]);
          BOOST_CHECK_EQUAL(view.range().finish()[i], upper[i]);
          BOOST_CHECK_EQUAL(view.range().size()[i], upper[i] - lower[i]);
          BOOST_CHECK_EQUAL(view.range().weight()[i], t.range().weight()[i]);
          volume *= upper[i] - lower[i];
        }
        BOOST_CHECK_EQUAL(view.size(), volume);

        // Check that the subrange ordinal calculation returns the same offset
        // as the original range.
        Range::size_type i = 0ul;
        for(auto it = view.range().begin(); it != view.range().end(); ++it, ++i) {
          BOOST_CHECK_EQUAL(view(i), view(*it));
          BOOST_CHECK_EQUAL(view(*it), t(*it));
          BOOST_CHECK_EQUAL(view(i), t(*it));
        }
      }
    }
  }
}


BOOST_AUTO_TEST_CASE( const_view )
{

  for(auto lower_it = t.range().begin(); lower_it != t.range().end(); ++lower_it) {
    const auto lower = *lower_it;
    for(auto upper_it = t.range().begin(); upper_it != t.range().end(); ++upper_it) {
      std::vector<std::size_t> upper = *upper_it;
      for(unsigned int i = 0u; i < upper.size(); ++i)
        ++(upper[i]);


      if(std::equal(lower.begin(), lower.end(), upper.begin(),
          [] (std::size_t l, std::size_t r) { return l < r; })) {

        // Check that the sub-block is constructed without exceptions
        BOOST_CHECK_NO_THROW(TensorConstView<int> view(t, lower, upper));
        TensorConstView<int> view(t, lower, upper);

        // Check that the data of the block range is correct
        std::size_t volume = 1ul;
        for(unsigned int i = 0u; i < t.range().rank(); ++i) {
          BOOST_CHECK_EQUAL(view.range().start()[i], lower[i]);
          BOOST_CHECK_EQUAL(view.range().finish()[i], upper[i]);
          BOOST_CHECK_EQUAL(view.range().size()[i], upper[i] - lower[i]);
          BOOST_CHECK_EQUAL(view.range().weight()[i], t.range().weight()[i]);
          volume *= upper[i] - lower[i];
        }
        BOOST_CHECK_EQUAL(view.size(), volume);

        // Check that the subrange ordinal calculation returns the same offset
        // as the original range.
        Range::size_type i = 0ul;
        for(auto it = view.range().begin(); it != view.range().end(); ++it, ++i) {
          BOOST_CHECK_EQUAL(view(i), view(*it));
          BOOST_CHECK_EQUAL(view(*it), t(*it));
          BOOST_CHECK_EQUAL(view(i), t(*it));
        }
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( assign_tensor_to_view )
{

  for(auto lower_it = t.range().begin(); lower_it != t.range().end(); ++lower_it) {
    const auto lower = *lower_it;
    for(auto upper_it = t.range().begin(); upper_it != t.range().end(); ++upper_it) {
      std::vector<std::size_t> upper = *upper_it;
      for(unsigned int i = 0u; i < upper.size(); ++i)
        ++(upper[i]);


      if(std::equal(lower.begin(), lower.end(), upper.begin(),
          [] (std::size_t l, std::size_t r) { return l < r; })) {

        // Check that the sub-block is constructed without exceptions
        BOOST_CHECK_NO_THROW(TensorView<int> view(t, lower,upper));
        TensorView<int> view(t, lower,upper);
        Tensor<int> tensor = random_tensor(Range(lower, upper));

        BOOST_CHECK_NO_THROW(view = tensor);

        // Check that the subrange ordinal calculation returns the same offset
        // as the original range.
        std::size_t i = 0ul;
        for(auto it = view.range().begin(); it != view.range().end(); ++it, ++i) {
          BOOST_CHECK_EQUAL(view(i), view(*it));
          BOOST_CHECK_EQUAL(view(*it), tensor(*it));
          BOOST_CHECK_EQUAL(view(i), tensor(*it));
        }
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
