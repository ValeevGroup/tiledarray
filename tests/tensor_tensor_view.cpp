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

#include "TiledArray/tensor/tensor_interface.h"
#include "tiledarray.h"
#include "unit_test_config.h"
#include <random>

using namespace TiledArray;

struct TensorViewFixture {

  TensorViewFixture() { }

  ~TensorViewFixture() { }

  static Tensor<int> random_tensor(const Range& range) {
    Tensor<int> result(range);

    std::default_random_engine generator(
        std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> distribution(0,100);

    for(auto& value : result)
      value = distribution(generator);

    return result;
  }

  static const std::array<int, 3> lower_bound;
  static const std::array<int, 3> upper_bound;

  Tensor<int> t{random_tensor(Range(lower_bound, upper_bound))};

}; // TensorViewFixture

const std::array<int, 3> TensorViewFixture::lower_bound{{0,1,2}};
const std::array<int, 3> TensorViewFixture::upper_bound{{5,7,11}};

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
        BOOST_CHECK_NO_THROW(t.block(lower,upper));
        TensorView<int> view = t.block(lower,upper);

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
        BOOST_CHECK_NO_THROW(t.block(lower, upper));
        TensorConstView<int> view = t.block(lower, upper);

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
        BOOST_CHECK_NO_THROW(t.block(lower,upper));
        TensorView<int> view = t.block(lower,upper);
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

BOOST_AUTO_TEST_CASE( add_tensor_to_view )
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
        BOOST_CHECK_NO_THROW(t.block(lower,upper));
        TensorView<int> view = t.block(lower,upper);
        Tensor<int> tensor = random_tensor(Range(lower, upper));

        Tensor<int> temp(view);
        BOOST_CHECK_NO_THROW(view.add_to(tensor));

        // Check that the subrange ordinal calculation returns the same offset
        // as the original range.
        std::size_t i = 0ul;
        for(auto it = view.range().begin(); it != view.range().end(); ++it, ++i) {
          BOOST_CHECK_EQUAL(view(i), view(*it));
          BOOST_CHECK_EQUAL(view(*it), temp(*it) + tensor(*it));
          BOOST_CHECK_EQUAL(view(i), temp(*it) + tensor(*it));
        }
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
