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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  tensor_shift_wrapper.cpp
 *  Jun 2, 2015
 *
 */

#include <chrono>
#include <random>
#include "TiledArray/tensor/shift_wrapper.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using TiledArray::BlockRange;
using TiledArray::Range;
using TiledArray::shift;
using TiledArray::Tensor;
using TiledArray::TensorConstView;
using TiledArray::TensorView;
using TiledArray::detail::ShiftWrapper;

struct ShiftWrapperFixture {
  ShiftWrapperFixture() {}

  ~ShiftWrapperFixture() {}

  static Tensor<int> random_tensor(const Range& range) {
    Tensor<int> result(range);

    std::default_random_engine generator(
        std::chrono::system_clock::now().time_since_epoch().count());
    std::uniform_int_distribution<int> distribution(0, 100);

    for (auto& value : result) value = distribution(generator);

    return result;
  }

  std::array<int, 3> lower_bound{{0, 1, 2}};
  std::array<int, 3> upper_bound{{5, 7, 11}};
  std::array<int, 3> shifted_lower_bound{{4, 5, 6}};
  std::array<int, 3> shifted_upper_bound{{9, 11, 15}};
  Tensor<int> t1{random_tensor(Range(lower_bound, upper_bound))};
  Tensor<int> t2{
      random_tensor(Range(shifted_lower_bound, shifted_upper_bound))};
};  // ShiftWrapperFixture

BOOST_FIXTURE_TEST_SUITE(tensor_shift_wrapper_suite, ShiftWrapperFixture,
                         TA_UT_SKIP_IF_DISTRIBUTED)

BOOST_AUTO_TEST_CASE(constructor) {
  BOOST_CHECK_NO_THROW(shift(t1));
  auto s = shift(t1);

  BOOST_CHECK_EQUAL(&s.get(), &t1);
  BOOST_CHECK_EQUAL(s.data(), t1.data());
  BOOST_CHECK_EQUAL(s.range(), t1.range());
}

BOOST_AUTO_TEST_CASE(shift_assignment_lhs) {
  BOOST_CHECK_NO_THROW(shift(t1) = t2);
  BOOST_CHECK_NE(t1.data(), t2.data());

  for (std::size_t i = 0ul; i < t1.size(); ++i) BOOST_CHECK_EQUAL(t1[i], t2[i]);
}

BOOST_AUTO_TEST_CASE(shift_assignment_rhs) {
  BOOST_CHECK_NO_THROW(t1 = shift(t2));
  BOOST_CHECK_NE(t1.data(), t2.data());

  for (std::size_t i = 0ul; i < t1.size(); ++i) BOOST_CHECK_EQUAL(t1[i], t2[i]);
}

BOOST_AUTO_TEST_SUITE_END()
