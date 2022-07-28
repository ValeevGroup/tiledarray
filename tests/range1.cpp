/*
 * This file is a part of TiledArray.
 * Copyright (C) 2022  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "TiledArray/range1.h"
#include "TiledArray/range.h"
#include "TiledArray/type_traits.h"

#include <boost/container/small_vector.hpp>
#include <vector>

#include "unit_test_config.h"

using namespace TiledArray;

BOOST_AUTO_TEST_SUITE(range1_suite, TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(traits) {
  static_assert(detail::is_integral_range_v<Range1>);
  static_assert(detail::is_integral_range_v<const Range1>);
  static_assert(detail::is_integral_pair_v<Range1>);
  static_assert(detail::is_integral_pair_v<const Range1>);
  static_assert(detail::is_gettable_pair_v<Range1>);
  static_assert(detail::is_gettable_pair_v<const Range1>);
  static_assert(detail::is_range_v<Range1>);
  static_assert(detail::is_range_v<const Range1>);
  static_assert(!detail::is_contiguous_range_v<
                Range1>);  // it's not a container! so not contiguous according
                           // to the defintion of std::contiguous_range
  static_assert(
      detail::is_gpair_range_v<boost::container::small_vector<Range1, 8>>);
  static_assert(detail::is_gpair_range_v<
                boost::container::small_vector<const Range1, 8>>);
  static_assert(detail::is_gpair_range_v<std::vector<Range1>>);
  static_assert(detail::is_gpair_range_v<std::vector<const Range1>>);
}

BOOST_AUTO_TEST_CASE(constructors) {
  BOOST_CHECK_NO_THROW(Range1{});
  BOOST_CHECK_EQUAL(Range1{}.first, 0);
  BOOST_CHECK_EQUAL(Range1{}.second, 0);
  // decomposable via structure bindings;
  auto [first, second] = Range1{};

  BOOST_CHECK_NO_THROW((Range1{1, 1}));
  BOOST_CHECK_NO_THROW(Range1(1, 1));
  BOOST_CHECK_EQUAL(Range1(1, 1).first, 1);
  BOOST_CHECK_EQUAL(Range1(1, 1).first, 1);

  BOOST_CHECK_NO_THROW((Range1{-11, 13}));
  BOOST_CHECK_EQUAL(Range1(-11, 13).first, -11);
  BOOST_CHECK_EQUAL(Range1(-11, 13).second, 13);

  Range1 rng{-11, 13};
  // copy
  BOOST_CHECK_NO_THROW((Range1{rng}));
  // move
  BOOST_CHECK_NO_THROW(Range1(Range1{-11, 13}));
}

BOOST_AUTO_TEST_CASE(accessors) {
  Range1 r{0, 10};
  BOOST_CHECK_NO_THROW(r.extent());
  BOOST_CHECK_EQUAL(r.extent(), 10);
}

BOOST_AUTO_TEST_CASE(iteration) {
  Range1 r{0, 10};

  BOOST_CHECK_NO_THROW(r.begin());
  BOOST_CHECK_NO_THROW(r.cbegin());
  BOOST_CHECK_NO_THROW(r.end());
  BOOST_CHECK_NO_THROW(r.cend());

  auto it = r.begin();
  BOOST_CHECK_EQUAL(*it, 0);
  BOOST_CHECK_NO_THROW(++it);
  BOOST_CHECK_EQUAL(*it, 1);
  BOOST_CHECK_NO_THROW(it++);
  BOOST_CHECK_EQUAL(*it, 2);
  BOOST_CHECK_NO_THROW(--it);
  BOOST_CHECK_EQUAL(*it, 1);
  BOOST_CHECK_NO_THROW(it--);
  BOOST_CHECK_EQUAL(*it, 0);
  BOOST_CHECK_NO_THROW(it += 2);
  BOOST_CHECK_EQUAL(*it, 2);
  BOOST_CHECK_NO_THROW(it -= 2);
  BOOST_CHECK_EQUAL(*it, 0);
  BOOST_CHECK(it == r.begin());

  auto end = r.end();
  BOOST_CHECK_EQUAL(*end, 10);
  BOOST_CHECK(it != end);
}

BOOST_AUTO_TEST_SUITE_END()
