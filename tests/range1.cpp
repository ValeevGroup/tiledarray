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
}

BOOST_AUTO_TEST_CASE(constructors) {
  BOOST_CHECK_NO_THROW(Range1{});
  BOOST_CHECK_EQUAL(Range1{}.first, 0);
  BOOST_CHECK_EQUAL(Range1{}.second, 0);
  // decomposable via structure bindings;
  auto [first, second] = Range1{};
  BOOST_CHECK_EQUAL(first, Range1{}.first);
  BOOST_CHECK_EQUAL(second, Range1{}.second);

  BOOST_CHECK_NO_THROW(Range1{1});
  BOOST_CHECK_EQUAL(Range1{1}.first, 0);
  BOOST_CHECK_EQUAL(Range1{1}.second, 1);

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
  Range1 r{1, 10};
  BOOST_CHECK_NO_THROW(r.lobound());
  BOOST_CHECK_EQUAL(r.lobound(), 1);
  BOOST_CHECK_NO_THROW(r.upbound());
  BOOST_CHECK_EQUAL(r.upbound(), 10);
  BOOST_CHECK_NO_THROW(r.extent());
  BOOST_CHECK_EQUAL(r.extent(), 9);
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

BOOST_AUTO_TEST_CASE(comparison) {
  Range1 r1{0, 10};
  Range1 r2{0, 10};
  Range1 r3{1, 10};
  Range1 r4{0, 9};
  BOOST_CHECK(r1 == r1);
  BOOST_CHECK(r1 == r2);
  BOOST_CHECK(r1 != r3);
  BOOST_CHECK(r1 != r4);
}

BOOST_AUTO_TEST_CASE(serialization) {
  Range1 r{1, 10};

  std::size_t buf_size = sizeof(Range1);
  unsigned char* buf = new unsigned char[buf_size];
  madness::archive::BufferOutputArchive oar(buf, buf_size);
  oar& r;
  std::size_t nbyte = oar.size();
  oar.close();

  Range1 rs;
  madness::archive::BufferInputArchive iar(buf, nbyte);
  iar& rs;
  iar.close();

  delete[] buf;

  BOOST_CHECK(rs == r);
}

BOOST_AUTO_TEST_CASE(swap) {
  Range1 r{0, 10};
  Range1 empty_range;

  // swap with empty range
  BOOST_CHECK_NO_THROW(r.swap(empty_range));
  BOOST_CHECK(r == Range1{});

  // Check that empty_range contains the data of r.
  BOOST_CHECK(empty_range == Range1(0, 10));

  // Swap the data back
  BOOST_CHECK_NO_THROW(r.swap(empty_range));
  BOOST_CHECK(empty_range == Range1{});
  BOOST_CHECK(r == Range1(0, 10));
}

BOOST_AUTO_TEST_CASE(irange) {
  // to avoid ambiguity between test scope and fn
  BOOST_CHECK_NO_THROW(TA::irange(1));
  BOOST_CHECK(TA::irange(1) == Range1(1));

  BOOST_CHECK_NO_THROW(TA::irange(1, 2));
  BOOST_CHECK(TA::irange(1, 2) == Range1(1, 2));
}

BOOST_AUTO_TEST_SUITE_END()
