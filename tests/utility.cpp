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
 *  utility.cpp
 *  Oct 20, 2013
 *
 */

#include "TiledArray/utility.h"
#include "unit_test_config.h"
#include "TiledArray/size_array.h"

using TiledArray::detail::begin;
using TiledArray::detail::cbegin;
using TiledArray::detail::end;
using TiledArray::detail::cend;
using TiledArray::detail::size;

struct UtilityFixture {

  UtilityFixture() { }

  ~UtilityFixture() { }

}; // UtilityFixture

BOOST_FIXTURE_TEST_SUITE( utility_suite, UtilityFixture )

BOOST_AUTO_TEST_CASE( vector )
{
  std::vector<int> array(10, 1);
  const std::vector<int>& carray = array;

  // Check begin() and cbegin()
  BOOST_CHECK(TiledArray::detail::begin(array) == array.begin());
  BOOST_CHECK(TiledArray::detail::begin(carray) == carray.begin());
  BOOST_CHECK(TiledArray::detail::cbegin(array) == carray.begin());

  // Check end() and cend()
  BOOST_CHECK(TiledArray::detail::end(array) == array.end());
  BOOST_CHECK(TiledArray::detail::end(carray) == carray.end());
  BOOST_CHECK(TiledArray::detail::cend(array) == carray.end());

  // Check size()
  BOOST_CHECK_EQUAL(TiledArray::detail::size(array), array.size());
}

BOOST_AUTO_TEST_CASE( array )
{
  std::array<int, 10> array = {{ 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 }};
  const std::array<int, 10>& carray = array;

  // Check begin() and cbegin()
  BOOST_CHECK(TiledArray::detail::begin(array) == array.begin());
  BOOST_CHECK(TiledArray::detail::begin(carray) == carray.begin());
  BOOST_CHECK(TiledArray::detail::cbegin(array) == carray.begin());

  // Check end() and cend()
  BOOST_CHECK(TiledArray::detail::end(array) == array.end());
  BOOST_CHECK(TiledArray::detail::end(carray) == carray.end());
  BOOST_CHECK(TiledArray::detail::cend(array) == carray.end());

  // Check size()
  BOOST_CHECK_EQUAL(TiledArray::detail::size(array), array.size());
}

BOOST_AUTO_TEST_CASE( c_array )
{
  int array[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
  const int (& carray)[10] = array;

  // Check begin() and cbegin()
  BOOST_CHECK_EQUAL(TiledArray::detail::begin(array), static_cast<int*>(array));
  BOOST_CHECK_EQUAL(TiledArray::detail::begin(carray), static_cast<const int*>(array));
  BOOST_CHECK_EQUAL(TiledArray::detail::cbegin(array), static_cast<const int*>(array));

  // Check end() and cend()
  BOOST_CHECK_EQUAL(TiledArray::detail::end(array), static_cast<int*>(array + 10));
  BOOST_CHECK_EQUAL(TiledArray::detail::end(carray), static_cast<const int*>(array + 10));
  BOOST_CHECK_EQUAL(TiledArray::detail::cend(array), static_cast<const int*>(array + 10));

  // Check size()
  BOOST_CHECK_EQUAL(TiledArray::detail::size(array), 10);
}

BOOST_AUTO_TEST_CASE( size_array )
{
  int buffer[10] = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
  TiledArray::detail::SizeArray<int> array(buffer, buffer + 10);
  const TiledArray::detail::SizeArray<int>& carray = array;

  // Check begin() and cbegin()
  BOOST_CHECK(TiledArray::detail::begin(array) == array.begin());
  BOOST_CHECK(TiledArray::detail::begin(carray) == carray.begin());
  BOOST_CHECK(TiledArray::detail::cbegin(array) == carray.begin());

  // Check end() and cend()
  BOOST_CHECK(TiledArray::detail::end(array) == array.end());
  BOOST_CHECK(TiledArray::detail::end(carray) == carray.end());
  BOOST_CHECK(TiledArray::detail::cend(array) == carray.end());

  // Check size()
  BOOST_CHECK_EQUAL(TiledArray::detail::size(array), array.size());
}

BOOST_AUTO_TEST_SUITE_END()
