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
#include "TiledArray/size_array.h"
#include "unit_test_config.h"

using std::size;

struct UtilityFixture {
  UtilityFixture() {}

  ~UtilityFixture() {}

};  // UtilityFixture

BOOST_FIXTURE_TEST_SUITE(utility_suite, UtilityFixture,
                         TA_UT_SKIP_IF_DISTRIBUTED)

BOOST_AUTO_TEST_CASE(vector) {
  std::vector<int> array(10, 1);

  // Check size()
  BOOST_CHECK_EQUAL(std::size(array), array.size());
}

BOOST_AUTO_TEST_CASE(array) {
  std::array<int, 10> array = {{1, 1, 1, 1, 1, 1, 1, 1, 1, 1}};

  // Check size()
  BOOST_CHECK_EQUAL(std::size(array), array.size());
}

BOOST_AUTO_TEST_CASE(c_array) {
  int array[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

  // Check size()
  BOOST_CHECK_EQUAL(std::size(array), 10);
}

BOOST_AUTO_TEST_CASE(size_array) {
  int buffer[10] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  TiledArray::detail::SizeArray<int> array(buffer, buffer + 10);

  // Check size()
  BOOST_CHECK_EQUAL(std::size(array), array.size());
}

BOOST_AUTO_TEST_SUITE_END()
