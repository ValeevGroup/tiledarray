/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
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

#include "TiledArray/tiled_range1.h"
#include <sstream>
#include "TiledArray/utility.h"
#include "range_fixture.h"
#include "unit_test_config.h"

using namespace TiledArray;

BOOST_FIXTURE_TEST_SUITE(tiled_range1_suite, Range1Fixture, TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(range_accessor) {
  BOOST_CHECK_EQUAL(tr1.tiles_range().first, tiles.first);
  BOOST_CHECK_EQUAL(tr1.tiles_range().second, tiles.second);
  BOOST_CHECK_EQUAL(tr1.elements_range().first, elements.first);
  BOOST_CHECK_EQUAL(tr1.elements_range().second, elements.second);

  // Check individual tiles
  for (std::size_t i = 0; i < a.size() - 1; ++i) {
    BOOST_CHECK_EQUAL(tr1.tile(i).first, a[i]);
    BOOST_CHECK_EQUAL(tr1.tile(i).second, a[i + 1]);
  }
}

BOOST_AUTO_TEST_CASE(range_info) {
  BOOST_CHECK_EQUAL(tr1.tiles_range().first, 0ul);
  BOOST_CHECK_EQUAL(tr1.tiles_range().second, a.size() - 1);
  BOOST_CHECK_EQUAL(tr1.elements_range().first, 0ul);
  BOOST_CHECK_EQUAL(tr1.elements_range().second, a.back());
  for (std::size_t i = 0; i < a.size() - 1; ++i) {
    BOOST_CHECK_EQUAL(tr1.tile(i).first, a[i]);
    BOOST_CHECK_EQUAL(tr1.tile(i).second, a[i + 1]);
  }
}

BOOST_AUTO_TEST_CASE(constructor) {
  // check default construction and range info.
  {
    BOOST_REQUIRE_NO_THROW(TiledRange1 r);
    TiledRange1 r;
    BOOST_CHECK_EQUAL(r.tiles_range().first, 0ul);
    BOOST_CHECK_EQUAL(r.tiles_range().second, 0ul);
    BOOST_CHECK_EQUAL(r.elements_range().first, 0ul);
    BOOST_CHECK_EQUAL(r.elements_range().second, 0ul);
    BOOST_CHECK_TA_ASSERT(r.tile(0), Exception);
  }

  // check construction with single tile boundary (hence zero tiles)
  {
    {
      BOOST_REQUIRE_NO_THROW(TiledRange1 r(0));
      TiledRange1 r(0);
      BOOST_CHECK_EQUAL(r, TiledRange1{});
    }
    {
      BOOST_REQUIRE_NO_THROW(TiledRange1 r(1));
      TiledRange1 r(1);
      BOOST_CHECK_NE(r, TiledRange1{});
      BOOST_CHECK_EQUAL(r.tiles_range().first, 0);
      BOOST_CHECK_EQUAL(r.tiles_range().second, 0);
      BOOST_CHECK_EQUAL(r.elements_range().first, 1);
      BOOST_CHECK_EQUAL(r.elements_range().second, 1);
      BOOST_CHECK_TA_ASSERT(r.tile(0), Exception);
    }
  }

  // check construction with a iterators and the range info.
  {
    BOOST_REQUIRE_NO_THROW(TiledRange1 r(a.begin(), a.end()));
    TiledRange1 r(a.begin(), a.end());
    BOOST_CHECK_EQUAL(r.tiles_range().first, tiles.first);
    BOOST_CHECK_EQUAL(r.tiles_range().second, tiles.second);
    BOOST_CHECK_EQUAL(r.elements_range().first, elements.first);
    BOOST_CHECK_EQUAL(r.elements_range().second, elements.second);
    for (std::size_t i = 0; i < a.size() - 1; ++i) {
      BOOST_CHECK_EQUAL(r.tile(i).first, a[i]);
      BOOST_CHECK_EQUAL(r.tile(i).second, a[i + 1]);
    }
  }

  // check variadic constructor using a's tile boundaries. NOTE: a is a runtime
  // object, must manually specify sizes here
  {
    BOOST_REQUIRE_NO_THROW(TiledRange1 r(0, 1, 2, 3, 4, 5));
    if (Range1Fixture::ntiles == 5) {
      TiledRange1 r(0, 2, 5, 10, 17, 28);
      BOOST_CHECK_EQUAL(r.tiles_range().first, tiles.first);
      BOOST_CHECK_EQUAL(r.tiles_range().second, tiles.second);
      BOOST_CHECK_EQUAL(r.elements_range().first, elements.first);
      BOOST_CHECK_EQUAL(r.elements_range().second, elements.second);
      for (std::size_t i = 0; i < a.size() - 1; ++i) {
        BOOST_CHECK_EQUAL(r.tile(i).first, a[i]);
        BOOST_CHECK_EQUAL(r.tile(i).second, a[i + 1]);
      }
    }
  }

  // check initializer list constructor using a's tile boundaries. NOTE: a is a
  // runtime object, must manually specify sizes here
  {
    if (Range1Fixture::ntiles == 5) {
      TiledRange1 r{0, 2, 5, 10, 17, 28};
      BOOST_CHECK_EQUAL(r.tiles_range().first, tiles.first);
      BOOST_CHECK_EQUAL(r.tiles_range().second, tiles.second);
      BOOST_CHECK_EQUAL(r.elements_range().first, elements.first);
      BOOST_CHECK_EQUAL(r.elements_range().second, elements.second);
      for (std::size_t i = 0; i < a.size() - 1; ++i) {
        BOOST_CHECK_EQUAL(r.tile(i).first, a[i]);
        BOOST_CHECK_EQUAL(r.tile(i).second, a[i + 1]);
      }
    }
  }

  // check constructor using range of tile boundaries.
  {
    if (Range1Fixture::ntiles == 5) {
      TiledRange1 r(a);
      BOOST_CHECK_EQUAL(r.tiles_range().first, tiles.first);
      BOOST_CHECK_EQUAL(r.tiles_range().second, tiles.second);
      BOOST_CHECK_EQUAL(r.elements_range().first, elements.first);
      BOOST_CHECK_EQUAL(r.elements_range().second, elements.second);
      for (std::size_t i = 0; i < a.size() - 1; ++i) {
        BOOST_CHECK_EQUAL(r.tile(i).first, a[i]);
        BOOST_CHECK_EQUAL(r.tile(i).second, a[i + 1]);
      }
    }
  }

  // check construction with negative index values
#ifdef TA_SIGNED_1INDEX_TYPE
  {
    TiledRange1 r{-1, 0, 2, 5, 10, 17, 28};
    BOOST_CHECK_EQUAL(r.tiles_range().first, 0);
    BOOST_CHECK_EQUAL(r.tiles_range().second, 6);
    BOOST_CHECK_EQUAL(r.elements_range().first, -1);
    BOOST_CHECK_EQUAL(r.elements_range().second, 28);
  }
#else   // TA_SIGNED_1INDEX_TYPE
  BOOST_CHECK_TA_ASSERT(TiledRange1 r({-1, 0, 2, 5, 10, 17, 28}),
                        TiledArray::Exception);
#endif  // TA_SIGNED_1INDEX_TYPE

  // check copy constructor
  {
    BOOST_REQUIRE_NO_THROW(TiledRange1 r(tr1));
    TiledRange1 r(tr1);
    BOOST_CHECK_EQUAL(r.tiles_range().first, tiles.first);
    BOOST_CHECK_EQUAL(r.tiles_range().second, tiles.second);
    BOOST_CHECK_EQUAL(r.elements_range().first, elements.first);
    BOOST_CHECK_EQUAL(r.elements_range().second, elements.second);
    for (std::size_t i = 0; i < a.size() - 1; ++i) {
      BOOST_CHECK_EQUAL(r.tile(i).first, a[i]);
      BOOST_CHECK_EQUAL(r.tile(i).second, a[i + 1]);
    }
  }

  // check construction with element range that does not start at 0.
  {
    BOOST_REQUIRE_NO_THROW(TiledRange1 r(a.begin() + 1, a.end()));
    TiledRange1 r(a.begin() + 1, a.end());
    BOOST_CHECK_EQUAL(r.tiles_range().first, 0ul);
    BOOST_CHECK_EQUAL(r.tiles_range().second, a.size() - 2);
    BOOST_CHECK_EQUAL(r.elements_range().first, a[1]);
    BOOST_CHECK_EQUAL(r.elements_range().second, a.back());
    for (std::size_t i = 1; i < a.size() - 1; ++i) {
      BOOST_CHECK_EQUAL(r.tile(i - 1).first, a[i]);
      BOOST_CHECK_EQUAL(r.tile(i - 1).second, a[i + 1]);
    }
  }

  // corner cases
  {
    // range with 1 empty tile
    {
      TiledRange1 r{0, 0};
      BOOST_CHECK_EQUAL(r.tiles_range().first, 0);
      BOOST_CHECK_EQUAL(r.tiles_range().second, 1);
      BOOST_CHECK_EQUAL(r.elements_range().first, 0);
      BOOST_CHECK_EQUAL(r.elements_range().second, 0);
      BOOST_CHECK(r.tile(0) == Range1(0, 0));
    }
    // range with some empty tiles
    {
      TiledRange1 r{1, 3, 3, 5, 5};
      BOOST_CHECK_EQUAL(r.tiles_range().first, 0);
      BOOST_CHECK_EQUAL(r.tiles_range().second, 4);
      BOOST_CHECK_EQUAL(r.elements_range().first, 1);
      BOOST_CHECK_EQUAL(r.elements_range().second, 5);
      // test tiles
      BOOST_CHECK(r.tile(0) == Range1(1, 3));
      BOOST_CHECK(r.tile(1) == Range1(3, 3));
      BOOST_CHECK(r.tile(2) == Range1(3, 5));
      BOOST_CHECK(r.tile(3) == Range1(5, 5));
    }
  }

  // Check that invalid input throws an exception.
#ifndef NDEBUG
  {
    std::vector<std::size_t> boundaries;
    BOOST_CHECK_TA_ASSERT(TiledRange1 r(boundaries.begin(), boundaries.end()),
                          Exception);
    BOOST_CHECK_TA_ASSERT(TiledRange1 r(a.begin(), a.begin()), Exception);
    BOOST_CHECK_NO_THROW(TiledRange1 r(a.begin(), a.begin() + 1));
    boundaries.push_back(2);
    boundaries.push_back(0);
    BOOST_CHECK_TA_ASSERT(TiledRange1 r(boundaries.begin(), boundaries.end()),
                          Exception);
  }
#endif
}

BOOST_AUTO_TEST_CASE(ostream) {
  std::stringstream stm;
  stm << "( tiles = [ " << 0 << ", " << a.size() - 1 << " ), elements = [ "
      << a.front() << ", " << a.back() << " ) )";

  boost::test_tools::output_test_stream output;
  output << tr1;
  BOOST_CHECK(!output.is_empty(false));
  BOOST_CHECK(output.check_length(stm.str().size(), false));
  BOOST_CHECK(output.is_equal(stm.str().c_str()));
}

BOOST_AUTO_TEST_CASE(element_to_tile) {
  // construct a map that should match the element to tile map for tr1.
  std::vector<std::size_t> e;
  for (auto t = tr1.tiles_range().first; t < tr1.tiles_range().second; ++t)
    for (auto i = tr1.tile(t).first; i < tr1.tile(t).second; ++i)
      e.push_back(t);

  // Construct a map that matches the internal element to tile map for tr1.
  std::vector<std::size_t> c;
  for (auto i = tr1.elements_range().first; i < tr1.elements_range().second;
       ++i)
    c.push_back(tr1.element_to_tile(i));

  // Check that the expected and internal element to tile maps match.
  BOOST_CHECK_EQUAL_COLLECTIONS(c.begin(), c.end(), e.begin(), e.end());

  // corner case: empty tiles
  {
    // range with some empty tiles
    {
      TiledRange1 r{1, 3, 3, 5, 5};
      BOOST_CHECK_TA_ASSERT(r.element_to_tile(0), Exception);
      BOOST_CHECK_EQUAL(r.element_to_tile(1), 0);
      BOOST_CHECK_EQUAL(r.element_to_tile(2), 0);
      BOOST_CHECK_EQUAL(r.element_to_tile(3), 2);
      BOOST_CHECK_EQUAL(r.element_to_tile(4), 2);
      BOOST_CHECK_TA_ASSERT(r.element_to_tile(5), Exception);
    }
  }
}

BOOST_AUTO_TEST_CASE(comparison) {
  TiledRange1 r1{1, 2, 4, 6, 8, 10};
  TiledRange1 r2{1, 2, 4, 6, 8, 10};
  TiledRange1 r3{1, 3, 6, 9, 12, 15};
  BOOST_CHECK(r1 == r2);     // check equality operator
  BOOST_CHECK(!(r1 != r2));  // check not-equal operator
  BOOST_CHECK(
      !(r1 == r3));  // check for inequality with different number of tiles.
  BOOST_CHECK(r1 != r3);
}

BOOST_AUTO_TEST_CASE(congruency) {
  TiledRange1 r1{1, 2, 4, 6, 8, 10};
  TiledRange1 r2{2, 3, 5, 7, 9, 11};
  BOOST_CHECK(is_congruent(r1, r2));
}

BOOST_AUTO_TEST_CASE(iteration) {
  // check for proper iteration functionality.
  std::size_t count = 0;
  for (TiledRange1::const_iterator it = tr1.begin(); it != tr1.end();
       ++it, ++count) {
    BOOST_CHECK_EQUAL(it->first, a[count]);
    BOOST_CHECK_EQUAL(it->second, a[count + 1]);
  }
  BOOST_CHECK_EQUAL(count, tr1.tiles_range().second - tr1.tiles_range().first);
}

BOOST_AUTO_TEST_CASE(find) {
  // check that find returns an iterator to the correct tile.
  BOOST_CHECK_EQUAL(tr1.find(tr1.tile(3).first + 1)->first, a[3]);
  BOOST_CHECK_EQUAL(tr1.find(tr1.tile(3).first + 1)->second, a[4]);

  // check that the iterator points to the end() iterator if the element is out
  // of range.
  BOOST_CHECK(tr1.find(a.back() + 10) == tr1.end());
}

BOOST_AUTO_TEST_CASE(assignment) {
  TiledRange1 r1;
  BOOST_CHECK_NE(r1, tr1);
  BOOST_CHECK_EQUAL((r1 = tr1), tr1);  // check operator=
  BOOST_CHECK_EQUAL(r1, tr1);
}

BOOST_AUTO_TEST_CASE(concatenation) {
  TiledRange1 r0;
  TiledRange1 r1{1, 3, 7, 9};
  TiledRange1 r2{0, 3, 4, 5};
  BOOST_CHECK(concat(r0, r0) == r0);
  BOOST_CHECK(concat(r1, r0) == r1);
  BOOST_CHECK(concat(r0, r1) == r1);
  BOOST_CHECK(concat(r1, r2) == (TiledRange1{1, 3, 7, 9, 12, 13, 14}));
  BOOST_CHECK(concat(r2, r1) == (TiledRange1{0, 3, 4, 5, 7, 11, 13}));
}

BOOST_AUTO_TEST_CASE(make_uniform) {
  BOOST_REQUIRE_NO_THROW(TiledRange1::make_uniform(Range1{1, 1}, 0));
  BOOST_CHECK(TiledRange1::make_uniform(Range1{1, 1}, 0) == TiledRange1{});
  BOOST_REQUIRE_NO_THROW(TiledRange1::make_uniform(Range1{3, 6}, 10));
  BOOST_CHECK(TiledRange1::make_uniform(Range1{3, 6}, 10) ==
              (TiledRange1{3, 6}));
  BOOST_REQUIRE_NO_THROW(TiledRange1::make_uniform(Range1{10, 60}, 10));
  BOOST_CHECK(TiledRange1::make_uniform(Range1{10, 60}, 10) ==
              (TiledRange1{10, 20, 30, 40, 50, 60}));
  BOOST_REQUIRE_NO_THROW(TiledRange1::make_uniform(Range1{10, 65}, 10));
  BOOST_CHECK(TiledRange1::make_uniform(Range1{10, 65}, 10) ==
              (TiledRange1{10, 20, 29, 38, 47, 56, 65}));
  BOOST_REQUIRE_NO_THROW(TiledRange1::make_uniform(Range1{10, 69}, 10));
  BOOST_CHECK(TiledRange1::make_uniform(Range1{10, 69}, 10) ==
              (TiledRange1{10, 20, 30, 40, 50, 60, 69}));

  BOOST_REQUIRE_NO_THROW(TiledRange1::make_uniform(0, 0));
  BOOST_CHECK(TiledRange1::make_uniform(0, 0) == TiledRange1{});
  BOOST_REQUIRE_NO_THROW(TiledRange1::make_uniform(0, 1));
  BOOST_CHECK(TiledRange1::make_uniform(0, 1) == TiledRange1{});
  BOOST_REQUIRE_NO_THROW(TiledRange1::make_uniform(3, 10));
  BOOST_CHECK(TiledRange1::make_uniform(3, 10) == (TiledRange1{0, 3}));
  BOOST_REQUIRE_NO_THROW(TiledRange1::make_uniform(50, 10));
  BOOST_CHECK(TiledRange1::make_uniform(50, 10) ==
              (TiledRange1{0, 10, 20, 30, 40, 50}));
  BOOST_REQUIRE_NO_THROW(TiledRange1::make_uniform(55, 10));
  BOOST_CHECK(TiledRange1::make_uniform(55, 10) ==
              (TiledRange1{0, 10, 19, 28, 37, 46, 55}));
  BOOST_REQUIRE_NO_THROW(TiledRange1::make_uniform(59, 10));
  BOOST_CHECK(TiledRange1::make_uniform(59, 10) ==
              (TiledRange1{0, 10, 20, 30, 40, 50, 59}));
}

BOOST_AUTO_TEST_SUITE_END()
