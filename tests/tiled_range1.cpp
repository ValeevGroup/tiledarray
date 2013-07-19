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
#include "config.h"
#include "range_fixture.h"
#include "TiledArray/coordinates.h"
#include <sstream>

using namespace TiledArray;

BOOST_FIXTURE_TEST_SUITE( range1_suite, Range1Fixture )

BOOST_AUTO_TEST_CASE( range_accessor )
{
  BOOST_CHECK_EQUAL(tr1.tiles().first, tiles.first);
  BOOST_CHECK_EQUAL(tr1.tiles().second, tiles.second);
  BOOST_CHECK_EQUAL(tr1.elements().first, elements.first);
  BOOST_CHECK_EQUAL(tr1.elements().second, elements.second);

  // Check individual tiles
  for(std::size_t i = 0; i < a.size() - 1; ++i) {
    BOOST_CHECK_EQUAL(tr1.tile(i).first, a[i]);
    BOOST_CHECK_EQUAL(tr1.tile(i).second, a[i + 1]);
  }
}

BOOST_AUTO_TEST_CASE( range_info )
{
  BOOST_CHECK_EQUAL(tr1.tiles().first, 0ul);
  BOOST_CHECK_EQUAL(tr1.tiles().second, a.size() - 1);
  BOOST_CHECK_EQUAL(tr1.elements().first, 0ul);
  BOOST_CHECK_EQUAL(tr1.elements().second, a.back());
  for(std::size_t i = 0; i < a.size() - 1; ++i) {
    BOOST_CHECK_EQUAL(tr1.tile(i).first, a[i]);
    BOOST_CHECK_EQUAL(tr1.tile(i).second, a[i + 1]);
  }
}

BOOST_AUTO_TEST_CASE( constructor )
{
  // check default construction and range info.
  {
    BOOST_REQUIRE_NO_THROW(TiledRange1 r);
    TiledRange1 r;
    BOOST_CHECK_EQUAL(r.tiles().first, 0ul);
    BOOST_CHECK_EQUAL(r.tiles().second, 0ul);
    BOOST_CHECK_EQUAL(r.elements().first, 0ul);
    BOOST_CHECK_EQUAL(r.elements().second, 0ul);
#ifdef TA_EXCEPTION_ERROR
    BOOST_CHECK_THROW(r.tile(0), Exception);
#endif // TA_EXCEPTION_ERROR
  }

  // check construction with a iterators and the range info.
  {
    BOOST_REQUIRE_NO_THROW(TiledRange1 r(a.begin(), a.end()));
    TiledRange1 r(a.begin(), a.end());
    BOOST_CHECK_EQUAL(r.tiles().first, tiles.first);
    BOOST_CHECK_EQUAL(r.tiles().second, tiles.second);
    BOOST_CHECK_EQUAL(r.elements().first, elements.first);
    BOOST_CHECK_EQUAL(r.elements().second, elements.second);
    for(std::size_t i = 0; i < a.size() - 1; ++i) {
      BOOST_CHECK_EQUAL(r.tile(i).first, a[i]);
      BOOST_CHECK_EQUAL(r.tile(i).second, a[i + 1]);
    }
  }


  // check copy constructor
  {
    BOOST_REQUIRE_NO_THROW(TiledRange1 r(tr1));
    TiledRange1 r(tr1);
    BOOST_CHECK_EQUAL(r.tiles().first, tiles.first);
    BOOST_CHECK_EQUAL(r.tiles().second, tiles.second);
    BOOST_CHECK_EQUAL(r.elements().first, elements.first);
    BOOST_CHECK_EQUAL(r.elements().second, elements.second);
    for(std::size_t i = 0; i < a.size() - 1; ++i) {
      BOOST_CHECK_EQUAL(r.tile(i).first, a[i]);
      BOOST_CHECK_EQUAL(r.tile(i).second, a[i + 1]);
    }
  }

  // check construction with a with a tile offset.
  {
    BOOST_REQUIRE_NO_THROW(TiledRange1 r(a.begin(), a.end(), 2));
    TiledRange1 r(a.begin(), a.end(), 2);
    BOOST_CHECK_EQUAL(r.tiles().first, 2ul);
    BOOST_CHECK_EQUAL(r.tiles().second, 1 + a.size());
    BOOST_CHECK_EQUAL(r.elements().first, elements.first);
    BOOST_CHECK_EQUAL(r.elements().second, elements.second);
    for(std::size_t i = 0; i < a.size() - 1; ++i) {
      BOOST_CHECK_EQUAL(r.tile(i + 2).first, a[i]);
      BOOST_CHECK_EQUAL(r.tile(i + 2).second, a[i + 1]);
    }
  }


  // check construction with a with a element offset.
  {
    BOOST_REQUIRE_NO_THROW(TiledRange1 r(a.begin() + 1, a.end()));
    TiledRange1 r(a.begin() + 1, a.end());
    BOOST_CHECK_EQUAL(r.tiles().first, 0ul);
    BOOST_CHECK_EQUAL(r.tiles().second, a.size() - 2);
    BOOST_CHECK_EQUAL(r.elements().first, a[1]);
    BOOST_CHECK_EQUAL(r.elements().second, a.back());
    for(std::size_t i = 1; i < a.size() - 1; ++i) {
      BOOST_CHECK_EQUAL(r.tile(i - 1).first, a[i]);
      BOOST_CHECK_EQUAL(r.tile(i - 1).second, a[i + 1]);
    }
  }

  // Check that invalid input throws an exception.
  {
#ifndef NDEBUG
    std::vector<std::size_t> boundaries;
    BOOST_CHECK_THROW(TiledRange1 r(boundaries.begin(), boundaries.end()), Exception);
    BOOST_CHECK_THROW(TiledRange1 r(a.begin(), a.begin()), Exception);
    BOOST_CHECK_THROW(TiledRange1 r(a.begin(), a.begin() + 1), Exception);
    boundaries.push_back(2);
    boundaries.push_back(0);
    BOOST_CHECK_THROW(TiledRange1 r(boundaries.begin(), boundaries.end()), Exception);
#endif // NDEBUG
  }
}

BOOST_AUTO_TEST_CASE( ostream )
{
  std::stringstream stm;
  stm << "( tiles = [ " << 0 << ", " << a.size() - 1 <<
      " ), elements = [ " << a.front() << ", " << a.back() << " ) )";

  boost::test_tools::output_test_stream output;
  output << tr1;
  BOOST_CHECK( !output.is_empty( false ) );
  BOOST_CHECK( output.check_length( stm.str().size(), false ) );
  BOOST_CHECK( output.is_equal( stm.str().c_str() ) );
}

BOOST_AUTO_TEST_CASE( element2tile )
{
  // construct a map that should match the element to tile map for tr1.
  std::vector<std::size_t> e;
  for(std::size_t t = tr1.tiles().first; t < tr1.tiles().second; ++t)
    for(std::size_t i = tr1.tile(t).first; i < tr1.tile(t).second; ++i)
      e.push_back(t);

  // Construct a map that matches the internal element to tile map for tr1.
  std::vector<std::size_t> c;
  for(std::size_t i = tr1.elements().first; i < tr1.elements().second; ++i)
    c.push_back(tr1.element2tile(i));

  // Check that the expected and internal element to tile maps match.
  BOOST_CHECK_EQUAL_COLLECTIONS(c.begin(), c.end(), e.begin(), e.end());
}

BOOST_AUTO_TEST_CASE( comparison )
{
  TiledRange1 r1(tr1);
  BOOST_CHECK(r1 == tr1);     // check equality operator
  BOOST_CHECK(! (r1 != tr1)); // check not-equal operator
  TiledRange1(a.begin(), a.end(), 3).swap(r1);
  BOOST_CHECK(! (r1 == tr1)); // check for inequality with different start  point for tiles
  BOOST_CHECK(r1 != tr1);
  std::array<std::size_t, 6> a1 = a;
  a1[2] = 8;
  TiledRange1(a1.begin(), a1.end(), 0).swap(r1);
  BOOST_CHECK(! (r1 == tr1)); // check for inequality with different tile boundaries.
  BOOST_CHECK(r1 != tr1);
  a1[2] = 7;
  a1[4] = 50;
  TiledRange1(a1.begin(), a1.end() - 1, 0).swap(r1);
  BOOST_CHECK(! (r1 == tr1)); // check for inequality with different number of tiles.
  BOOST_CHECK(r1 != tr1);
}

BOOST_AUTO_TEST_CASE( iteration )
{
  // check for proper iteration functionality.
  std::size_t count = 0;
  for(TiledRange1::const_iterator it = tr1.begin(); it != tr1.end(); ++it, ++count) {
    BOOST_CHECK_EQUAL(it->first, a[count]);
    BOOST_CHECK_EQUAL(it->second, a[count + 1]);
  }
  BOOST_CHECK_EQUAL(count, tr1.tiles().second - tr1.tiles().first);
}

BOOST_AUTO_TEST_CASE( find )
{
  // check that find returns an iterator to the correct tile.
  BOOST_CHECK_EQUAL( tr1.find(tr1.tile(3).first + 1)->first, a[3]);
  BOOST_CHECK_EQUAL( tr1.find(tr1.tile(3).first + 1)->second, a[4]);

  // check that the iterator points to the end() iterator if the element is out of range.
  BOOST_CHECK( tr1.find(a.back() + 10) == tr1.end());

}

BOOST_AUTO_TEST_CASE( assignment )
{
  TiledRange1 r1;
  BOOST_CHECK_NE( r1, tr1);
  BOOST_CHECK_EQUAL((r1 = tr1), tr1); // check operator=
  BOOST_CHECK_EQUAL(r1, tr1);
}

BOOST_AUTO_TEST_SUITE_END()
