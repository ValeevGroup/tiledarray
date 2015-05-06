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
 */

#include "TiledArray/tiled_range.h"
#include "tiledarray.h"
#include "unit_test_config.h"
#include "range_fixture.h"

using namespace TiledArray;

BOOST_FIXTURE_TEST_SUITE( tiled_range_suite, TiledRangeFixture )

BOOST_AUTO_TEST_CASE( accessor )
{
  BOOST_CHECK_EQUAL(tr.tiles(), tile_range);
  BOOST_CHECK_EQUAL(tr.elements(), element_range);
}

BOOST_AUTO_TEST_CASE( constructor )
{
  // check default constructor
  {
    BOOST_REQUIRE_NO_THROW(TiledRange r0);
    TiledRange r0;
    std::vector<std::size_t> s0(3,0);
    BOOST_CHECK_EQUAL(r0.tiles().size().size(), 0);
    BOOST_CHECK_EQUAL(r0.elements().size().size(), 0);
  }


  // check ranges constructor
  {
    BOOST_REQUIRE_NO_THROW(TiledRange r1(dims.begin(), dims.end()));
    TiledRange r1(dims.begin(), dims.end());
    BOOST_CHECK_EQUAL(r1.tiles(), tile_range);
    BOOST_CHECK_EQUAL(r1.elements(), element_range);

    std::vector<TiledRange1> dims2;
    for(std::size_t i = 0; i < GlobalFixture::dim; ++i)
      dims2.push_back(TiledRange1(a.begin(), a.end(), 1));
    TiledRange::range_type t2(p1, p6);
  }

  // check initializer list of initializer list constructor
  {
    TiledRange r1 { {0,2,5,10,17,28},
                    {0,2,5,10,17,28},
                    {0,2,5,10,17,28} };
    BOOST_CHECK_EQUAL(r1.tiles(), tile_range);
    BOOST_CHECK_EQUAL(r1.elements(), element_range);
  }

  // check ranges constructor w/ offset tile origin.
  {

    std::vector<TiledRange1> dims2;
    for(std::size_t i = 0; i < GlobalFixture::dim; ++i)
      dims2.push_back(TiledRange1(a.begin(), a.end(), 1));
    TiledRange::range_type t2(p1, p6);

    BOOST_REQUIRE_NO_THROW(TiledRange r2(dims2.begin(), dims2.end()));
    TiledRange r2(dims2.begin(), dims2.end());
    BOOST_CHECK_EQUAL(r2.tiles(), t2);
    BOOST_CHECK_EQUAL(r2.elements(), element_range);

  }

  // check ranges constructor w/ offset element origin.
  {
    std::array<std::size_t, 6> a3;
    std::copy(GlobalFixture::primes.begin(), GlobalFixture::primes.begin() + 6, a3.begin());
    std::vector<TiledRange1> dims3(GlobalFixture::dim, TiledRange1(a3.begin(), a3.end()));
    TiledRange::tile_range_type e3 = TiledRange::tile_range_type(
        tile_index(GlobalFixture::dim,a3[0]),
        tile_index(GlobalFixture::dim,a3[5]));

    BOOST_REQUIRE_NO_THROW(TiledRange r3(dims3.begin(), dims3.end()));
    TiledRange r3(dims3.begin(), dims3.end());
    BOOST_CHECK_EQUAL(r3.tiles(), tile_range);
    BOOST_CHECK_EQUAL(r3.elements(), e3);
  }

  // check copy constructor
  {
    BOOST_REQUIRE_NO_THROW(TiledRange r4(tr));
    TiledRange r4(tr);
    BOOST_CHECK_EQUAL(r4.tiles(), tr.tiles());
    BOOST_CHECK_EQUAL(r4.elements(), tr.elements());
  }
}

BOOST_AUTO_TEST_CASE( ostream )
{

  std::stringstream stm;
  stm << "( tiles = " << tr.tiles() <<
      ", elements = " << tr.elements() << " )";

  boost::test_tools::output_test_stream output;
  output << tr;
  BOOST_CHECK( !output.is_empty( false ) );
  BOOST_CHECK( output.check_length( stm.str().size(), false ) );
  BOOST_CHECK( output.is_equal( stm.str().c_str() ) );
}

BOOST_AUTO_TEST_CASE( comparison ) {
  TiledRange r1(tr);

  // check equality operator for identical ranges
  BOOST_CHECK(r1 == tr);
  // check that the inequality operator for identical ranges
  BOOST_CHECK(! (r1 != tr));

  std::vector<TiledRange1> dims2;
  for(std::size_t i = 0; i < GlobalFixture::dim; ++i)
    dims2.push_back(TiledRange1(a.begin(), a.end(), 1));
  TiledRange r2(dims2.begin(), dims2.end());

  // comparison w/ offset tile origin.
  BOOST_CHECK(! (r2 == tr));
  BOOST_CHECK(r2 != tr);

  std::array<std::size_t, 6> a3;
  std::copy(GlobalFixture::primes.begin(), GlobalFixture::primes.begin() + 6, a3.begin());
  std::vector<TiledRange1> dims3(GlobalFixture::dim,
      TiledRange1(a3.begin(), a3.end()));

  TiledRange r3(dims3.begin(), dims3.end());

  // comparison operators w/ offset elements and different tiling
  BOOST_CHECK(! (r3 == tr));
  BOOST_CHECK(r3 != tr);
}

BOOST_AUTO_TEST_CASE( assignment )
{
  TiledRange r1;

  // verify they are not equal before assignment.
  BOOST_CHECK_NE(r1, tr);

  // check that assignment returns itself.
  BOOST_CHECK_EQUAL((r1 = tr), tr);

  // check that assignment is valid.
  BOOST_CHECK_EQUAL(r1, tr);
}

BOOST_AUTO_TEST_CASE( permutation )
{
  Permutation p({2,0,1});
  TiledRange r1 = p ^ tr;
  BOOST_CHECK_EQUAL(r1.tiles(), p ^ tr.tiles()); // check that tile data was permuted properly.
  BOOST_CHECK_EQUAL(r1.elements(), p ^ tr.elements()); // check that element data was permuted properly.

  TiledRange r2(tr);
  BOOST_CHECK_EQUAL((r2 ^= p), r1); // check that permutation returns itself.
  BOOST_CHECK_EQUAL(r2, r1);// check that the permutation was assigned correctly.
}

BOOST_AUTO_TEST_CASE( make_tile_range )
{
  tile_index start(GlobalFixture::dim);
  tile_index finish(GlobalFixture::dim);

  // iterate over all the tile indexes in the tiled range.
  TiledRange::size_type i = 0;
  for(Range::const_iterator it = tr.tiles().begin(); it != tr.tiles().end(); ++it, ++i) {
    // get the start and finish indexes of the current range.
    for(unsigned int d = 0; d < GlobalFixture::dim; ++d) {
      start[d] = a[ (*it)[d] ];
      finish[d] = a[ (*it)[d] + 1 ];
    }

    // construct a range object that should match the range constructed by TiledRange.
    TiledRange::tile_range_type range(start, finish);

    // Get the two ranges to be tested.
    TiledRange::tile_range_type range_index = tr.make_tile_range(*it);
    TiledRange::tile_range_type range_ordinal = tr.make_tile_range(i);

    BOOST_CHECK_EQUAL(range_index, range);
    BOOST_CHECK_EQUAL(range_ordinal, range);
  }
}

BOOST_AUTO_TEST_SUITE_END()

