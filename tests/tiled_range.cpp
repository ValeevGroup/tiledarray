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
#include "range_fixture.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;

BOOST_FIXTURE_TEST_SUITE(tiled_range_suite, TiledRangeFixture,
                         TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(accessor) {
  BOOST_CHECK_EQUAL(tr.tiles_range(), tiles_range);
  BOOST_CHECK_EQUAL(tr.elements_range(), elements_range);
}

BOOST_AUTO_TEST_CASE(constructor) {
  // check default constructor
  {
    BOOST_REQUIRE_NO_THROW(TiledRange r0);
    TiledRange r0;
    std::vector<std::size_t> s0(3, 0);
    BOOST_CHECK(!r0.tiles_range());
    BOOST_CHECK(!r0.elements_range());
  }

  // check ranges constructor
  {
    BOOST_REQUIRE_NO_THROW(TiledRange r1(dims.begin(), dims.end()));
    TiledRange r1(dims.begin(), dims.end());
    BOOST_CHECK_EQUAL(r1.tiles_range(), tiles_range);
    BOOST_CHECK_EQUAL(r1.elements_range(), elements_range);
  }

  // construct with empty ranges
  {
    BOOST_REQUIRE_NO_THROW(TiledRange r1({dims[0], TiledRange1{}}));
    TiledRange r1{dims[0], TiledRange1{}};
    BOOST_CHECK_EQUAL(r1.tiles_range().area(), 0);
    BOOST_CHECK_EQUAL(r1.elements_range().area(), 0);
  }

  // construct with ranges containing empty tiles only
  {
    BOOST_REQUIRE_NO_THROW(TiledRange r1({dims[0], TiledRange1{1, 1, 1}}));
    TiledRange r1{dims[0], TiledRange1{1, 1, 1}};
    BOOST_CHECK_EQUAL(r1.tiles_range().area(), dims[0].tile_extent() * 2);
    BOOST_CHECK_EQUAL(r1.elements_range().area(), 0);
  }

  // check initializer list of initializer list constructor
  {
    TiledRange r1{
        {0, 2, 5, 10, 17, 28}, {0, 2, 5, 10, 17, 28}, {0, 2, 5, 10, 17, 28}};
    BOOST_CHECK_EQUAL(r1.tiles_range(), tiles_range);
    BOOST_CHECK_EQUAL(r1.elements_range(), elements_range);
  }

  // check range of trange1s constructor
  {
    std::vector trange1s(3, TiledRange1{0, 2, 5, 10, 17, 28});
    BOOST_REQUIRE_NO_THROW(TiledRange r1(trange1s));
    TiledRange r1(trange1s);
    BOOST_CHECK_EQUAL(r1.tiles_range(), tiles_range);
    BOOST_CHECK_EQUAL(r1.elements_range(), elements_range);
  }

  // check negative index range
#ifdef TA_SIGNED_1INDEX_TYPE
  {
    TiledRange r1{{-1, 0, 2, 5, 10, 17, 28},
                  {-1, 0, 2, 5, 10, 17, 28},
                  {-5, 0, 2, 5, 10, 17, 28}};
    BOOST_CHECK_EQUAL(r1.tiles_range(), Range({6, 6, 6}));
    BOOST_CHECK_EQUAL(r1.elements_range(), Range({-1, -1, -5}, {28, 28, 28}));
  }
#endif  // TA_SIGNED_1INDEX_TYPE

  // check copy constructor
  {
    BOOST_REQUIRE_NO_THROW(TiledRange r4(tr));
    TiledRange r4(tr);
    BOOST_CHECK_EQUAL(r4.tiles_range(), tr.tiles_range());
    BOOST_CHECK_EQUAL(r4.elements_range(), tr.elements_range());
  }
}

BOOST_AUTO_TEST_CASE(ostream) {
  std::stringstream stm;
  stm << "( tiles = " << tr.tiles_range()
      << ", elements = " << tr.elements_range() << " )";

  boost::test_tools::output_test_stream output;
  output << tr;
  BOOST_CHECK(!output.is_empty(false));
  BOOST_CHECK(output.check_length(stm.str().size(), false));
  BOOST_CHECK(output.is_equal(stm.str().c_str()));
}

BOOST_AUTO_TEST_CASE(comparison) {
  TiledRange r1{{0, 2, 4, 6, 8, 10}, {0, 2, 4, 6, 8, 10}};
  TiledRange r2{{0, 2, 4, 6, 8, 10}, {0, 2, 4, 6, 8, 10}};
  TiledRange r3{{0, 3, 6, 9, 12, 15}, {0, 3, 6, 9, 12, 15}};
  BOOST_CHECK(r1 == r2);     // check equality operator
  BOOST_CHECK(!(r1 != r2));  // check not-equal operator
  BOOST_CHECK(
      !(r1 == r3));  // check for inequality with different number of tiles.
  BOOST_CHECK(r1 != r3);
}

BOOST_AUTO_TEST_CASE(assignment) {
  TiledRange r1;

  // verify they are not equal before assignment.
  BOOST_CHECK_NE(r1, tr);

  // check that assignment returns itself.
  BOOST_CHECK_EQUAL((r1 = tr), tr);

  // check that assignment is valid.
  BOOST_CHECK_EQUAL(r1, tr);
}

BOOST_AUTO_TEST_CASE(permutation) {
  Permutation p({2, 0, 1});
  TiledRange r1 = p * tr;
  BOOST_CHECK_EQUAL(
      r1.tiles_range(),
      p * tr.tiles_range());  // check that tile data was permuted properly.
  BOOST_CHECK_EQUAL(r1.elements_range(),
                    p * tr.elements_range());  // check that element data was
                                               // permuted properly.

  TiledRange r2(tr);
  BOOST_CHECK_EQUAL((r2 *= p), r1);  // check that permutation returns itself.
  BOOST_CHECK_EQUAL(r2,
                    r1);  // check that the permutation was assigned correctly.
}

BOOST_AUTO_TEST_CASE(make_tiles_range) {
  tile_index start(GlobalFixture::dim);
  tile_index finish(GlobalFixture::dim);

  // iterate over all the tile indexes in the tiled range.
  TiledRange::ordinal_type i = 0;
  for (Range::const_iterator it = tr.tiles_range().begin();
       it != tr.tiles_range().end(); ++it, ++i) {
    // get the start and finish indexes of the current range.
    for (unsigned int d = 0; d < GlobalFixture::dim; ++d) {
      start[d] = a[(*it)[d]];
      finish[d] = a[(*it)[d] + 1];
    }

    // construct a range object that should match the range constructed by
    // TiledRange.
    TiledRange::range_type range(start, finish);

    // Get the two ranges to be tested.
    TiledRange::range_type range_index = tr.make_tile_range(*it);
    TiledRange::range_type range_ordinal = tr.make_tile_range(i);

    BOOST_CHECK_EQUAL(range_index, range);
    BOOST_CHECK_EQUAL(range_ordinal, range);
  }
}

BOOST_AUTO_TEST_SUITE_END()
