/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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

#include "TiledArray/elemental.h"
#include "unit_test_config.h"
#include "range_fixture.h"

struct ElemFixture : public TiledRangeFixture {
  ElemFixture():
    trange(dims.begin(), dims.begin() + 2),
    array(*GlobalFixture::world, trange),
    grid(elem::DefaultGrid().Comm()),
    matrix(grid)
  {}

  TiledRange trange;
  Array<int, 2> array;
  elem::Grid grid;
  elem::DistMatrix<int> matrix;
};

BOOST_FIXTURE_TEST_SUITE(elemental_suite, ElemFixture)

BOOST_AUTO_TEST_CASE(array_to_elem_test) {
  // Fill array with random data
  GlobalFixture::world->srand(27);
  for(Range::const_iterator it = array.range().begin(); it != array.range().end(); ++it) {
    Array<int, 2>::value_type tile(array.trange().make_tile_range(*it));
    for(Array<int, 2>::value_type::iterator tile_it = tile.begin(); tile_it != tile.end(); ++tile_it) {
      *tile_it = GlobalFixture::world->rand();
    }
    array.set(*it, tile);
  }

  // Convert the array to an elemental matrix
  BOOST_CHECK_NO_THROW(matrix = array_to_elem(array, grid));
  // Check dims
  BOOST_CHECK_EQUAL(matrix.Width(), array.trange().elements().size()[0]);
  BOOST_CHECK_EQUAL(matrix.Height(), array.trange().elements().size()[1]);
}

BOOST_AUTO_TEST_SUITE_END()
