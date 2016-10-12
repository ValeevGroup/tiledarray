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

#include "TiledArray/config.h"

#ifdef TILEDARRAY_HAS_ELEMENTAL

#ifdef HAVE_ELEMENTAL_H

#include "TiledArray/elemental.h"
#include "unit_test_config.h"
#include "range_fixture.h"

struct ElemFixture : public TiledRangeFixture {
  ElemFixture():
    trange(dims.begin(), dims.begin() + 2),
    array(*GlobalFixture::world, trange),
    grid(elem::DefaultGrid().Comm())
  {}

  TiledRange trange;
  Array<int, 2> array;
  elem::Grid grid;
};

void check_equal(Array<int,2> &array, elem::DistMatrix<int> &matrix){
  elem::DistMatrix<int, elem::STAR, elem::STAR> rep_matrix(matrix);
  for(Array<int,2>::range_type::const_iterator it = array.range().begin();
                                               it != array.range().end();
    ++it){
      Future<Array<int,2>::value_type> tile = array.find(*it);
      for(Array<int,2>::value_type::range_type::const_iterator it = tile.get().range().begin();
                                             it != tile.get().range().end();
          ++it){
            BOOST_CHECK_EQUAL(tile.get()[*it], rep_matrix.Get((*it)[0], (*it)[1]));
      }
  }
}

BOOST_FIXTURE_TEST_SUITE(elemental_suite, ElemFixture)

BOOST_AUTO_TEST_CASE(array_to_elem_test) {
  GlobalFixture::world->gop.fence();

  // Fill array with random data
  GlobalFixture::world->srand(27);
  for(Array<int,2>::iterator it = array.begin(); it != array.end(); ++it) {
    Array<int, 2>::value_type tile(it.range());
    for(auto& v : tile) {
      v = GlobalFixture::world->rand();
    }
    *it = tile;
  }

  // Convert the array to an elemental matrix
  elem::DistMatrix<int> matrix(grid);
  BOOST_CHECK_NO_THROW(matrix = array_to_elem(array, grid));
  // Check dims
  BOOST_CHECK_EQUAL(matrix.Width(), array.trange().elements_range().extent_data()[0]);
  BOOST_CHECK_EQUAL(matrix.Height(), array.trange().elements_range().extent_data()[1]);

  check_equal(array, matrix);
  GlobalFixture::world->gop.fence();
}


BOOST_AUTO_TEST_CASE(elem_to_array_test) {
  // Fill array with random data
  GlobalFixture::world->srand(27);
  for(Array<int,2>::iterator it = array.begin(); it != array.end(); ++it) {
    Array<int, 2>::value_type tile(it.range());
    for(Array<int, 2>::value_type::iterator tile_it = tile.begin(); tile_it != tile.end(); ++tile_it) {
      *tile_it = GlobalFixture::world->rand();
    }
    *it = tile;
  }

  // Convert the array to an elemental matrix
  elem::DistMatrix<int> matrix(grid);
  BOOST_CHECK_NO_THROW(matrix = array_to_elem(array, grid));
  // Check dims
  BOOST_CHECK_EQUAL(matrix.Width(), array.trange().elements_range().extent_data()[0]);
  BOOST_CHECK_EQUAL(matrix.Height(), array.trange().elements_range().extent_data()[1]);

  // Re-fill elemental matrix with other random values
  for(int i = 0; i < matrix.Width(); ++i){
    for(int j = 0; j < matrix.Height(); ++j){
      matrix.Set(i,j, GlobalFixture::world->rand() );
    }
  }

  // Copy matrix to TiledArray Array
  elem_to_array(array, matrix);
  array.world().gop.fence();

  check_equal(array, matrix);
}

BOOST_AUTO_TEST_SUITE_END()

#else // HAVE_ELEMENTAL_H

# warning "TA<->Elemental conversions have not been reimplemented for recent Elemental API; check back soon"

#endif // HAVE_EL_H

#endif // TILEDARRAY_HAS_ELEMENTAL
