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
    grid(elem::DefaultGrid().Comm())
  {}

  TiledRange trange;
  Array<int, 2> array;
  elem::Grid grid;
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
  elem::DistMatrix<int> matrix(grid);
  BOOST_CHECK_NO_THROW(matrix = array_to_elem(array, grid));
  // Check dims
  BOOST_CHECK_EQUAL(matrix.Width(), array.trange().elements().size()[0]);
  BOOST_CHECK_EQUAL(matrix.Height(), array.trange().elements().size()[1]);

  //Check data
  for(std::size_t i = array.range().begin(); i != array.range().end(); ++i){
    madness::Future<Array<int,2>::value_type > tile = array.find(*i);
    for(std::size_t j = tile.get().range().begin(); j != tile.get().range().end();++j){
      BOOST_CHECK_EQUAL(tile.get()[*j], matrix.Get((*j)[0], (*j)[1]) );
    }
  }
}


BOOST_AUTO_TEST_CASE(elem_to_array_test) {
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
  elem::DistMatrix<int> matrix(grid);
  BOOST_CHECK_NO_THROW(matrix = array_to_elem(array, grid));
  // Check dims
  BOOST_CHECK_EQUAL(matrix.Width(), array.trange().elements().size()[0]);
  BOOST_CHECK_EQUAL(matrix.Height(), array.trange().elements().size()[1]);

  // Reassign elemental matrix to something else
  for(int i = 0; i < matrix.Width(); ++i){
    for(int j = 0; j < matrix.Height(); ++j){
      matrix.Set(i,j, i+j);
    }
  }

  // Copy matrix to TiledArray Array
  elem_to_array(array, matrix);
  array.get_world().gop.fence();

  //Check data
  for(std::size_t i = array.range().begin(); i != array.range().end(); ++i){
    madness::Future<Array<int,2>::value_type > tile = array.find(*i);
    for(std::size_t j = tile.get().range().begin(); j != tile.get().range().end();++j){
      BOOST_CHECK_EQUAL(tile.get()[*j], matrix.Get((*j)[0], (*j)[1]) );
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
