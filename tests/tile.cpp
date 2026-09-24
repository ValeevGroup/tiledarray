/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
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
 *  tile.cpp
 *  September 24, 2026
 *
 */

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_BTAS

#include <TiledArray/external/btas.h>
#include <TiledArray/tile.h>
#include "tiledarray.h"
#include "unit_test_config.h"

BOOST_AUTO_TEST_SUITE(tile_suite, TA_UT_LABEL_SERIAL)

// TA::Tile is meant to wrap deep-copy tensor types, such as btas::Tensor
using tile_type = TiledArray::Tile<btas::Tensor<double, TiledArray::Range>>;

BOOST_AUTO_TEST_CASE(null_tile) {
  tile_type null_tile;
  const tile_type& const_null_tile = null_tile;
  BOOST_CHECK(null_tile.empty());
  BOOST_CHECK_EQUAL(null_tile.use_count(), 0);
  // tensor() on a null tile must trigger TA_ASSERT
  BOOST_CHECK_TA_ASSERT(null_tile.tensor(), TiledArray::Exception);
  BOOST_CHECK_TA_ASSERT(const_null_tile.tensor(), TiledArray::Exception);
}

BOOST_AUTO_TEST_SUITE_END()

#endif  // TILEDARRAY_HAS_BTAS
