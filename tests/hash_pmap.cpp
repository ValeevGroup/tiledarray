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

#include "TiledArray/pmap/hash_pmap.h"
#include "global_fixture.h"
#include "unit_test_config.h"

using namespace TiledArray;

struct HashPmapFixture {
  constexpr static std::size_t max_ntiles = 10ul;
};

// =============================================================================
// BlockdPmap Test Suite

BOOST_FIXTURE_TEST_SUITE(hash_pmap_suite, HashPmapFixture)

BOOST_AUTO_TEST_CASE(constructor) {
  for (std::size_t tiles = 1ul; tiles < max_ntiles; ++tiles) {
    BOOST_REQUIRE_NO_THROW(
        TiledArray::detail::HashPmap pmap(*GlobalFixture::world, tiles));
    TiledArray::detail::HashPmap pmap(*GlobalFixture::world, tiles);
    BOOST_CHECK_EQUAL(pmap.rank(), GlobalFixture::world->rank());
    BOOST_CHECK_EQUAL(pmap.procs(), GlobalFixture::world->size());
    BOOST_CHECK_EQUAL(pmap.size(), tiles);
  }
}

BOOST_AUTO_TEST_CASE(owner) {
  const std::size_t rank = GlobalFixture::world->rank();
  const std::size_t size = GlobalFixture::world->size();

  ProcessID* p_owner = new ProcessID[size];

  // Check various pmap sizes
  for (std::size_t tiles = 1ul; tiles < max_ntiles; ++tiles) {
    TiledArray::detail::HashPmap pmap(*GlobalFixture::world, tiles);

    for (std::size_t tile = 0; tile < tiles; ++tile) {
      std::fill_n(p_owner, size, 0);
      p_owner[rank] = pmap.owner(tile);
      // check that the value is in range
      BOOST_CHECK_LT(p_owner[rank], size);
      GlobalFixture::world->gop.sum(p_owner, size);

      // Make sure everyone agrees on who owns what.
      for (std::size_t p = 0ul; p < size; ++p)
        BOOST_CHECK_EQUAL(p_owner[p], p_owner[rank]);
    }
  }

  delete[] p_owner;
}

BOOST_AUTO_TEST_CASE(local_size) {
  TiledArray::detail::HashPmap pmap(*GlobalFixture::world, 100);
  BOOST_CHECK(!pmap.known_local_size());
}

BOOST_AUTO_TEST_CASE(local_group) {
  ProcessID tile_owners[100];

  for (std::size_t tiles = 1ul; tiles < max_ntiles; ++tiles) {
    TiledArray::detail::HashPmap pmap(*GlobalFixture::world, tiles);

    // Check that all local elements map to this rank
    for (detail::HashPmap::const_iterator it = pmap.begin(); it != pmap.end();
         ++it) {
      BOOST_CHECK_EQUAL(pmap.owner(*it), GlobalFixture::world->rank());
    }

    std::fill_n(tile_owners, tiles, 0);
    for (detail::HashPmap::const_iterator it = pmap.begin(); it != pmap.end();
         ++it) {
      tile_owners[*it] += GlobalFixture::world->rank();
    }

    GlobalFixture::world->gop.sum(tile_owners, tiles);
    for (std::size_t tile = 0; tile < tiles; ++tile) {
      BOOST_CHECK_EQUAL(tile_owners[tile], pmap.owner(tile));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
