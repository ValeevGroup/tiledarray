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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  replicated_pmap.cpp
 *  Aug 3, 2013
 *
 */

#include "TiledArray/pmap/replicated_pmap.h"
#include "unit_test_config.h"

struct ReplicatedPmapFixture {
  constexpr static std::size_t max_ntiles = 10ul;
};  // Fixture

BOOST_FIXTURE_TEST_SUITE(replicated_pmap_suite, ReplicatedPmapFixture)

BOOST_AUTO_TEST_CASE(constructor) {
  for (std::size_t tiles = 1ul; tiles < max_ntiles; ++tiles) {
    BOOST_REQUIRE_NO_THROW(
        TiledArray::detail::ReplicatedPmap pmap(*GlobalFixture::world, tiles));
    TiledArray::detail::ReplicatedPmap pmap(*GlobalFixture::world, tiles);
    BOOST_CHECK_EQUAL(pmap.rank(), GlobalFixture::world->rank());
    BOOST_CHECK_EQUAL(pmap.procs(), GlobalFixture::world->size());
    BOOST_CHECK_EQUAL(pmap.size(), tiles);
  }
}

BOOST_AUTO_TEST_CASE(owner) {
  const std::size_t rank = GlobalFixture::world->rank();

  // Check various pmap sizes
  for (std::size_t tiles = 1ul; tiles < max_ntiles; ++tiles) {
    TiledArray::detail::ReplicatedPmap pmap(*GlobalFixture::world, tiles);

    for (std::size_t tile = 0; tile < tiles; ++tile) {
      BOOST_CHECK_EQUAL(pmap.owner(tile), rank);
    }
  }
}

BOOST_AUTO_TEST_CASE(local_size) {
  for (std::size_t tiles = 1ul; tiles < max_ntiles; ++tiles) {
    TiledArray::detail::ReplicatedPmap pmap(*GlobalFixture::world, tiles);

    // Check that the total number of elements in all local groups is equal to
    // the number of tiles in the map.
    BOOST_CHECK_EQUAL(pmap.local_size(), tiles);
    BOOST_CHECK(pmap.empty() == (pmap.local_size() == 0ul));
  }
}

BOOST_AUTO_TEST_CASE(local_group) {
  for (std::size_t tiles = 1ul; tiles < max_ntiles; ++tiles) {
    TiledArray::detail::ReplicatedPmap pmap(*GlobalFixture::world, tiles);

    // Check that all local elements map to this rank
    for (TiledArray::detail::ReplicatedPmap::const_iterator it = pmap.begin();
         it != pmap.end(); ++it) {
      BOOST_CHECK_EQUAL(pmap.owner(*it), GlobalFixture::world->rank());
    }

    std::size_t i = 0ul;
    for (TiledArray::detail::ReplicatedPmap::const_iterator it = pmap.begin();
         it != pmap.end(); ++it, ++i) {
      BOOST_CHECK_EQUAL(*it, i);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
