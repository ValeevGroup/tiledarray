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

#include "TiledArray/pmap/cyclic_pmap.h"
#include "global_fixture.h"
#include "unit_test_config.h"

using namespace TiledArray;

struct CyclicPmapFixture {
  CyclicPmapFixture() {}
};

// =============================================================================
// BlockdPmap Test Suite

BOOST_FIXTURE_TEST_SUITE(cyclic_pmap_suite, CyclicPmapFixture)

BOOST_AUTO_TEST_CASE(constructor) {
  for (ProcessID x = 1ul; x <= GlobalFixture::world->size(); ++x) {
    for (ProcessID y = 1ul; y <= GlobalFixture::world->size(); ++y) {
      // Compute the limits for process rows
      const std::size_t min_proc_rows = std::max<std::size_t>(
          ((GlobalFixture::world->size() + y - 1ul) / y), 1ul);
      const std::size_t max_proc_rows =
          std::min<std::size_t>(GlobalFixture::world->size(), x);

      // Compute process rows and process columns
      const std::size_t p_rows = std::max<std::size_t>(
          min_proc_rows,
          std::min<std::size_t>(std::sqrt(GlobalFixture::world->size() * x / y),
                                max_proc_rows));
      const std::size_t p_cols = GlobalFixture::world->size() / p_rows;

      BOOST_REQUIRE_NO_THROW(TiledArray::detail::CyclicPmap pmap(
          *GlobalFixture::world, x, y, p_rows, p_cols));
      TiledArray::detail::CyclicPmap pmap(*GlobalFixture::world, x, y, p_rows,
                                          p_cols);
      BOOST_CHECK_EQUAL(pmap.rank(), GlobalFixture::world->rank());
      BOOST_CHECK_EQUAL(pmap.procs(), GlobalFixture::world->size());
      BOOST_CHECK_EQUAL(pmap.size(), x * y);
    }
  }

  ProcessID size = GlobalFixture::world->size();

  BOOST_CHECK_TA_ASSERT(TiledArray::detail::CyclicPmap pmap(
                            *GlobalFixture::world, 10ul, 10ul, 0, 1),
                        TiledArray::Exception);
  BOOST_CHECK_TA_ASSERT(TiledArray::detail::CyclicPmap pmap(
                            *GlobalFixture::world, 10ul, 10ul, 1, 0),
                        TiledArray::Exception);
  BOOST_CHECK_TA_ASSERT(TiledArray::detail::CyclicPmap pmap(
                            *GlobalFixture::world, 10ul, 10ul, size * 2, 1),
                        TiledArray::Exception);
  BOOST_CHECK_TA_ASSERT(TiledArray::detail::CyclicPmap pmap(
                            *GlobalFixture::world, 10ul, 10ul, 1, size * 2),
                        TiledArray::Exception);
  if (size > 1) {
    BOOST_CHECK_TA_ASSERT(TiledArray::detail::CyclicPmap pmap(
                              *GlobalFixture::world, 10ul, 10ul, size, size),
                          TiledArray::Exception);
  }
}

BOOST_AUTO_TEST_CASE(owner) {
  const std::size_t rank = GlobalFixture::world->rank();
  const std::size_t size = GlobalFixture::world->size();

  ProcessID* p_owner = new ProcessID[size];

  // Check various pmap sizes
  for (std::size_t x = 1ul; x < 10ul; ++x) {
    for (std::size_t y = 1ul; y < 10ul; ++y) {
      // Compute the limits for process rows
      const std::size_t min_proc_rows = std::max<std::size_t>(
          ((GlobalFixture::world->size() + y - 1ul) / y), 1ul);
      const std::size_t max_proc_rows =
          std::min<std::size_t>(GlobalFixture::world->size(), x);

      // Compute process rows and process columns
      const std::size_t p_rows = std::max<std::size_t>(
          min_proc_rows,
          std::min<std::size_t>(std::sqrt(GlobalFixture::world->size() * x / y),
                                max_proc_rows));
      const std::size_t p_cols = GlobalFixture::world->size() / p_rows;

      const std::size_t tiles = x * y;
      TiledArray::detail::CyclicPmap pmap(*GlobalFixture::world, x, y, p_rows,
                                          p_cols);

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
  }

  delete[] p_owner;
}

BOOST_AUTO_TEST_CASE(local_size) {
  for (std::size_t x = 1ul; x < 10ul; ++x) {
    for (std::size_t y = 1ul; y < 10ul; ++y) {
      // Compute the limits for process rows
      const std::size_t min_proc_rows = std::max<std::size_t>(
          ((GlobalFixture::world->size() + y - 1ul) / y), 1ul);
      const std::size_t max_proc_rows =
          std::min<std::size_t>(GlobalFixture::world->size(), x);

      // Compute process rows and process columns
      const std::size_t p_rows = std::max<std::size_t>(
          min_proc_rows,
          std::min<std::size_t>(std::sqrt(GlobalFixture::world->size() * x / y),
                                max_proc_rows));
      const std::size_t p_cols = GlobalFixture::world->size() / p_rows;

      const std::size_t tiles = x * y;
      TiledArray::detail::CyclicPmap pmap(*GlobalFixture::world, x, y, p_rows,
                                          p_cols);

      std::size_t total_size = pmap.local_size();
      GlobalFixture::world->gop.sum(total_size);

      // Check that the total number of elements in all local groups is equal to
      // the number of tiles in the map.
      BOOST_CHECK_EQUAL(total_size, tiles);
      BOOST_CHECK(pmap.empty() == (pmap.local_size() == 0ul));
    }
  }
}

BOOST_AUTO_TEST_CASE(local_group) {
  ProcessID tile_owners[100];

  for (std::size_t x = 1ul; x < 10ul; ++x) {
    for (std::size_t y = 1ul; y < 10ul; ++y) {
      // Compute the limits for process rows
      const std::size_t min_proc_rows = std::max<std::size_t>(
          ((GlobalFixture::world->size() + y - 1ul) / y), 1ul);
      const std::size_t max_proc_rows =
          std::min<std::size_t>(GlobalFixture::world->size(), x);

      // Compute process rows and process columns
      const std::size_t p_rows = std::max<std::size_t>(
          min_proc_rows,
          std::min<std::size_t>(std::sqrt(GlobalFixture::world->size() * x / y),
                                max_proc_rows));
      const std::size_t p_cols = GlobalFixture::world->size() / p_rows;

      const std::size_t tiles = x * y;
      TiledArray::detail::CyclicPmap pmap(*GlobalFixture::world, x, y, p_rows,
                                          p_cols);

      // Check that all local elements map to this rank
      for (detail::CyclicPmap::const_iterator it = pmap.begin();
           it != pmap.end(); ++it) {
        BOOST_CHECK_EQUAL(pmap.owner(*it), GlobalFixture::world->rank());
      }

      std::fill_n(tile_owners, tiles, 0);
      for (detail::CyclicPmap::const_iterator it = pmap.begin();
           it != pmap.end(); ++it) {
        tile_owners[*it] += GlobalFixture::world->rank();
      }

      GlobalFixture::world->gop.sum(tile_owners, tiles);
      for (std::size_t tile = 0; tile < tiles; ++tile) {
        BOOST_CHECK_EQUAL(tile_owners[tile], pmap.owner(tile));
      }
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
