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

#include "TiledArray/pmap/blocked_pmap.h"
#include "config.h"
#include "global_fixture.h"

using namespace TiledArray;

struct BlockedPmapFixture {

  BlockedPmapFixture() : pmap(* GlobalFixture::world, 100ul) {
    pmap.set_seed();
  }

  detail::BlockedPmap pmap;
};


// =============================================================================
// BlockdPmap Test Suite


BOOST_FIXTURE_TEST_SUITE( blocked_pmap_suite, BlockedPmapFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(TiledArray::detail::BlockedPmap pmap(* GlobalFixture::world, 100ul));
}

BOOST_AUTO_TEST_CASE( owner )
{
  const std::size_t rank = GlobalFixture::world->rank();
  const std::size_t size = GlobalFixture::world->size();

  ProcessID* p_owner = new ProcessID[size];

  for(std::size_t i = 0; i < 100; ++i) {
    std::fill_n(p_owner, size, 0);
    p_owner[rank] = pmap.owner(i);
    // check that the value is in range
    BOOST_CHECK_LT(p_owner[rank], size);
    GlobalFixture::world->gop.sum(p_owner, size);

    // Make sure everyone agrees on who owns what.
    for(std::size_t p = 0ul; p < size; ++p)
      BOOST_CHECK_EQUAL(p_owner[p], p_owner[rank]);
  }

  delete [] p_owner;
}

BOOST_AUTO_TEST_CASE( local_size )
{
  std::size_t total_size = pmap.local_size();
  GlobalFixture::world->gop.sum(total_size);

  BOOST_CHECK_EQUAL(total_size, 100ul);
  BOOST_CHECK(pmap.empty() == (pmap.local_size() == 0ul));
}

BOOST_AUTO_TEST_CASE( local_group )
{
  // Make sure the total number of elements in the local groups is correct
  std::size_t total_size = std::distance(pmap.begin(), pmap.end());
  GlobalFixture::world->gop.sum(total_size);

  BOOST_CHECK_EQUAL(total_size, 100ul);

  // Check that all local elements map to this rank
  for(detail::BlockedPmap::const_iterator it = pmap.begin(); it != pmap.end(); ++it) {
    BOOST_CHECK_EQUAL(pmap.owner(*it), GlobalFixture::world->rank());
  }


  // Check that all elements owned by this rank are in the local group exactly once
  for(std::size_t i = 0; i < 100ul; ++i) {
    if(pmap.owner(i) == GlobalFixture::world->rank()) {
      BOOST_CHECK_EQUAL(std::count(pmap.begin(), pmap.end(), i), 1);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()






