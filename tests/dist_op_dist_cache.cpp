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
 *  dist_op_dist_cache.cpp
 *  Oct 14, 2013
 *
 */

#include "TiledArray/dist_op/dist_cache.h"
#include "unit_test_config.h"

using TiledArray::dist_op::detail::DistCache;

struct DistCacheFixture {

  DistCacheFixture() :
    dist_cache(* GlobalFixture::world),
    left_neighbor((GlobalFixture::world->rank() + GlobalFixture::world->size() - 1) %
        GlobalFixture::world->size()),
    right_neighbor( (GlobalFixture::world->rank() + GlobalFixture::world->size() + 1) %
        GlobalFixture::world->size())
  { }

  ~DistCacheFixture() { }

  DistCache dist_cache;
  const ProcessID left_neighbor;
  const ProcessID right_neighbor;
}; // DistCacheFixture

BOOST_FIXTURE_TEST_SUITE( dist_op_dist_cache_suite, DistCacheFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_CHECK_NO_THROW(DistCache x(* GlobalFixture::world));
}

BOOST_AUTO_TEST_CASE( ring_send_recv )
{
  // Send messages in a ring.

  // Get the Future that will hold the remote data
  madness::Future<int> remote_data;
  BOOST_REQUIRE_NO_THROW(remote_data = dist_cache.recv<int>(0));

  // Send a future to the right neighbor
  madness::Future<int> local_data;
  BOOST_REQUIRE_NO_THROW(dist_cache.send(left_neighbor, 0, local_data));

  // Set the local data, which should be forwarded to the right neighbor
  local_data.set(GlobalFixture::world->rank());

  BOOST_CHECK_EQUAL(remote_data.get(), right_neighbor);
}

BOOST_AUTO_TEST_SUITE_END()
