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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  dist_op.cpp
 *  Oct 24, 2013
 *
 */

#include "TiledArray/dist_op/communicator.h"
#include "unit_test_config.h"

using TiledArray::Communicator;

struct DistOpFixture {

  DistOpFixture() :
    d_op(* GlobalFixture::world)
  { }

  ~DistOpFixture() {
    GlobalFixture::world->gop.fence();
  }

  Communicator d_op;

}; // DistOpFixture

BOOST_FIXTURE_TEST_SUITE( dist_op_suite, DistOpFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_CHECK_NO_THROW(Communicator x(* GlobalFixture::world));
}

BOOST_AUTO_TEST_CASE( ring_send_recv )
{
  // Send messages in a ring.
  const ProcessID left_neighbor((GlobalFixture::world->rank() +
      GlobalFixture::world->size() - 1) % GlobalFixture::world->size());
  const ProcessID right_neighbor( (GlobalFixture::world->rank() +
      GlobalFixture::world->size() + 1) % GlobalFixture::world->size());

  // Get the Future that will hold the remote data
  madness::Future<int> remote_data;
  BOOST_REQUIRE_NO_THROW(remote_data = d_op.recv<int>(0));

  // Send a future to the right neighbor
  madness::Future<int> local_data;
  BOOST_REQUIRE_NO_THROW(d_op.send(left_neighbor, 0, local_data));

  // Set the local data, which should be forwarded to the right neighbor
  local_data.set(GlobalFixture::world->rank());

  BOOST_CHECK_EQUAL(remote_data.get(), right_neighbor);
}

BOOST_AUTO_TEST_CASE( bcast_world )
{
  TiledArray::dist_op::DistributedID did(madness::uniqueidT(), 0);
  madness::Future<int> data;

  BOOST_REQUIRE_NO_THROW(d_op.bcast(did, data, 0));

  if(GlobalFixture::world->rank() == 0)
    data.set(42);

  BOOST_CHECK_EQUAL(data.get(), 42);
}

//BOOST_AUTO_TEST_CASE( bcast_group )
//{
//  std::vector<ProcessID> group_list;
//  for(ProcessID p = GlobalFixture::world->rank() % 2; p < GlobalFixture::world->size(); p += 2)
//    group_list.push_back(p);
//  Group group(*GlobalFixture::world, did, group_list);
//  register_group(group);
//
//  madness::Future<int> data;
//  BOOST_REQUIRE_NO_THROW(bcast(did, data, 0, group));
//
//  if(group.rank() == 0)
//    data.set(42 + (GlobalFixture::world->rank() % 2));
//
//  BOOST_CHECK_EQUAL(data.get(), 42 + (GlobalFixture::world->rank() % 2));
//
//  unregister_group(group);
//  GlobalFixture::world->gop.fence();
//}

BOOST_AUTO_TEST_SUITE_END()

