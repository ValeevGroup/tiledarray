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

#include "TiledArray/external/madness.h"
#include "unit_test_config.h"

struct DistOpFixture {
  DistOpFixture()
      : group_list(),
        world_group_list(),
        group_did(GlobalFixture::world->make_unique_obj_id(),
                  GlobalFixture::world->rank() % 2),
        world_did(GlobalFixture::world->make_unique_obj_id(),
                  GlobalFixture::world->size()) {
    for (ProcessID p = GlobalFixture::world->rank() % 2;
         p < GlobalFixture::world->size(); p += 2)
      group_list.push_back(p);
    for (ProcessID p = 0; p < GlobalFixture::world->size(); ++p)
      world_group_list.push_back(p);
  }

  ~DistOpFixture() { GlobalFixture::world->gop.fence(); }

  std::vector<ProcessID> group_list;
  std::vector<ProcessID> world_group_list;
  madness::DistributedID group_did;
  madness::DistributedID world_did;

};  // DistOpFixture

struct SyncTester {
  TiledArray::Future<int> f;

  SyncTester() : f() {}
  SyncTester(const SyncTester& other) : f(other.f) {}

  void operator()() { f.set(GlobalFixture::world->size()); }
};  // struct sync_tester

template <typename T>
struct plus {
  typedef T result_type;
  typedef T argument_type;

  result_type operator()() const { return result_type(); }

  void operator()(result_type& result, const argument_type& arg) const {
    result += arg;
  }

  void operator()(result_type& result, const argument_type& arg1,
                  const argument_type& arg2) const {
    result += arg1 + arg2;
  }
};

BOOST_FIXTURE_TEST_SUITE(dist_op_suite, DistOpFixture)

BOOST_AUTO_TEST_CASE(ring_send_recv) {
  // Send messages in a ring.
  const ProcessID left_neighbor(
      (GlobalFixture::world->rank() + GlobalFixture::world->size() - 1) %
      GlobalFixture::world->size());
  const ProcessID right_neighbor(
      (GlobalFixture::world->rank() + GlobalFixture::world->size() + 1) %
      GlobalFixture::world->size());

  // Get the Future that will hold the remote data
  TiledArray::Future<int> remote_data;
  BOOST_REQUIRE_NO_THROW(
      remote_data = GlobalFixture::world->gop.recv<int>(right_neighbor, 0));

  // Send a future to the right neighbor
  TiledArray::Future<int> local_data;
  BOOST_REQUIRE_NO_THROW(
      GlobalFixture::world->gop.send(left_neighbor, 0, local_data));

  // Set the local data, which will be forwarded to the right neighbor
  local_data.set(GlobalFixture::world->rank());

  // Check that the message was received
  BOOST_CHECK_EQUAL(remote_data.get(), right_neighbor);
}

BOOST_AUTO_TEST_CASE(lazy_sync) {
  SyncTester sync_tester;
  int key = 1;

  // Start the lazy sync
  BOOST_REQUIRE_NO_THROW(GlobalFixture::world->gop.lazy_sync(key, sync_tester));

  // Test for completion
  BOOST_CHECK_EQUAL(sync_tester.f.get(), GlobalFixture::world->size());
}

BOOST_AUTO_TEST_CASE(lazy_sync_group) {
  // Create broadcast group
  madness::Group group(*GlobalFixture::world, group_list, group_did);

  SyncTester sync_tester;
  int key = 1;

  // Start the lazy sync
  BOOST_REQUIRE_NO_THROW(
      GlobalFixture::world->gop.lazy_sync(key, sync_tester, group));

  // Test for completion
  BOOST_CHECK_EQUAL(sync_tester.f.get(), GlobalFixture::world->size());

  // Cleanup the group
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(lazy_sync_world_group) {
  // Create broadcast group
  madness::Group group(*GlobalFixture::world, world_group_list, world_did);

  SyncTester sync_tester;
  int key = 1;

  // Start the lazy sync
  BOOST_REQUIRE_NO_THROW(
      GlobalFixture::world->gop.lazy_sync(key, sync_tester, group));

  // Test for completion
  BOOST_CHECK_EQUAL(sync_tester.f.get(), GlobalFixture::world->size());

  // Cleanup the group
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(bcast_world) {
  // Pick a random root
  const ProcessID root = 101 % GlobalFixture::world->size();

  // Setup the broadcast
  TiledArray::Future<int> data;
  BOOST_REQUIRE_NO_THROW(GlobalFixture::world->gop.bcast(0, data, root));

  // Set the data on the root process, which will initiate the broadcast.
  if (GlobalFixture::world->rank() == root) data.set(42);

  // Check that all processes got the same message.
  BOOST_CHECK_EQUAL(data.get(), 42);
}

BOOST_AUTO_TEST_CASE(bcast_group) {
  // Create broadcast group
  madness::Group group(*GlobalFixture::world, group_list, group_did);

  // Pick a random root
  const ProcessID root = 101 % group.size();

  // Setup the group broadcast
  TiledArray::Future<int> data;
  BOOST_REQUIRE_NO_THROW(GlobalFixture::world->gop.bcast(0, data, root, group));

  // Set the data on the root process, which will initiate the broadcast.
  if (group.rank() == root) data.set(42 + (GlobalFixture::world->rank() % 2));

  // Check that all processes in the group got the same message.
  BOOST_CHECK_EQUAL(data.get(), 42 + (GlobalFixture::world->rank() % 2));

  // Cleanup the group
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(bcast_world_group) {
  // Create broadcast group
  madness::Group group(*GlobalFixture::world, world_group_list, world_did);

  // Pick a random root
  const ProcessID root = 101 % group.size();

  // Setup the group broadcast
  TiledArray::Future<int> data;
  BOOST_REQUIRE_NO_THROW(GlobalFixture::world->gop.bcast(0, data, root, group));

  // Set the data which will initiate the broadcast
  if (GlobalFixture::world->rank() == root) data.set(42);

  // Check that all processes in the group got the same message.
  BOOST_CHECK_EQUAL(data.get(), 42);

  // Cleanup the group
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(reduce_world) {
  // Pick a random root
  const ProcessID root = 101 % GlobalFixture::world->size();

  // Setup the reduction
  TiledArray::Future<int> data;
  TiledArray::Future<int> result;
  BOOST_REQUIRE_NO_THROW(
      result = GlobalFixture::world->gop.reduce(0, data, plus<int>(), root));

  // Set the local value to be reduced
  data.set(42);

  // Check that the result has been reduced to the root process
  if (GlobalFixture::world->rank() == root)
    BOOST_CHECK_EQUAL(result.get(), GlobalFixture::world->size() * 42);
  else
    BOOST_CHECK(result.is_default_initialized());
}

BOOST_AUTO_TEST_CASE(reduce_group) {
  // Create reduction group
  madness::Group group(*GlobalFixture::world, group_list, group_did);

  // Pick a random root
  const ProcessID root = 101 % group.size();

  // Setup the reduction
  TiledArray::Future<int> data;
  TiledArray::Future<int> result;
  BOOST_REQUIRE_NO_THROW(result = GlobalFixture::world->gop.reduce(
                             0, data, plus<int>(), root, group));

  // Set the local value to be reduced
  data.set(42);

  // Check that the result has been reduced to the root process
  if (group.rank() == root)
    BOOST_CHECK_EQUAL(result.get(), group.size() * 42);
  else
    BOOST_CHECK(result.is_default_initialized());

  // Cleanup the group
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(reduce_world_group) {
  // Create reduction group
  madness::Group group(*GlobalFixture::world, world_group_list, world_did);

  // Pick a random root
  const ProcessID root = 101 % group.size();

  // Setup the reduction
  TiledArray::Future<int> data;
  TiledArray::Future<int> result;
  BOOST_REQUIRE_NO_THROW(result = GlobalFixture::world->gop.reduce(
                             0, data, plus<int>(), root, group));

  // Set the local value to be reduced
  data.set(42);

  // Check that the result has been reduced to the root process
  if (group.rank() == root)
    BOOST_CHECK_EQUAL(result.get(), group.size() * 42);
  else
    BOOST_CHECK(result.is_default_initialized());

  // Cleanup the group
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(all_reduce_world) {
  // Setup the reduction
  TiledArray::Future<int> data;
  TiledArray::Future<int> result;
  BOOST_REQUIRE_NO_THROW(
      result = GlobalFixture::world->gop.all_reduce(0, data, plus<int>()));

  // Set the local value to be reduced
  data.set(42);

  // Check that the result has been reduced to the root process
  BOOST_CHECK_EQUAL(result.get(), GlobalFixture::world->size() * 42);
}

BOOST_AUTO_TEST_CASE(all_reduce_group) {
  // Create reduction group
  madness::Group group(*GlobalFixture::world, group_list, group_did);

  // Setup the reduction
  TiledArray::Future<int> data;
  TiledArray::Future<int> result;
  BOOST_REQUIRE_NO_THROW(result = GlobalFixture::world->gop.all_reduce(
                             0, data, plus<int>(), group));

  // Set the local value to be reduced
  data.set(42);

  // Check that the result has been reduced to the root process
  BOOST_CHECK_EQUAL(result.get(), group.size() * 42);

  // Cleanup the group
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_CASE(all_reduce_world_group) {
  // Create reduction group
  madness::Group group(*GlobalFixture::world, world_group_list, world_did);

  // Setup the reduction
  TiledArray::Future<int> data;
  TiledArray::Future<int> result;
  BOOST_REQUIRE_NO_THROW(result = GlobalFixture::world->gop.all_reduce(
                             0, data, plus<int>(), group));

  // Set the local value to be reduced
  data.set(42);

  // Check that the result has been reduced to the root process
  BOOST_CHECK_EQUAL(result.get(), group.size() * 42);

  // Cleanup the group
  GlobalFixture::world->gop.fence();
}

BOOST_AUTO_TEST_SUITE_END()
