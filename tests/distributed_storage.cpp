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

#include "TiledArray/distributed_storage.h"
#include <iterator>
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;

struct DistributedStorageFixture {
  typedef TiledArray::detail::DistributedStorage<int> Storage;
  typedef Storage::size_type size_type;

  DistributedStorageFixture()
      : world(*GlobalFixture::world),
        pmap(new detail::BlockedPmap(world, 10)),
        t(world, 10, pmap) {}

  ~DistributedStorageFixture() { world.gop.fence(); }

  TiledArray::World& world;
  std::shared_ptr<detail::BlockedPmap> pmap;
  Storage t;
};

struct DistributeOp {
  static madness::AtomicInt count;

  void operator()(std::size_t, int) const { ++count; }

  template <typename Archive>
  void serialize(Archive&) {}
};

madness::AtomicInt DistributeOp::count;

BOOST_FIXTURE_TEST_SUITE(distributed_storage_suite, DistributedStorageFixture)

BOOST_AUTO_TEST_CASE(constructor) {
  BOOST_CHECK_NO_THROW(Storage s(world, 10, pmap));
  Storage s(world, 10, pmap);

  BOOST_CHECK_EQUAL(s.size(), 0ul);
  BOOST_CHECK_EQUAL(s.max_size(), 10ul);
}

BOOST_AUTO_TEST_CASE(get_world) {
  BOOST_CHECK_EQUAL(&t.get_world(), GlobalFixture::world);
}

BOOST_AUTO_TEST_CASE(get_pmap) { BOOST_CHECK_EQUAL(t.pmap(), pmap); }

BOOST_AUTO_TEST_CASE(set_value) {
  // Check that we can set all elements
  for (std::size_t i = 0; i < t.max_size(); ++i)
    if (t.is_local(i)) t.set(i, world.rank());

  world.gop.fence();
  std::size_t n = t.size();
  world.gop.sum(n);

  BOOST_CHECK_EQUAL(n, t.max_size());

  // Check throw for an out-of-range set.
  BOOST_CHECK_THROW(t.set(t.max_size(), 1), TiledArray::Exception);
  BOOST_CHECK_THROW(t.set(t.max_size() + 2, 1), TiledArray::Exception);
}

BOOST_AUTO_TEST_CASE(array_operator) {
  // Check that elements are inserted properly for access requests.
  for (std::size_t i = 0; i < t.max_size(); ++i) {
    t.get(i).probe();
    if (t.is_local(i)) t.set(i, world.rank());
  }

  world.gop.fence();
  std::size_t n = t.size();
  world.gop.sum(n);

  BOOST_CHECK_EQUAL(n, t.max_size());

  // Check throw for an out-of-range set.
  BOOST_CHECK_THROW(t.get(t.max_size()), TiledArray::Exception);
  BOOST_CHECK_THROW(t.get(t.max_size() + 2), TiledArray::Exception);
}

BOOST_AUTO_TEST_SUITE_END()
