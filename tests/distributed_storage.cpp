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
#include "TiledArray/pmap/blocked_pmap.h"
#include "unit_test_config.h"
#include <world/worlddc.h>
#include <iterator>

using namespace TiledArray;

struct DistributedStorageFixture {
  typedef TiledArray::detail::DistributedStorage<int> Storage;
  typedef Storage::size_type size_type;

  DistributedStorageFixture() :
      world(* GlobalFixture::world),
      pmap(new detail::BlockedPmap(world, 10)),
      t(new Storage(world, 10, pmap), madness::make_deferred_deleter<Storage>(world))
  { }

  ~DistributedStorageFixture() {
    world.gop.fence();
  }


  madness::World& world;
  std::shared_ptr<detail::BlockedPmap> pmap;
  std::shared_ptr<Storage> t;
};

struct DistributeOp {
  static madness::AtomicInt count;

  void operator()(std::size_t, int) const {
    ++count;
  }

  template <typename Archive>
  void serialize(const Archive&) { }
};

madness::AtomicInt DistributeOp::count;

BOOST_FIXTURE_TEST_SUITE( distributed_storage_suite , DistributedStorageFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  std::shared_ptr<Storage> s;
  BOOST_CHECK_NO_THROW(s.reset(new Storage(world, 10, pmap),
      madness::make_deferred_deleter<Storage>(world)));

  BOOST_CHECK_EQUAL(s->size(), 0ul);
  BOOST_CHECK_EQUAL(s->max_size(), 10ul);

}

BOOST_AUTO_TEST_CASE( get_world )
{
  BOOST_CHECK_EQUAL(& t->get_world(), GlobalFixture::world);
}

BOOST_AUTO_TEST_CASE( get_pmap )
{
  BOOST_CHECK_EQUAL(t->get_pmap(), pmap);
}

BOOST_AUTO_TEST_CASE( set_value )
{
  // Check that we can set all elements
  for(std::size_t i = 0; i < t->max_size(); ++i)
    if(t->is_local(i))
      t->set(i, world.rank());

  world.gop.fence();
  std::size_t n = t->size();
  world.gop.sum(n);

  BOOST_CHECK_EQUAL(n, t->max_size());

  // Check throw for an out-of-range set.
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(t->set(t->max_size(), 1), TiledArray::Exception);
  BOOST_CHECK_THROW(t->set(t->max_size() + 2, 1), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( array_operator )
{
  // Check that elements are inserted properly for access requests.
  for(std::size_t i = 0; i < t->max_size(); ++i) {
    t->get(i).probe();
    if(t->is_local(i))
      t->set(i, world.rank());
  }

  world.gop.fence();
  std::size_t n = t->size();
  world.gop.sum(n);

  BOOST_CHECK_EQUAL(n, t->max_size());

  // Check throw for an out-of-range set.
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(t->get(t->max_size()), TiledArray::Exception);
  BOOST_CHECK_THROW(t->get(t->max_size() + 2), TiledArray::Exception);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( move_local )
{
  std::size_t local_size = 0ul;

  for(std::size_t i = 0; i < t->max_size(); ++i) {
    if(t->is_local(i)) {
      t->set_cache(i, world.rank());
      ++local_size;
    }

    BOOST_CHECK_EQUAL(t->size(), local_size);
  }

  world.gop.fence();

  for(std::size_t i = 0; i < t->max_size(); ++i) {
    if(t->is_local(i)) {
      madness::Future<int> f = t->get_cache(i);

      BOOST_CHECK_EQUAL(f.get(), world.rank());
      --local_size;
    }

    BOOST_CHECK_EQUAL(t->size(), local_size);
  }

  BOOST_CHECK_EQUAL(t->size(), 0ul);
}

BOOST_AUTO_TEST_CASE( delayed_move_local )
{
  std::size_t local_size = 0ul;
  std::deque<madness::Future<int> > local_data;

  for(std::size_t i = 0; i < t->max_size(); ++i) {
    if(t->is_local(i)) {
      local_data.push_front(t->get_cache(i));
      ++local_size;
    }

    BOOST_CHECK_EQUAL(t->size(), local_size);
  }

  world.gop.fence();

  for(std::size_t i = 0; i < t->max_size(); ++i) {
    if(t->is_local(i)) {
      t->set_cache(i, world.rank());

      BOOST_CHECK_EQUAL(local_data.back().get(), world.rank());
      local_data.pop_back();
      --local_size;
    }

    BOOST_CHECK_EQUAL(t->size(), local_size);
  }

  BOOST_CHECK_EQUAL(t->size(), 0ul);
}

BOOST_AUTO_TEST_CASE( move_remote )
{
  std::size_t local_size = 0ul;

  // Insert all elements
  for(std::size_t i = 0; i < t->max_size(); ++i) {
    if(t->is_local(i)) {
      t->set_cache(i, world.rank());
      ++local_size;
    }

    BOOST_CHECK_EQUAL(t->size(), local_size);
  }

  world.gop.fence();

  for(std::size_t i = 0; i < t->max_size(); ++i) {
    if(world.rank() == 0) {
      madness::Future<int> f = t->get_cache(i);
      BOOST_CHECK_EQUAL(f.get(), t->owner(i));
    }

    if(t->is_local(i))
      --local_size;

    world.gop.fence();

    BOOST_CHECK_EQUAL(local_size, t->size());

    world.gop.fence();
  }
}


BOOST_AUTO_TEST_CASE( delayed_move_remote )
{
  std::vector<madness::Future<int> > local_data;

  for(std::size_t i = 0; i < t->max_size(); ++i) {
    if(world.rank() == 0)
      local_data.push_back(t->get_cache(i));
  }

  world.gop.fence();

  std::size_t local_size = t->size();

  for(std::size_t i = 0; i < t->max_size(); ++i) {

    world.gop.fence();

    if(t->is_local(i)) {
      t->set_cache(i, world.rank());
      --local_size;
    }

    BOOST_CHECK_EQUAL(local_size, t->size());
  }

  for(std::vector<madness::Future<int> >::iterator it = local_data.begin(); it != local_data.end(); ++it) {
    BOOST_CHECK_EQUAL(it->get(), t->owner(std::distance(local_data.begin(), it)));
  }
}

BOOST_AUTO_TEST_SUITE_END()
