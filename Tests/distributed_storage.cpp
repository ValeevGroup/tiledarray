#include "TiledArray/distributed_storage.h"
#include "unit_test_config.h"
#include <world/worlddc.h>

using namespace TiledArray;

struct DistributedStorageFixture {
  typedef TiledArray::detail::DistributedStorage<int> Storage;
  typedef Storage::size_type size_type;

  DistributedStorageFixture() :
      world(* GlobalFixture::world),
      pmap(new madness::WorldDCDefaultPmap<std::size_t>(world)),
      t(new Storage(world, 10, pmap), madness::make_deferred_deleter<Storage>(world))
  { }

  ~DistributedStorageFixture() {
    world.gop.fence();
  }


  madness::World& world;
  std::shared_ptr<madness::WorldDCPmapInterface<std::size_t> > pmap;
  std::shared_ptr<Storage> t;
};

BOOST_FIXTURE_TEST_SUITE( distributed_storage_suite , DistributedStorageFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  std::shared_ptr<Storage> s;
  BOOST_CHECK_NO_THROW(s.reset(new Storage(world, 10, pmap),
      madness::make_deferred_deleter<Storage>(world)));

  BOOST_CHECK_EQUAL(s->size(), 0);
  BOOST_CHECK_EQUAL(s->max_size(), 10);
  BOOST_CHECK(s->begin() == s->end());

}

BOOST_AUTO_TEST_CASE( get_world )
{
  BOOST_CHECK_EQUAL(& t->get_world(), GlobalFixture::world);
}

BOOST_AUTO_TEST_CASE( get_pmap )
{
  BOOST_CHECK_EQUAL(t->get_pmap(), pmap);
}

BOOST_AUTO_TEST_CASE( insert )
{
  // Make sure the element is only inserted once when every node inserts it.
  t->insert(0);

  world.gop.fence();
  int n = t->size();
  world.gop.sum(n);

  BOOST_CHECK_EQUAL(n, 1);

  for(std::size_t i = 0; i < t->max_size(); ++i)
    t->insert(i);


  world.gop.fence();
  n = t->size();
  world.gop.sum(n);

  BOOST_CHECK_EQUAL(n, t->max_size());


  // Check throw for an out-of-range insert.
#ifndef NDEBUG
  BOOST_CHECK_THROW(t->insert(t->max_size()), TiledArray::Exception);
  BOOST_CHECK_THROW(t->insert(t->max_size() + 2), TiledArray::Exception);
#endif // NDEBUG
}

BOOST_AUTO_TEST_CASE( set_value )
{
  // Check that we can set all elements
  for(std::size_t i = 0; i < t->max_size(); ++i)
    if(t->is_local(i))
      t->set(i, world.rank());

  world.gop.fence();
  int n = t->size();
  world.gop.sum(n);

  BOOST_CHECK_EQUAL(n, t->max_size());

  // Check throw for an out-of-range set.
#ifndef NDEBUG
  BOOST_CHECK_THROW(t->set(t->max_size(), 1), TiledArray::Exception);
  BOOST_CHECK_THROW(t->set(t->max_size() + 2, 1), TiledArray::Exception);
#endif // NDEBUG
}

BOOST_AUTO_TEST_CASE( array_operator )
{
  // Check that elements are inserted properly for access requests.
  for(std::size_t i = 0; i < t->max_size(); ++i)
    (*t)[i].probe();

  world.gop.fence();
  int n = t->size();
  world.gop.sum(n);

  BOOST_CHECK_EQUAL(n, t->max_size());

  // Check throw for an out-of-range set.
#ifndef NDEBUG
  BOOST_CHECK_THROW((*t)[t->max_size()], TiledArray::Exception);
  BOOST_CHECK_THROW((*t)[t->max_size() + 2], TiledArray::Exception);
#endif // NDEBUG
}


BOOST_AUTO_TEST_SUITE_END()
