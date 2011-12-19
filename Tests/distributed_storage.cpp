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

BOOST_AUTO_TEST_SUITE_END()
