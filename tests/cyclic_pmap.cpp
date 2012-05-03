#include "unit_test_config.h"
#include "global_fixture.h"
#include "TiledArray/cyclic_pmap.h"

using namespace TiledArray;

struct CyclicPmapFixture {

  CyclicPmapFixture() : pmap(* GlobalFixture::world, 5ul, 10ul) {
    pmap.set_seed();
  }

  detail::CyclicPmap pmap;
};


// =============================================================================
// BlockdPmap Test Suite


BOOST_FIXTURE_TEST_SUITE( cyclic_pmap_suite, CyclicPmapFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(TiledArray::detail::CyclicPmap pmap(* GlobalFixture::world, 5ul, 10ul));
}

BOOST_AUTO_TEST_CASE( owner )
{
  const std::size_t rank = GlobalFixture::world->rank();
  const std::size_t size = GlobalFixture::world->size();

  ProcessID* p_owner = new ProcessID[size];

  for(std::size_t i = 0ul; i < 50ul; ++i) {
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

  BOOST_CHECK_EQUAL(total_size, 50ul);
  BOOST_CHECK(pmap.empty() == (pmap.local_size() == 0ul));

}

BOOST_AUTO_TEST_CASE( local_group )
{
  // Make sure the total number of elements in the local groups is correct
  std::size_t total_size = std::distance(pmap.begin(), pmap.end());
  GlobalFixture::world->gop.sum(total_size);

  BOOST_CHECK_EQUAL(total_size, 50ul);

  // Check that all local elements map to this rank
  for(detail::CyclicPmap::const_iterator it = pmap.begin(); it != pmap.end(); ++it) {
    BOOST_CHECK_EQUAL(pmap.owner(*it), GlobalFixture::world->rank());
  }

  // Check that all elements owned by this rank are in the local group exactly once
  for(std::size_t i = 0; i < 50ul; ++i) {
    if(pmap.owner(i) == GlobalFixture::world->rank()) {
      BOOST_CHECK_EQUAL(std::count(pmap.begin(), pmap.end(), i), 1);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()

