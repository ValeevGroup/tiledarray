#include "unit_test_config.h"
#include "global_fixture.h"
#include "TiledArray/cyclic_pmap.h"

using namespace TiledArray;

struct CyclicPmapFixture {

  CyclicPmapFixture() : pmap(* GlobalFixture::world, 10ul, 10ul) {
    pmap.set_seed();
  }

  detail::CyclicPmap pmap;
};


// =============================================================================
// BlockdPmap Test Suite


BOOST_FIXTURE_TEST_SUITE( cyclic_pmap_suite, CyclicPmapFixture )

BOOST_AUTO_TEST_CASE( constructor_all_default )
{
  BOOST_REQUIRE_NO_THROW(TiledArray::detail::CyclicPmap pmap(* GlobalFixture::world, 10ul, 10ul));

  std::size_t total_size = pmap.local_size();
  GlobalFixture::world->gop.sum(total_size);

  BOOST_CHECK_EQUAL(total_size, 100ul);
  BOOST_CHECK(pmap.empty() == (pmap.local_size() == 0ul));

}

BOOST_AUTO_TEST_CASE( local_group )
{
  std::size_t total_size = std::distance(pmap.begin(), pmap.end());
  GlobalFixture::world->gop.sum(total_size);

  BOOST_CHECK_EQUAL(total_size, 100ul);

  // Check that all local elements map to this rank
  for(detail::CyclicPmap::const_iterator it = pmap.begin(); it != pmap.end(); ++it) {
    BOOST_CHECK_EQUAL(pmap.owner(*it), GlobalFixture::world->rank());
  }

  // Check that all elements owned by this rank are in the local group
  for(std::size_t i = 0; i < 100ul; ++i) {
    if(pmap.owner(i) == GlobalFixture::world->rank()) {
      BOOST_CHECK(std::find(pmap.begin(), pmap.end(), i) != pmap.end());
    }
  }


}

BOOST_AUTO_TEST_SUITE_END()

