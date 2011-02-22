#include "TiledArray/array.h"
#include "TiledArray/utility.h"
#include "unit_test_config.h"
#include "range_fixture.h"

using namespace TiledArray;

struct ArrayFixture : public TiledRangeFixture {
  typedef Array<int, GlobalFixture::coordinate_system> ArrayN;
  typedef ArrayN::index index;
  typedef ArrayN::value_type tile_type;

  ArrayFixture() : a(world, tr) {

  }

  static madness::World& world;
  ArrayN a;
}; // struct ArrayFixture

// static veriables for fixture
madness::World& ArrayFixture::world = *GlobalFixture::world;

BOOST_FIXTURE_TEST_SUITE( array_suite , ArrayFixture )

BOOST_AUTO_TEST_CASE( constructors )
{

}

BOOST_AUTO_TEST_SUITE_END()
