#include "TiledArray/array.h"
#include "TiledArray/utility.h"
#include "unit_test_config.h"
#include "range_fixture.h"
#include "shape_fixtures.h"

using namespace TiledArray;

struct ArrayFixture : public TiledRangeFixture, public ShapeFixture {
  typedef Array<int, GlobalFixture::coordinate_system> ArrayN;
  typedef ArrayN::index index;
  typedef ArrayN::value_type tile_type;

  ArrayFixture() : world(*GlobalFixture::world), a(world, tr) {

  }

  madness::World& world;
  ArrayN a;
}; // struct ArrayFixture

BOOST_FIXTURE_TEST_SUITE( array_suite , ArrayFixture )

BOOST_AUTO_TEST_CASE( constructors )
{
  // Construct a dense array
  BOOST_REQUIRE_NO_THROW(ArrayN ad(world, tr));

  // Construct a sparse array
  BOOST_REQUIRE_NO_THROW(ArrayN as(world, tr, list.begin(), list.end()));

  // Construct a predicated array
  BOOST_REQUIRE_NO_THROW(ArrayN ap(world, tr, p));

}

BOOST_AUTO_TEST_SUITE_END()
