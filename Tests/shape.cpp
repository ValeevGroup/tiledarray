#include "TiledArray/shape.h"
#include "TiledArray/range.h"
#include "unit_test_config.h"
#include "global_fixture.h"
#include "range_fixture.h"
#include "shape_fixtures.h"

using namespace TiledArray;

const BaseShapeFixture::RangeN BaseShapeFixture::r(index(0), index(5));
const BaseShapeFixture::PmapT BaseShapeFixture::m(GlobalFixture::world);

// =============================================================================
// Shape Test Suite

BOOST_FIXTURE_TEST_SUITE( shape_suite, ShapeFixture )

BOOST_AUTO_TEST_CASE( is_dense )
{

  BOOST_CHECK(is_dense_shape(ds));
  BOOST_CHECK(! is_dense_shape(ss));

  ShapeT& s1 = ds;
  ShapeT& s2 = ss;

  BOOST_CHECK(is_dense_shape(s1));
  BOOST_CHECK(! is_dense_shape(s2));

  int i = 0;

  BOOST_CHECK(! is_dense_shape(i));
}

BOOST_AUTO_TEST_CASE( is_sparse )
{

  BOOST_CHECK(is_sparse_shape(ss));
  BOOST_CHECK(! is_sparse_shape(ds));

  ShapeT& s1 = ss;
  ShapeT& s2 = ds;

  BOOST_CHECK(is_sparse_shape(s1));
  BOOST_CHECK(! is_sparse_shape(s2));

  int i = 0;

  BOOST_CHECK(! is_sparse_shape(i));
}

BOOST_AUTO_TEST_SUITE_END()

