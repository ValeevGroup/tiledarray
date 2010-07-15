#include "unit_test_config.h"
#include "global_fixture.h"
#include "shape_fixtures.h"

using namespace TiledArray;
/*
// =============================================================================
// DenseShape Test Suite

BOOST_FIXTURE_TEST_SUITE( dense_shape_suite, DenseShapeFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(DenseShapeT ds0(r));
  DenseShapeT ds0(r);
  BOOST_CHECK(ds0.is_initialized()); // Check that the shape is immediately available
  BOOST_CHECK_EQUAL(ds0.type(), detail::dense_shape);
  BOOST_CHECK_EQUAL(ds0.volume(), volume);
  BOOST_CHECK_EQUAL_COLLECTIONS(ds0.size().begin(), ds0.size().end(),
      size.begin(), size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ds0.start().begin(), ds0.start().end(),
      start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ds0.finish().begin(), ds0.finish().end(),
      finish.begin(), finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ds0.weight().begin(), ds0.weight().end(),
      weight.begin(), weight.end());

  BOOST_REQUIRE_NO_THROW(DenseShapeT ds1(r.size(), detail::decreasing_dimension_order));
  DenseShapeT ds1(r.size(), detail::decreasing_dimension_order);
  BOOST_CHECK(ds1.is_initialized()); // Check that the shape is immediately available
  BOOST_CHECK_EQUAL(ds1.type(), detail::dense_shape);
  BOOST_CHECK_EQUAL(ds1.volume(), volume);
  BOOST_CHECK_EQUAL_COLLECTIONS(ds1.size().begin(), ds1.size().end(),
      size.begin(), size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ds1.start().begin(), ds1.start().end(),
      start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ds1.finish().begin(), ds1.finish().end(),
      finish.begin(), finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ds1.weight().begin(), ds1.weight().end(),
      weight.begin(), weight.end());

  BOOST_REQUIRE_NO_THROW(DenseShapeT ds2(r.start(), r.finish(), detail::decreasing_dimension_order));
  DenseShapeT ds2(r.start(), r.finish(), detail::decreasing_dimension_order);
  BOOST_CHECK(ds2.is_initialized()); // Check that the shape is immediately available
  BOOST_CHECK_EQUAL(ds2.type(), detail::dense_shape);
  BOOST_CHECK_EQUAL(ds2.volume(), volume);
  BOOST_CHECK_EQUAL_COLLECTIONS(ds2.size().begin(), ds2.size().end(),
      size.begin(), size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ds2.start().begin(), ds2.start().end(),
      start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ds2.finish().begin(), ds2.finish().end(),
      finish.begin(), finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ds2.weight().begin(), ds2.weight().end(),
      weight.begin(), weight.end());
}

BOOST_AUTO_TEST_CASE( includes )
{
  // Check the points that are included
  BOOST_CHECK(ds.includes(index_type(0,0,0)).get());
  BOOST_CHECK(ds.includes(index_type(0,0,1)).get());
  BOOST_CHECK(ds.includes(index_type(0,0,2)).get());
  BOOST_CHECK(ds.includes(index_type(0,1,0)).get());
  BOOST_CHECK(ds.includes(index_type(0,1,1)).get());
  BOOST_CHECK(ds.includes(index_type(0,1,2)).get());

  BOOST_CHECK(ds.includes(0u).get());
  BOOST_CHECK(ds.includes(1u).get());
  BOOST_CHECK(ds.includes(2u).get());
  BOOST_CHECK(ds.includes(3u).get());
  BOOST_CHECK(ds.includes(4u).get());
  BOOST_CHECK(ds.includes(5u).get());

  // Check points outside the limits
  BOOST_CHECK(! ds.includes(index_type(1,0,0)).get());
  BOOST_CHECK(! ds.includes(index_type(1,0,1)).get());
  BOOST_CHECK(! ds.includes(index_type(1,0,2)).get());
  BOOST_CHECK(! ds.includes(index_type(1,1,0)).get());
  BOOST_CHECK(! ds.includes(index_type(1,1,1)).get());
  BOOST_CHECK(! ds.includes(index_type(1,1,2)).get());
  BOOST_CHECK(! ds.includes(index_type(0,2,0)).get());
  BOOST_CHECK(! ds.includes(index_type(0,2,1)).get());
  BOOST_CHECK(! ds.includes(index_type(0,2,2)).get());
  BOOST_CHECK(! ds.includes(index_type(0,0,3)).get());
  BOOST_CHECK(! ds.includes(index_type(0,1,3)).get());
  BOOST_CHECK(! ds.includes(index_type(1,2,0)).get());
  BOOST_CHECK(! ds.includes(index_type(1,2,1)).get());
  BOOST_CHECK(! ds.includes(index_type(1,2,2)).get());
  BOOST_CHECK(! ds.includes(index_type(1,0,3)).get());
  BOOST_CHECK(! ds.includes(index_type(1,1,3)).get());
  BOOST_CHECK(! ds.includes(index_type(0,2,3)).get());
  BOOST_CHECK(! ds.includes(index_type(1,2,3)).get());

  BOOST_CHECK(! ds.includes(6u).get());
  BOOST_CHECK(! ds.includes(7u).get());
}

BOOST_AUTO_TEST_SUITE_END()
*/
