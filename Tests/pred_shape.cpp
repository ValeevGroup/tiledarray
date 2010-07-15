#include "unit_test_config.h"
#include "global_fixture.h"
#include "shape_fixtures.h"

using namespace TiledArray;

/*
// =============================================================================
// PredicatedShape Test Suite

BOOST_FIXTURE_TEST_SUITE( pred_shape_suite, PredShapeFixture )

BOOST_AUTO_TEST_CASE( pred_clone )
{
  boost::shared_ptr<PredShapeT::PredInterface> p = ps.clone_pred();
  BOOST_CHECK(p.get() != NULL);
  BOOST_CHECK(p->check(0));
  BOOST_CHECK(! p->check(1));
  BOOST_CHECK(p->check(2));
  BOOST_CHECK(! p->check(3));

  BOOST_CHECK(p->type() == typeid(Even));
}

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(PredShapeT ps0(r, p));
  PredShapeT ps0(r, p);
  BOOST_CHECK(ps0.is_initialized()); // Check that the shape is immediately available
  BOOST_CHECK_EQUAL(ps0.type(), detail::predicated_shape);
  BOOST_CHECK_EQUAL(ps0.volume(), volume);
  BOOST_CHECK_EQUAL_COLLECTIONS(ps0.size().begin(), ps0.size().end(),
      size.begin(), size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ps0.start().begin(), ps0.start().end(),
      start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ps0.finish().begin(), ps0.finish().end(),
      finish.begin(), finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ps0.weight().begin(), ps0.weight().end(),
      weight.begin(), weight.end());

  BOOST_REQUIRE_NO_THROW(PredShapeT ps1(r.size(), detail::decreasing_dimension_order, p));
  PredShapeT ps1(r.size(), detail::decreasing_dimension_order, p);
  BOOST_CHECK(ps1.is_initialized()); // Check that the shape is immediately available
  BOOST_CHECK_EQUAL(ps1.type(), detail::predicated_shape);
  BOOST_CHECK_EQUAL(ps1.volume(), volume);
  BOOST_CHECK_EQUAL_COLLECTIONS(ps1.size().begin(), ps1.size().end(),
      size.begin(), size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ps1.start().begin(), ps1.start().end(),
      start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ps1.finish().begin(), ps1.finish().end(),
      finish.begin(), finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ps1.weight().begin(), ps1.weight().end(),
      weight.begin(), weight.end());

  BOOST_REQUIRE_NO_THROW(PredShapeT ps2(r.start(), r.finish(), detail::decreasing_dimension_order, p));
  PredShapeT ps2(r.start(), r.finish(), detail::decreasing_dimension_order, p);
  BOOST_CHECK(ps2.is_initialized()); // Check that the shape is immediately available
  BOOST_CHECK_EQUAL(ps2.type(), detail::predicated_shape);
  BOOST_CHECK_EQUAL(ps2.volume(), volume);
  BOOST_CHECK_EQUAL_COLLECTIONS(ps2.size().begin(), ps2.size().end(),
      size.begin(), size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ps2.start().begin(), ps2.start().end(),
      start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ps2.finish().begin(), ps2.finish().end(),
      finish.begin(), finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ps2.weight().begin(), ps2.weight().end(),
      weight.begin(), weight.end());

  BOOST_REQUIRE_NO_THROW(PredShapeT ps3(r, p));
  PredShapeT ps3(r, ps0.clone_pred());
  BOOST_CHECK(ps3.clone_pred()->type() == ps0.clone_pred()->type());
}

BOOST_AUTO_TEST_CASE( includes )
{
  // Check the points that are included
  BOOST_CHECK(ps.includes(index_type(0,0,0)).get());
  BOOST_CHECK(ps.includes(index_type(0,0,2)).get());
  BOOST_CHECK(ps.includes(index_type(0,1,1)).get());

  BOOST_CHECK(ps.includes(0u).get());
  BOOST_CHECK(ps.includes(2u).get());
  BOOST_CHECK(ps.includes(4u).get());

  // Check the points excluded by the predicate
  BOOST_CHECK(! ps.includes(index_type(0,0,1)).get());
  BOOST_CHECK(! ps.includes(index_type(0,1,0)).get());
  BOOST_CHECK(! ps.includes(index_type(0,1,2)).get());

  BOOST_CHECK(! ps.includes(1u).get());
  BOOST_CHECK(! ps.includes(3u).get());
  BOOST_CHECK(! ps.includes(5u).get());

  // Check points outside the limits
  BOOST_CHECK(! ps.includes(index_type(1,0,0)).get());
  BOOST_CHECK(! ps.includes(index_type(1,0,1)).get());
  BOOST_CHECK(! ps.includes(index_type(1,0,2)).get());
  BOOST_CHECK(! ps.includes(index_type(1,1,0)).get());
  BOOST_CHECK(! ps.includes(index_type(1,1,1)).get());
  BOOST_CHECK(! ps.includes(index_type(1,1,2)).get());
  BOOST_CHECK(! ps.includes(index_type(0,2,0)).get());
  BOOST_CHECK(! ps.includes(index_type(0,2,1)).get());
  BOOST_CHECK(! ps.includes(index_type(0,2,2)).get());
  BOOST_CHECK(! ps.includes(index_type(0,0,3)).get());
  BOOST_CHECK(! ps.includes(index_type(0,1,3)).get());
  BOOST_CHECK(! ps.includes(index_type(1,2,0)).get());
  BOOST_CHECK(! ps.includes(index_type(1,2,1)).get());
  BOOST_CHECK(! ps.includes(index_type(1,2,2)).get());
  BOOST_CHECK(! ps.includes(index_type(1,0,3)).get());
  BOOST_CHECK(! ps.includes(index_type(1,1,3)).get());
  BOOST_CHECK(! ps.includes(index_type(0,2,3)).get());
  BOOST_CHECK(! ps.includes(index_type(1,2,3)).get());
}

BOOST_AUTO_TEST_SUITE_END()
*/
