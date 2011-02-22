#include "TiledArray/shape.h"
#include "TiledArray/range.h"
#include "unit_test_config.h"
#include "global_fixture.h"
#include "range_fixture.h"
#include "shape_fixtures.h"

using namespace TiledArray;

// =============================================================================
// Shape Test Suite

BOOST_FIXTURE_TEST_SUITE( shape_suite, ShapeFixture )

BOOST_AUTO_TEST_CASE( cast_dense_shape )
{
  ShapePtr shape = dynamic_cast<ShapePtr>(&ds);
  BOOST_REQUIRE(shape != NULL);

  DenseShapeT* dense_shape = dynamic_cast<DenseShapeT*>(shape);
  BOOST_CHECK(dense_shape != NULL);
}

BOOST_AUTO_TEST_CASE( cast_sparse_shape )
{
  ShapePtr shape = dynamic_cast<ShapePtr>(&ss);
  BOOST_REQUIRE(shape != NULL);

  SparseShapeT* sparse_shape = dynamic_cast<SparseShapeT*>(shape);
  BOOST_CHECK(sparse_shape != NULL);
}

BOOST_AUTO_TEST_CASE( cast_pred_shape )
{
  ShapePtr shape = dynamic_cast<ShapePtr>(&ps);
  BOOST_REQUIRE(shape != NULL);

  PredShapeT* pred_shape = dynamic_cast<PredShapeT*>(shape);
  BOOST_CHECK(pred_shape != NULL);
}

BOOST_AUTO_TEST_SUITE_END()

