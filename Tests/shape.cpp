#include "TiledArray/shape.h"
#include "TiledArray/range.h"
#include "unit_test_config.h"
#include "global_fixture.h"
#include "range_fixture.h"
#include "shape_fixtures.h"

using namespace TiledArray;
/*
// =============================================================================
// Shape Test Suite

BOOST_FIXTURE_TEST_SUITE( shape_suite, ShapeFixture )

BOOST_AUTO_TEST_CASE( cast_dense_shape )
{
  ShapePtr shape = dynamic_cast<ShapePtr>(&ds);
  BOOST_REQUIRE(shape != NULL);
  BOOST_CHECK(shape->is_initialized());

  DenseShapeT* dense_shape = dynamic_cast<DenseShapeT*>(shape);
  BOOST_CHECK(dense_shape != NULL);
}

BOOST_AUTO_TEST_CASE( cast_sparse_shape )
{
  ShapePtr shape = dynamic_cast<ShapePtr>(&ss);
  BOOST_REQUIRE(shape != NULL);
  BOOST_CHECK(! shape->is_initialized());

  SparseShapeT* sparse_shape = dynamic_cast<SparseShapeT*>(shape);
  BOOST_CHECK(sparse_shape != NULL);
}

BOOST_AUTO_TEST_CASE( cast_pred_shape )
{
  ShapePtr shape = dynamic_cast<ShapePtr>(&ps);
  BOOST_REQUIRE(shape != NULL);
  BOOST_CHECK(shape->is_initialized());

  PredShapeT* pred_shape = dynamic_cast<PredShapeT*>(shape);
  BOOST_CHECK(pred_shape != NULL);
}

BOOST_AUTO_TEST_CASE( dense_shape_range )
{
  ShapePtr shape = dynamic_cast<ShapePtr>(&ds);
  BOOST_CHECK_EQUAL(shape->volume(), DenseShapeFixture::volume);
  BOOST_CHECK_EQUAL_COLLECTIONS(shape->size().begin(), shape->size().end(),
      DenseShapeFixture::size.begin(), DenseShapeFixture::size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(shape->start().begin(), shape->start().end(),
      DenseShapeFixture::start.begin(), DenseShapeFixture::start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(shape->finish().begin(), shape->finish().end(),
      DenseShapeFixture::finish.begin(), DenseShapeFixture::finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(shape->weight().begin(), shape->weight().end(),
      DenseShapeFixture::weight.begin(), DenseShapeFixture::weight.end());
}

BOOST_AUTO_TEST_CASE( sparse_shape_range )
{
  ShapePtr shape = dynamic_cast<ShapePtr>(&ss);
  BOOST_CHECK_EQUAL(shape->volume(), SparseShapeFixture::volume);
  BOOST_CHECK_EQUAL_COLLECTIONS(shape->size().begin(), shape->size().end(),
      SparseShapeFixture::size.begin(), SparseShapeFixture::size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(shape->start().begin(), shape->start().end(),
      SparseShapeFixture::start.begin(), SparseShapeFixture::start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(shape->finish().begin(), shape->finish().end(),
      SparseShapeFixture::finish.begin(), SparseShapeFixture::finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(shape->weight().begin(), shape->weight().end(),
      SparseShapeFixture::weight.begin(), SparseShapeFixture::weight.end());
}

BOOST_AUTO_TEST_CASE( pred_shape_range )
{
  ShapePtr shape = dynamic_cast<ShapePtr>(&ps);
  BOOST_CHECK_EQUAL(shape->volume(), PredShapeFixture::volume);
  BOOST_CHECK_EQUAL_COLLECTIONS(shape->size().begin(), shape->size().end(),
      PredShapeFixture::size.begin(), PredShapeFixture::size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(shape->start().begin(), shape->start().end(),
      PredShapeFixture::start.begin(), PredShapeFixture::start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(shape->finish().begin(), shape->finish().end(),
      PredShapeFixture::finish.begin(), PredShapeFixture::finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(shape->weight().begin(), shape->weight().end(),
      PredShapeFixture::weight.begin(), PredShapeFixture::weight.end());
}

BOOST_AUTO_TEST_CASE( dense_includes )
{
  ShapePtr shape = dynamic_cast<ShapePtr>(&ds);

  // Check the points that are included
  BOOST_CHECK(shape->includes(index_type(0,0,0)).get());
  BOOST_CHECK(shape->includes(index_type(0,0,1)).get());
  BOOST_CHECK(shape->includes(index_type(0,0,2)).get());
  BOOST_CHECK(shape->includes(index_type(0,1,0)).get());
  BOOST_CHECK(shape->includes(index_type(0,1,1)).get());
  BOOST_CHECK(shape->includes(index_type(0,1,2)).get());

  BOOST_CHECK(shape->includes(0u).get());
  BOOST_CHECK(shape->includes(1u).get());
  BOOST_CHECK(shape->includes(2u).get());
  BOOST_CHECK(shape->includes(3u).get());
  BOOST_CHECK(shape->includes(4u).get());
  BOOST_CHECK(shape->includes(5u).get());

  // Check points outside the limits
  BOOST_CHECK(! shape->includes(index_type(1,0,0)).get());
  BOOST_CHECK(! shape->includes(index_type(1,0,1)).get());
  BOOST_CHECK(! shape->includes(index_type(1,0,2)).get());
  BOOST_CHECK(! shape->includes(index_type(1,1,0)).get());
  BOOST_CHECK(! shape->includes(index_type(1,1,1)).get());
  BOOST_CHECK(! shape->includes(index_type(1,1,2)).get());
  BOOST_CHECK(! shape->includes(index_type(0,2,0)).get());
  BOOST_CHECK(! shape->includes(index_type(0,2,1)).get());
  BOOST_CHECK(! shape->includes(index_type(0,2,2)).get());
  BOOST_CHECK(! shape->includes(index_type(0,0,3)).get());
  BOOST_CHECK(! shape->includes(index_type(0,1,3)).get());
  BOOST_CHECK(! shape->includes(index_type(1,2,0)).get());
  BOOST_CHECK(! shape->includes(index_type(1,2,1)).get());
  BOOST_CHECK(! shape->includes(index_type(1,2,2)).get());
  BOOST_CHECK(! shape->includes(index_type(1,0,3)).get());
  BOOST_CHECK(! shape->includes(index_type(1,1,3)).get());
  BOOST_CHECK(! shape->includes(index_type(0,2,3)).get());
  BOOST_CHECK(! shape->includes(index_type(1,2,3)).get());

  BOOST_CHECK(! shape->includes(6u).get());
  BOOST_CHECK(! shape->includes(7u).get());
}

BOOST_AUTO_TEST_CASE( sparse_include )
{
  for(unsigned int i = 0; i < SparseShapeFixture::r.volume(); ++i) {
    if(ss.is_local(i))
      ss.add(i);
  }

  ss.set_initialized();


  ShapePtr shape = dynamic_cast<ShapePtr>(&ss);

  // Check the points that are included
  BOOST_CHECK(shape->includes(index_type(0,0,0)).get());
  BOOST_CHECK(shape->includes(index_type(0,0,1)).get());
  BOOST_CHECK(shape->includes(index_type(0,0,2)).get());
  BOOST_CHECK(shape->includes(index_type(0,1,0)).get());
  BOOST_CHECK(shape->includes(index_type(0,1,1)).get());
  BOOST_CHECK(shape->includes(index_type(0,1,2)).get());

  BOOST_CHECK(shape->includes(0u).get());
  BOOST_CHECK(shape->includes(1u).get());
  BOOST_CHECK(shape->includes(2u).get());
  BOOST_CHECK(shape->includes(3u).get());
  BOOST_CHECK(shape->includes(4u).get());
  BOOST_CHECK(shape->includes(5u).get());

  // Check points outside the limits
  BOOST_CHECK(! shape->includes(index_type(1,0,0)).get());
  BOOST_CHECK(! shape->includes(index_type(1,0,1)).get());
  BOOST_CHECK(! shape->includes(index_type(1,0,2)).get());
  BOOST_CHECK(! shape->includes(index_type(1,1,0)).get());
  BOOST_CHECK(! shape->includes(index_type(1,1,1)).get());
  BOOST_CHECK(! shape->includes(index_type(1,1,2)).get());
  BOOST_CHECK(! shape->includes(index_type(0,2,0)).get());
  BOOST_CHECK(! shape->includes(index_type(0,2,1)).get());
  BOOST_CHECK(! shape->includes(index_type(0,2,2)).get());
  BOOST_CHECK(! shape->includes(index_type(0,0,3)).get());
  BOOST_CHECK(! shape->includes(index_type(0,1,3)).get());
  BOOST_CHECK(! shape->includes(index_type(1,2,0)).get());
  BOOST_CHECK(! shape->includes(index_type(1,2,1)).get());
  BOOST_CHECK(! shape->includes(index_type(1,2,2)).get());
  BOOST_CHECK(! shape->includes(index_type(1,0,3)).get());
  BOOST_CHECK(! shape->includes(index_type(1,1,3)).get());
  BOOST_CHECK(! shape->includes(index_type(0,2,3)).get());
  BOOST_CHECK(! shape->includes(index_type(1,2,3)).get());

  BOOST_CHECK(! shape->includes(6u).get());
  BOOST_CHECK(! shape->includes(7u).get());

}

BOOST_AUTO_TEST_CASE( pred_includes )
{
  ShapePtr shape = dynamic_cast<ShapePtr>(&ps);

  // Check the points that are included
  BOOST_CHECK(shape->includes(index_type(0,0,0)).get());
  BOOST_CHECK(shape->includes(index_type(0,0,2)).get());
  BOOST_CHECK(shape->includes(index_type(0,1,1)).get());

  BOOST_CHECK(shape->includes(0u).get());
  BOOST_CHECK(shape->includes(2u).get());
  BOOST_CHECK(shape->includes(4u).get());

  // Check the points excluded by the predicate
  BOOST_CHECK(! shape->includes(index_type(0,0,1)).get());
  BOOST_CHECK(! shape->includes(index_type(0,1,0)).get());
  BOOST_CHECK(! shape->includes(index_type(0,1,2)).get());

  BOOST_CHECK(! shape->includes(1u).get());
  BOOST_CHECK(! shape->includes(3u).get());
  BOOST_CHECK(! shape->includes(5u).get());

  // Check points outside the limits
  BOOST_CHECK(! shape->includes(index_type(1,0,0)).get());
  BOOST_CHECK(! shape->includes(index_type(1,0,1)).get());
  BOOST_CHECK(! shape->includes(index_type(1,0,2)).get());
  BOOST_CHECK(! shape->includes(index_type(1,1,0)).get());
  BOOST_CHECK(! shape->includes(index_type(1,1,1)).get());
  BOOST_CHECK(! shape->includes(index_type(1,1,2)).get());
  BOOST_CHECK(! shape->includes(index_type(0,2,0)).get());
  BOOST_CHECK(! shape->includes(index_type(0,2,1)).get());
  BOOST_CHECK(! shape->includes(index_type(0,2,2)).get());
  BOOST_CHECK(! shape->includes(index_type(0,0,3)).get());
  BOOST_CHECK(! shape->includes(index_type(0,1,3)).get());
  BOOST_CHECK(! shape->includes(index_type(1,2,0)).get());
  BOOST_CHECK(! shape->includes(index_type(1,2,1)).get());
  BOOST_CHECK(! shape->includes(index_type(1,2,2)).get());
  BOOST_CHECK(! shape->includes(index_type(1,0,3)).get());
  BOOST_CHECK(! shape->includes(index_type(1,1,3)).get());
  BOOST_CHECK(! shape->includes(index_type(0,2,3)).get());
  BOOST_CHECK(! shape->includes(index_type(1,2,3)).get());
}

BOOST_AUTO_TEST_CASE( set_range )
{
  ShapePtr shape = dynamic_cast<ShapePtr>(&ps);

  PredShapeFixture::RangeN r0(index_type(0,0,0), index_type(6,6,6));
  shape->set_range(r0);

  BOOST_CHECK_EQUAL(shape->volume(), r0.volume());
  BOOST_CHECK_EQUAL_COLLECTIONS(shape->size().begin(), shape->size().end(),
      r0.size().begin(), r0.size().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(shape->start().begin(), shape->start().end(),
      r0.start().begin(), r0.start().end());
  BOOST_CHECK_EQUAL_COLLECTIONS(shape->finish().begin(), shape->finish().end(),
      r0.finish().begin(), r0.finish().end());
}

BOOST_AUTO_TEST_SUITE_END()
*/
