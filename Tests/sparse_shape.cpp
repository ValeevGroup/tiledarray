#include "unit_test_config.h"
#include "global_fixture.h"
#include "shape_fixtures.h"

using namespace TiledArray;

// =============================================================================
// SparseShape Test Suite



/*
BOOST_FIXTURE_TEST_SUITE( sparse_shape_suite, SparseShapeFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  const boost::shared_ptr<madness::WorldDCPmapInterface<std::size_t> > pm(
      dynamic_cast<madness::WorldDCPmapInterface<std::size_t>*>(
          new madness::WorldDCDefaultPmap<std::size_t>(world)));
  BOOST_REQUIRE(pm.get() != NULL);

  BOOST_REQUIRE_NO_THROW(SparseShapeT ss0(world, r));
  SparseShapeT ss0(world, r);
  BOOST_CHECK(! ss0.is_initialized()); // Check that the shape is not immediately available
  BOOST_CHECK_EQUAL(ss0.type(), detail::sparse_shape);
  BOOST_CHECK_EQUAL(ss0.type(), detail::sparse_shape);
  BOOST_CHECK_EQUAL(ss0.volume(), volume);
  BOOST_CHECK_EQUAL_COLLECTIONS(ss0.size().begin(), ss0.size().end(),
      size.begin(), size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss0.start().begin(), ss0.start().end(),
      start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss0.finish().begin(), ss0.finish().end(),
      finish.begin(), finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss0.weight().begin(), ss0.weight().end(),
      weight.begin(), weight.end());

  BOOST_REQUIRE_NO_THROW(SparseShapeT ss2(world, r, pm));
  SparseShapeT ss2(world, r, pm);
  BOOST_CHECK(! ss2.is_initialized()); // Check that the shape is not immediately available
  BOOST_CHECK_EQUAL(ss2.type(), detail::sparse_shape);
  BOOST_CHECK_EQUAL(ss2.type(), detail::sparse_shape);
  BOOST_CHECK_EQUAL(ss2.volume(), volume);
  BOOST_CHECK_EQUAL_COLLECTIONS(ss2.size().begin(), ss2.size().end(),
      size.begin(), size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss2.start().begin(), ss2.start().end(),
      start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss2.finish().begin(), ss2.finish().end(),
      finish.begin(), finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss2.weight().begin(), ss2.weight().end(),
      weight.begin(), weight.end());

  BOOST_REQUIRE_NO_THROW(SparseShapeT ss4(world, r.size(), detail::decreasing_dimension_order, pm));
  SparseShapeT ss4(world, r.size(), detail::decreasing_dimension_order, pm);
  BOOST_CHECK(! ss4.is_initialized()); // Check that the shape is immediately available
  BOOST_CHECK_EQUAL(ss4.type(), detail::sparse_shape);
  BOOST_CHECK_EQUAL(ss4.volume(), volume);
  BOOST_CHECK_EQUAL_COLLECTIONS(ss4.size().begin(), ss4.size().end(),
      size.begin(), size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss4.start().begin(), ss4.start().end(),
      start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss4.finish().begin(), ss4.finish().end(),
      finish.begin(), finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss4.weight().begin(), ss4.weight().end(),
      weight.begin(), weight.end());

  BOOST_REQUIRE_NO_THROW(SparseShapeT ss6(world, r.start(), r.finish(), detail::decreasing_dimension_order, pm));
  SparseShapeT ss6(world, r.start(), r.finish(), detail::decreasing_dimension_order, pm);
  BOOST_CHECK(! ss6.is_initialized()); // Check that the shape is immediately available
  BOOST_CHECK_EQUAL(ss6.type(), detail::sparse_shape);
  BOOST_CHECK_EQUAL(ss6.volume(), volume);
  BOOST_CHECK_EQUAL_COLLECTIONS(ss6.size().begin(), ss6.size().end(),
      size.begin(), size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss6.start().begin(), ss6.start().end(),
      start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss6.finish().begin(), ss6.finish().end(),
      finish.begin(), finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss6.weight().begin(), ss6.weight().end(),
      weight.begin(), weight.end());
}

BOOST_AUTO_TEST_CASE( add_ordinal )
{
  for(unsigned int i = 0; i < r.volume(); ++i) {
    if(ss.is_local(i))
      ss.add(i);
#ifdef TA_EXCEPTION_ERROR
    else
      BOOST_CHECK_THROW(ss.add(i), std::runtime_error);
#endif
  }

  // Check the out of range assertion.
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(ss.add(6), std::out_of_range);
#endif
  ss.set_initialized();

  // Find the first non local tile and try to add it to make sure an exception
  // is thrown after set_initialized is called.
  for(unsigned int i = 0; i < r.volume(); ++i) {
    if(! ss.is_local(i)) {
      BOOST_CHECK_THROW(ss.add(i), std::runtime_error);
      break;
    }
  }

  // Check the points that are included
  BOOST_CHECK(ss.includes(index_type(0,0,0)).get());
  BOOST_CHECK(ss.includes(index_type(0,0,1)).get());
  BOOST_CHECK(ss.includes(index_type(0,0,2)).get());
  BOOST_CHECK(ss.includes(index_type(0,1,0)).get());
  BOOST_CHECK(ss.includes(index_type(0,1,1)).get());
  BOOST_CHECK(ss.includes(index_type(0,1,2)).get());

  BOOST_CHECK(ss.includes(0u).get());
  BOOST_CHECK(ss.includes(1u).get());
  BOOST_CHECK(ss.includes(2u).get());
  BOOST_CHECK(ss.includes(3u).get());
  BOOST_CHECK(ss.includes(4u).get());
  BOOST_CHECK(ss.includes(5u).get());

  // Check points outside the limits
  BOOST_CHECK(! ss.includes(index_type(1,0,0)).get());
  BOOST_CHECK(! ss.includes(index_type(1,0,1)).get());
  BOOST_CHECK(! ss.includes(index_type(1,0,2)).get());
  BOOST_CHECK(! ss.includes(index_type(1,1,0)).get());
  BOOST_CHECK(! ss.includes(index_type(1,1,1)).get());
  BOOST_CHECK(! ss.includes(index_type(1,1,2)).get());
  BOOST_CHECK(! ss.includes(index_type(0,2,0)).get());
  BOOST_CHECK(! ss.includes(index_type(0,2,1)).get());
  BOOST_CHECK(! ss.includes(index_type(0,2,2)).get());
  BOOST_CHECK(! ss.includes(index_type(0,0,3)).get());
  BOOST_CHECK(! ss.includes(index_type(0,1,3)).get());
  BOOST_CHECK(! ss.includes(index_type(1,2,0)).get());
  BOOST_CHECK(! ss.includes(index_type(1,2,1)).get());
  BOOST_CHECK(! ss.includes(index_type(1,2,2)).get());
  BOOST_CHECK(! ss.includes(index_type(1,0,3)).get());
  BOOST_CHECK(! ss.includes(index_type(1,1,3)).get());
  BOOST_CHECK(! ss.includes(index_type(0,2,3)).get());
  BOOST_CHECK(! ss.includes(index_type(1,2,3)).get());

  BOOST_CHECK(! ss.includes(6u).get());
  BOOST_CHECK(! ss.includes(7u).get());
}

BOOST_AUTO_TEST_CASE( add_index )
{
  index_type i(0,0,0);
  if(ss.is_local(i))
    ss.add(i);
#ifdef TA_EXCEPTION_ERROR
  else
    BOOST_CHECK_THROW(ss.add(i), std::runtime_error);
#endif

  i = index_type(0,0,1);
  if(ss.is_local(i))
    ss.add(i);
#ifdef TA_EXCEPTION_ERROR
  else
    BOOST_CHECK_THROW(ss.add(i), std::runtime_error);
#endif

  i = index_type(0,0,2);
  if(ss.is_local(i))
    ss.add(i);
#ifdef TA_EXCEPTION_ERROR
  else
    BOOST_CHECK_THROW(ss.add(i), std::runtime_error);
#endif

  i = index_type(0,1,0);
  if(ss.is_local(i))
    ss.add(i);
#ifdef TA_EXCEPTION_ERROR
  else
    BOOST_CHECK_THROW(ss.add(i), std::runtime_error);
#endif

  i = index_type(0,1,1);
  if(ss.is_local(i))
    ss.add(i);
#ifdef TA_EXCEPTION_ERROR
  else
    BOOST_CHECK_THROW(ss.add(i), std::runtime_error);
#endif

  i = index_type(0,1,2);
  if(ss.is_local(i))
    ss.add(i);
#ifdef TA_EXCEPTION_ERROR
  else
    BOOST_CHECK_THROW(ss.add(i), std::runtime_error);
#endif

  // Check the out of range assertion.
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(ss.add(index_type(1,2,3)), std::out_of_range);
#endif
  ss.set_initialized();

  // Make sure an exception is thrown after_set() initialized is called.
  BOOST_CHECK_THROW(ss.add(i), std::runtime_error);

  // Check the points that are included
  BOOST_CHECK(ss.includes(index_type(0,0,0)).get());
  BOOST_CHECK(ss.includes(index_type(0,0,1)).get());
  BOOST_CHECK(ss.includes(index_type(0,0,2)).get());
  BOOST_CHECK(ss.includes(index_type(0,1,0)).get());
  BOOST_CHECK(ss.includes(index_type(0,1,1)).get());
  BOOST_CHECK(ss.includes(index_type(0,1,2)).get());

  BOOST_CHECK(ss.includes(0u).get());
  BOOST_CHECK(ss.includes(1u).get());
  BOOST_CHECK(ss.includes(2u).get());
  BOOST_CHECK(ss.includes(3u).get());
  BOOST_CHECK(ss.includes(4u).get());
  BOOST_CHECK(ss.includes(5u).get());

  // Check points outside the limits
  BOOST_CHECK(! ss.includes(index_type(1,0,0)).get());
  BOOST_CHECK(! ss.includes(index_type(1,0,1)).get());
  BOOST_CHECK(! ss.includes(index_type(1,0,2)).get());
  BOOST_CHECK(! ss.includes(index_type(1,1,0)).get());
  BOOST_CHECK(! ss.includes(index_type(1,1,1)).get());
  BOOST_CHECK(! ss.includes(index_type(1,1,2)).get());
  BOOST_CHECK(! ss.includes(index_type(0,2,0)).get());
  BOOST_CHECK(! ss.includes(index_type(0,2,1)).get());
  BOOST_CHECK(! ss.includes(index_type(0,2,2)).get());
  BOOST_CHECK(! ss.includes(index_type(0,0,3)).get());
  BOOST_CHECK(! ss.includes(index_type(0,1,3)).get());
  BOOST_CHECK(! ss.includes(index_type(1,2,0)).get());
  BOOST_CHECK(! ss.includes(index_type(1,2,1)).get());
  BOOST_CHECK(! ss.includes(index_type(1,2,2)).get());
  BOOST_CHECK(! ss.includes(index_type(1,0,3)).get());
  BOOST_CHECK(! ss.includes(index_type(1,1,3)).get());
  BOOST_CHECK(! ss.includes(index_type(0,2,3)).get());
  BOOST_CHECK(! ss.includes(index_type(1,2,3)).get());

  BOOST_CHECK(! ss.includes(6u).get());
  BOOST_CHECK(! ss.includes(7u).get());
}

//BOOST_AUTO_TEST_CASE( add_future )
//{
//  madness::Future<bool> f1;
//  madness::Future<bool> f2;
//  ProcessID owner = ss.owner(0);
//
//  if(world.rand() == owner)
//    ss.add<std::logical_or<bool> >(0u, f1, f2);
//
//  ss.set_initialized();
//
//  world.gop.fence();
//
//  madness::Future<bool> result = ss.includes(0u);
//  world.gop.fence();
//  BOOST_CHECK(! result.probe());
//  f1.set(true);
//  world.gop.fence();
//  BOOST_CHECK(! result.probe());
//  f2.set(false);
//  world.gop.fence();
//  BOOST_CHECK(result.probe());
//
//  BOOST_CHECK(result.get());
//}

BOOST_AUTO_TEST_SUITE_END()
*/
