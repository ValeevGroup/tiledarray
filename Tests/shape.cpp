#include "TiledArray/shape.h"
#include "TiledArray/range.h"
#include "unit_test_config.h"
#include "global_fixture.h"
#include "range_fixture.h"

using namespace TiledArray;

// =============================================================================
// DenseShape Test Suite

struct DenseShapeFixture : public RangeFixture {
  typedef DenseShape<std::size_t> DenseShape3;

  DenseShapeFixture() : ds(r)
  { }

  DenseShape3 ds;
};

BOOST_FIXTURE_TEST_SUITE( dense_shape_suite, DenseShapeFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(DenseShape3 ds0(r));
  DenseShape3 ds0(r);
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

  BOOST_REQUIRE_NO_THROW(DenseShape3 ds1(r.size(), detail::decreasing_dimension_order));
  DenseShape3 ds1(r.size(), detail::decreasing_dimension_order);
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

  BOOST_REQUIRE_NO_THROW(DenseShape3 ds2(r.start(), r.finish(), detail::decreasing_dimension_order));
  DenseShape3 ds2(r.start(), r.finish(), detail::decreasing_dimension_order);
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

// =============================================================================
// SparseShape Test Suite

struct SparseShapeFixture : public RangeFixture {
  typedef SparseShape<std::size_t> SparseShape3;



  SparseShapeFixture() : world(* GlobalFixture::world), ss(world, r)
  { }

  madness::World& world;
  SparseShape3 ss;
};


BOOST_FIXTURE_TEST_SUITE( sparse_shape_suite, SparseShapeFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  madness::Hash_private::defhashT<std::size_t> h;
  const boost::shared_ptr<madness::WorldDCPmapInterface<std::size_t> > pm(
      dynamic_cast<madness::WorldDCPmapInterface<std::size_t>*>(
          new madness::WorldDCDefaultPmap<std::size_t>(world)));
  BOOST_REQUIRE(pm.get() != NULL);

  BOOST_REQUIRE_NO_THROW(SparseShape3 ss0(world, r));
  SparseShape3 ss0(world, r);
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

  BOOST_REQUIRE_NO_THROW(SparseShape3 ss1(world, r, h));
  SparseShape3 ss1(world, r, h);
  BOOST_CHECK(! ss1.is_initialized()); // Check that the shape is not immediately available
  BOOST_CHECK_EQUAL(ss1.type(), detail::sparse_shape);
  BOOST_CHECK_EQUAL(ss1.type(), detail::sparse_shape);
  BOOST_CHECK_EQUAL(ss1.volume(), volume);
  BOOST_CHECK_EQUAL_COLLECTIONS(ss1.size().begin(), ss1.size().end(),
      size.begin(), size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss1.start().begin(), ss1.start().end(),
      start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss1.finish().begin(), ss1.finish().end(),
      finish.begin(), finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss1.weight().begin(), ss1.weight().end(),
      weight.begin(), weight.end());

  BOOST_REQUIRE_NO_THROW(SparseShape3 ss2(world, r, pm));
  SparseShape3 ss2(world, r, pm);
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

  BOOST_REQUIRE_NO_THROW(SparseShape3 ss3(world, r.size(), detail::decreasing_dimension_order, h));
  SparseShape3 ss3(world, r.size(), detail::decreasing_dimension_order, h);
  BOOST_CHECK(! ss3.is_initialized()); // Check that the shape is immediately available
  BOOST_CHECK_EQUAL(ss3.type(), detail::sparse_shape);
  BOOST_CHECK_EQUAL(ss3.volume(), volume);
  BOOST_CHECK_EQUAL_COLLECTIONS(ss3.size().begin(), ss3.size().end(),
      size.begin(), size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss3.start().begin(), ss3.start().end(),
      start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss3.finish().begin(), ss3.finish().end(),
      finish.begin(), finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss3.weight().begin(), ss3.weight().end(),
      weight.begin(), weight.end());

  BOOST_REQUIRE_NO_THROW(SparseShape3 ss4(world, r.size(), detail::decreasing_dimension_order, pm));
  SparseShape3 ss4(world, r.size(), detail::decreasing_dimension_order, pm);
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

  BOOST_REQUIRE_NO_THROW(SparseShape3 ss5(world, r.start(), r.finish(), detail::decreasing_dimension_order, h));
  SparseShape3 ss5(world, r.start(), r.finish(), detail::decreasing_dimension_order, h);
  BOOST_CHECK(! ss5.is_initialized()); // Check that the shape is immediately available
  BOOST_CHECK_EQUAL(ss5.type(), detail::sparse_shape);
  BOOST_CHECK_EQUAL(ss5.volume(), volume);
  BOOST_CHECK_EQUAL_COLLECTIONS(ss5.size().begin(), ss5.size().end(),
      size.begin(), size.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss5.start().begin(), ss5.start().end(),
      start.begin(), start.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss5.finish().begin(), ss5.finish().end(),
      finish.begin(), finish.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(ss5.weight().begin(), ss5.weight().end(),
      weight.begin(), weight.end());

  BOOST_REQUIRE_NO_THROW(SparseShape3 ss6(world, r.start(), r.finish(), detail::decreasing_dimension_order, pm));
  SparseShape3 ss6(world, r.start(), r.finish(), detail::decreasing_dimension_order, pm);
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

BOOST_AUTO_TEST_SUITE_END()

// =============================================================================
// PredicatedShape Test Suite

struct PredShapeFixture : public RangeFixture {
  struct Even {
    template<std::size_t N>
    Even(const boost::array<std::size_t, N>& w) : weight(w) { }

    bool operator()(std::size_t i) const {
      return (i % 2) == 0;
    }

    template<typename I>
    bool operator()(detail::ArrayRef<I> i) const {
      return operator()(std::inner_product(i.begin(), i.end(), weight.begin(), std::size_t(0)));
    }

    detail::ArrayRef<const std::size_t> weight;
  }; // struct Even

  typedef PredShape<std::size_t, Even> EvenShape3;

  PredShapeFixture() : p(r.weight()), ps(r, p)
  { }

  Even p;
  EvenShape3 ps;
};

BOOST_FIXTURE_TEST_SUITE( pred_shape_suite, PredShapeFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(EvenShape3 ps0(r, p));
  EvenShape3 ps0(r, p);
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

  BOOST_REQUIRE_NO_THROW(EvenShape3 ps1(r.size(), detail::decreasing_dimension_order, p));
  EvenShape3 ps1(r.size(), detail::decreasing_dimension_order, p);
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

  BOOST_REQUIRE_NO_THROW(EvenShape3 ps2(r.start(), r.finish(), detail::decreasing_dimension_order, p));
  EvenShape3 ps2(r.start(), r.finish(), detail::decreasing_dimension_order, p);
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

// =============================================================================
// Shape Test Suite

struct ShapeFixture : public PredShapeFixture, public SparseShapeFixture, public DenseShapeFixture {
  typedef Shape<std::size_t>* ShapePtr;

};

BOOST_FIXTURE_TEST_SUITE( shape_suite, ShapeFixture )

BOOST_AUTO_TEST_CASE( cast_dense_shape )
{
  ShapePtr shape = dynamic_cast<ShapePtr>(&ds);
  BOOST_REQUIRE(shape != NULL);
  BOOST_CHECK(shape->is_initialized());
}

BOOST_AUTO_TEST_CASE( cast_sparse_shape )
{
  ShapePtr shape = dynamic_cast<ShapePtr>(&ss);
  BOOST_REQUIRE(shape != NULL);
  BOOST_CHECK(! shape->is_initialized());
}

BOOST_AUTO_TEST_CASE( cast_pred_shape )
{
  ShapePtr shape = dynamic_cast<ShapePtr>(&ps);
  BOOST_REQUIRE(shape != NULL);
  BOOST_CHECK(shape->is_initialized());
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

  PredShapeFixture::Range3 r0(index_type(0,0,0), index_type(6,6,6));
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
