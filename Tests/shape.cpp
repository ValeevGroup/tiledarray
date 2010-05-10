#include "TiledArray/shape.h"
#include "TiledArray/range.h"
#include "unit_test_config.h"
#include "global_fixture.h"
#include "range_fixture.h"

using namespace TiledArray;

// =============================================================================
// DenseShape Test Suite

struct RangeShapeFixture : public RangeFixture {
  typedef DenseShape<Range3> DenseShape3;

  RangeShapeFixture() : pr(&r, &no_delete) {

  }

  static void no_delete(Range3*) { /* do nothing */ }

  boost::shared_ptr<Range3> pr;
};

struct DenseShapeFixture : public RangeShapeFixture {
  typedef DenseShape<Range3> DenseShape3;

  DenseShapeFixture() : ds(pr)
  { }

  DenseShape3 ds;
};

BOOST_FIXTURE_TEST_SUITE( dense_shape_suite, DenseShapeFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(DenseShape3 ds0(pr));
  DenseShape3 ds0(pr);
  BOOST_CHECK(ds0.is_initialized()); // Check that the shape is immediately available
  BOOST_CHECK_EQUAL(ds0.type(), detail::dense_shape);
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

  BOOST_CHECK(ds.includes(0).get());
  BOOST_CHECK(ds.includes(1).get());
  BOOST_CHECK(ds.includes(2).get());
  BOOST_CHECK(ds.includes(3).get());
  BOOST_CHECK(ds.includes(4).get());
  BOOST_CHECK(ds.includes(5).get());

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

  BOOST_CHECK(! ds.includes(6).get());
  BOOST_CHECK(! ds.includes(7).get());
}

BOOST_AUTO_TEST_SUITE_END()

// =============================================================================
// SparseShape Test Suite

struct SparseShapeFixture : public RangeShapeFixture {
  typedef SparseShape<Range3> SparseShape3;



  SparseShapeFixture() : world(* GlobalFixture::world), ss(world, pr)
  { }

  madness::World& world;
  SparseShape3 ss;
};


BOOST_FIXTURE_TEST_SUITE( sparse_shape_suite, SparseShapeFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(SparseShape3 ss0(world, pr));
  SparseShape3 ss0(world, pr);
  BOOST_CHECK(! ss0.is_initialized()); // Check that the shape is not immediately available
  BOOST_CHECK_EQUAL(ss0.type(), detail::sparse_shape);

  madness::Hash_private::defhashT<std::size_t> h;
  BOOST_REQUIRE_NO_THROW(SparseShape3 ss1(world, pr, h));
  SparseShape3 ss1(world, pr, h);
  BOOST_CHECK(! ss1.is_initialized()); // Check that the shape is not immediately available
  BOOST_CHECK_EQUAL(ss1.type(), detail::sparse_shape);

  const boost::shared_ptr<madness::WorldDCDefaultPmap<std::size_t, madness::Hash_private::defhashT<std::size_t> > > pm;
  BOOST_REQUIRE_NO_THROW(SparseShape3 ss2(world, pr, pm));
  SparseShape3 ss2(world, pr, pm);
  BOOST_CHECK(! ss2.is_initialized()); // Check that the shape is not immediately available
  BOOST_CHECK_EQUAL(ss2.type(), detail::sparse_shape);
}


BOOST_AUTO_TEST_CASE( add_ordinal )
{
  for(unsigned int i = 0; i < pr->volume(); ++i) {
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
  for(unsigned int i = 0; i < pr->volume(); ++i) {
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

  BOOST_CHECK(ss.includes(0).get());
  BOOST_CHECK(ss.includes(1).get());
  BOOST_CHECK(ss.includes(2).get());
  BOOST_CHECK(ss.includes(3).get());
  BOOST_CHECK(ss.includes(4).get());
  BOOST_CHECK(ss.includes(5).get());

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

  BOOST_CHECK(! ss.includes(6).get());
  BOOST_CHECK(! ss.includes(7).get());
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

  i[2] = 1;
  if(ss.is_local(i))
    ss.add(i);
#ifdef TA_EXCEPTION_ERROR
  else
    BOOST_CHECK_THROW(ss.add(i), std::runtime_error);
#endif

  i[2] = 2;
  if(ss.is_local(i))
    ss.add(i);
#ifdef TA_EXCEPTION_ERROR
  else
    BOOST_CHECK_THROW(ss.add(i), std::runtime_error);
#endif

  i[1] = 1;
  i[2] = 0;
  if(ss.is_local(i))
    ss.add(i);
#ifdef TA_EXCEPTION_ERROR
  else
    BOOST_CHECK_THROW(ss.add(i), std::runtime_error);
#endif

  i[2] = 1;
  if(ss.is_local(i))
    ss.add(i);
#ifdef TA_EXCEPTION_ERROR
  else
    BOOST_CHECK_THROW(ss.add(i), std::runtime_error);
#endif

  i[2] = 2;
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

  BOOST_CHECK(ss.includes(0).get());
  BOOST_CHECK(ss.includes(1).get());
  BOOST_CHECK(ss.includes(2).get());
  BOOST_CHECK(ss.includes(3).get());
  BOOST_CHECK(ss.includes(4).get());
  BOOST_CHECK(ss.includes(5).get());

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

  BOOST_CHECK(! ss.includes(6).get());
  BOOST_CHECK(! ss.includes(7).get());
}

BOOST_AUTO_TEST_SUITE_END()

// =============================================================================
// PredicatedShape Test Suite

struct PredShapeFixture : public RangeShapeFixture {
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

  typedef PredShape<Range3, Even> EvenShape3;

  PredShapeFixture() : p(pr->weight()), ps(pr, p)
  { }

  Even p;
  EvenShape3 ps;
};

BOOST_FIXTURE_TEST_SUITE( pred_shape_suite, PredShapeFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(EvenShape3 ps0(pr, p));
  EvenShape3 ps0(pr, p);
  BOOST_CHECK(ps0.is_initialized()); // Check that the shape is immediately available
  BOOST_CHECK_EQUAL(ps0.type(), detail::predicated_shape);
}

BOOST_AUTO_TEST_CASE( includes )
{
  // Check the points that are included
  BOOST_CHECK(ps.includes(index_type(0,0,0)).get());
  BOOST_CHECK(ps.includes(index_type(0,0,2)).get());
  BOOST_CHECK(ps.includes(index_type(0,1,1)).get());

  BOOST_CHECK(ps.includes(0).get());
  BOOST_CHECK(ps.includes(2).get());
  BOOST_CHECK(ps.includes(4).get());

  // Check the points excluded by the predicate
  BOOST_CHECK(! ps.includes(index_type(0,0,1)).get());
  BOOST_CHECK(! ps.includes(index_type(0,1,0)).get());
  BOOST_CHECK(! ps.includes(index_type(0,1,2)).get());

  BOOST_CHECK(! ps.includes(1).get());
  BOOST_CHECK(! ps.includes(3).get());
  BOOST_CHECK(! ps.includes(5).get());

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

  BOOST_CHECK(shape->includes(0).get());
  BOOST_CHECK(shape->includes(1).get());
  BOOST_CHECK(shape->includes(2).get());
  BOOST_CHECK(shape->includes(3).get());
  BOOST_CHECK(shape->includes(4).get());
  BOOST_CHECK(shape->includes(5).get());

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

  BOOST_CHECK(! shape->includes(6).get());
  BOOST_CHECK(! shape->includes(7).get());
}

BOOST_AUTO_TEST_CASE( sparse_include )
{
  for(unsigned int i = 0; i < SparseShapeFixture::pr->volume(); ++i) {
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

  BOOST_CHECK(shape->includes(0).get());
  BOOST_CHECK(shape->includes(1).get());
  BOOST_CHECK(shape->includes(2).get());
  BOOST_CHECK(shape->includes(3).get());
  BOOST_CHECK(shape->includes(4).get());
  BOOST_CHECK(shape->includes(5).get());

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

  BOOST_CHECK(! shape->includes(6).get());
  BOOST_CHECK(! shape->includes(7).get());

}

BOOST_AUTO_TEST_CASE( pred_includes )
{
  ShapePtr shape = dynamic_cast<ShapePtr>(&ps);

  // Check the points that are included
  BOOST_CHECK(shape->includes(index_type(0,0,0)).get());
  BOOST_CHECK(shape->includes(index_type(0,0,2)).get());
  BOOST_CHECK(shape->includes(index_type(0,1,1)).get());

  BOOST_CHECK(shape->includes(0).get());
  BOOST_CHECK(shape->includes(2).get());
  BOOST_CHECK(shape->includes(4).get());

  // Check the points excluded by the predicate
  BOOST_CHECK(! shape->includes(index_type(0,0,1)).get());
  BOOST_CHECK(! shape->includes(index_type(0,1,0)).get());
  BOOST_CHECK(! shape->includes(index_type(0,1,2)).get());

  BOOST_CHECK(! shape->includes(1).get());
  BOOST_CHECK(! shape->includes(3).get());
  BOOST_CHECK(! shape->includes(5).get());

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

BOOST_AUTO_TEST_SUITE_END()
