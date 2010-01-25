#include "TiledArray/array_dim.h"
#include "TiledArray/coordinates.h"
#include "TiledArray/range.h"
#include "unit_test_config.h"

using namespace TiledArray;

ArrayDimFixture::ArrayDimFixture() {
  s[0] = 2;
  s[1] = 3;
  s[2] = 4;

  w[0] = 12;
  w[1] = 4;
  w[2] = 1;

  v = 24;

  d.resize(s);
}



BOOST_FIXTURE_TEST_SUITE( array_dim_suite, ArrayDimFixture )

BOOST_AUTO_TEST_CASE( access )
{
  BOOST_CHECK_EQUAL( d.volume(), v); // check volume calculation
  BOOST_CHECK_EQUAL( d.size(), s);    // check the size accessor
  BOOST_CHECK_EQUAL( d.weight(), w);  // check weight accessor
}

BOOST_AUTO_TEST_CASE( includes )
{
  Range<std::size_t, 3> r(s);
  for(Range<std::size_t, 3>::const_iterator it = r.begin(); it != r.end(); ++it)
    BOOST_CHECK(d.includes( *it )); // check that all the expected indexes are
                                     // included.

  std::vector<index_type> p;
  p.push_back(index_type(2,2,3));
  p.push_back(index_type(1,3,3));
  p.push_back(index_type(1,2,4));
  p.push_back(index_type(2,3,4));

  for(std::vector<index_type>::const_iterator it = p.begin(); it != p.end(); ++it)
    BOOST_CHECK(! d.includes(*it));  // Check that elements outside the range
                                     // are not included.
}

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(ArrayDim3 d0); // Check default construction
  ArrayDim3 d0;
  BOOST_CHECK_EQUAL(d0.volume(), 0u);        // check for zero size with default
  ArrayDim3::size_array s0 = {{0,0,0}}; // construction.
  BOOST_CHECK_EQUAL(d0.size(), s0);
  BOOST_CHECK_EQUAL(d0.weight(), s0);

  BOOST_REQUIRE_NO_THROW(ArrayDim3 d1(s)); // check size constructor
  ArrayDim3 d1(s);
  BOOST_CHECK_EQUAL(d1.volume(), v);  // check that a has correct va
  BOOST_CHECK_EQUAL(d1.size(), s);
  BOOST_CHECK_EQUAL(d1.weight(), w);

  BOOST_REQUIRE_NO_THROW(ArrayDim3 d2(d));
  ArrayDim3 d2(d);
  BOOST_CHECK_EQUAL(d2.volume(), d.volume());
  BOOST_CHECK_EQUAL(d2.size(), d.size());
  BOOST_CHECK_EQUAL(d2.weight(), d.weight());
}

BOOST_AUTO_TEST_CASE( ordinal )
{
  Range<std::size_t, 3> r(s);
  ArrayDim3::ordinal_type o = 0;
  for(Range<std::size_t, 3>::const_iterator it = r.begin(); it != r.end(); ++it, ++o)
    BOOST_CHECK_EQUAL(d.ordinal( *it ), o); // check ordinal calculation and order

#ifdef TA_EXCEPTION_ERROR
  index_type p(2,3,4);
  BOOST_CHECK_THROW(d.ordinal(p), std::out_of_range); // check for throw with
                                                  // an out of range element.
#endif
}

BOOST_AUTO_TEST_CASE( ordinal_fortran )
{
  typedef detail::ArrayDim<std::size_t, 3, LevelTag<0>, CoordinateSystem<3, detail::increasing_dimension_order> > FArrayDim3;
  typedef Range<std::size_t, 3, LevelTag<0>, CoordinateSystem<3, detail::increasing_dimension_order> > range_type;

  FArrayDim3 d1(s);
  range_type r(s);
  FArrayDim3::ordinal_type o = 0;
  for(range_type::const_iterator it = r.begin(); it != r.end(); ++it, ++o)
    BOOST_CHECK_EQUAL(d1.ordinal( *it ), o); // check ordinal calculation and
                                             // order for fortran ordering.
}

BOOST_AUTO_TEST_CASE( resize )
{
  ArrayDim3 d1;
  d1.resize(s);
  BOOST_CHECK_EQUAL(d1.volume(), v); // check for correct resizing.
  BOOST_CHECK_EQUAL(d1.size(), s);
  BOOST_CHECK_EQUAL(d1.weight(), w);
}

BOOST_AUTO_TEST_SUITE_END()
