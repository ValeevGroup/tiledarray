#include "coordinates.h"
#include "array_storage.h"
#include "iterationtest.h"
#include "permutation.h"
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

using namespace TiledArray;

struct ArrayDimFixture {
  typedef detail::ArrayDim<3, LevelTag<0> > ArrayDim3;
  typedef ArrayDim3::index_type index_type;
  typedef ArrayDim3::size_array size_array;

  ArrayDimFixture() {
    s[0] = 2;
    s[1] = 3;
    s[2] = 4;

    w[0] = 12;
    w[1] = 4;
    w[2] = 1;

    v = 24;

    a.resize(s);
  }
  ~ArrayDimFixture() { }

  size_array s;
  size_array w;
  std::size_t v;
  ArrayDim3 a;
};


BOOST_FIXTURE_TEST_SUITE( array_dim_suite, ArrayDimFixture )

BOOST_AUTO_TEST_CASE( access )
{
  BOOST_CHECK_EQUAL( a.volume(), v); // check volume calculation
  BOOST_CHECK_EQUAL( a.size(), s);    // check the size accessor
  BOOST_CHECK_EQUAL( a.weight(), w);  // check weight accessor
}

BOOST_AUTO_TEST_CASE( includes )
{
  Range<std::size_t, 3> b(s);
  for(Range<std::size_t, 3>::const_iterator it = b.begin(); it != b.end(); ++it)
    BOOST_CHECK(a.includes( *it )); // check that all the expected indexes are
                                     // included.

  std::vector<index_type> p;
  p.push_back(index_type(2,2,3));
  p.push_back(index_type(1,3,3));
  p.push_back(index_type(1,2,4));
  p.push_back(index_type(2,3,4));

  for(std::vector<index_type>::const_iterator it = p.begin(); it != p.end(); ++it)
    BOOST_CHECK(! a.includes(*it));  // Check that elements outside the range
                                     // are not included.
}

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(ArrayDim3 a0); // Check default construction
  ArrayDim3 a0;
  BOOST_CHECK_EQUAL(a0.volume(), 0);        // check for zero size with default
  ArrayDim3::size_array s0 = {{0,0,0}}; // construction.
  BOOST_CHECK_EQUAL(a0.size(), s0);
  BOOST_CHECK_EQUAL(a0.weight(), s0);

  BOOST_REQUIRE_NO_THROW(ArrayDim3 a1(s)); // check size constructor
  ArrayDim3 a1(s);
  BOOST_CHECK_EQUAL(a1.volume(), v);  // check that a has correct va
  BOOST_CHECK_EQUAL(a1.size(), s);
  BOOST_CHECK_EQUAL(a1.weight(), w);

  BOOST_REQUIRE_NO_THROW(ArrayDim3 a2(a));
  ArrayDim3 a2(a);
  BOOST_CHECK_EQUAL(a2.volume(), a.volume());
  BOOST_CHECK_EQUAL(a2.size(), a.size());
  BOOST_CHECK_EQUAL(a2.weight(), a.weight());
}

BOOST_AUTO_TEST_CASE( ordinal )
{
  Range<std::size_t, 3> b(s);
  ArrayDim3::ordinal_type o = 0;
  for(Range<std::size_t, 3>::const_iterator it = b.begin(); it != b.end(); ++it, ++o)
    BOOST_CHECK_EQUAL(a.ordinal( *it ), o); // check ordinal calculation and order

#ifdef TA_EXCEPTION_ERROR
  index_type p(2,3,4);
  BOOST_CHECK_THROW(a.ordinal(p), std::out_of_range); // check for throw with
                                                  // an out of range element.
#endif
}

BOOST_AUTO_TEST_CASE( ordinal_fortran )
{
  typedef detail::ArrayDim<3, LevelTag<0>, CoordinateSystem<3, detail::increasing_dimension_order> > FArrayDim3;
  typedef Range<std::size_t, 3, LevelTag<0>, CoordinateSystem<3, detail::increasing_dimension_order> > range_type;

  FArrayDim3 a1(s);
  range_type r(s);
  FArrayDim3::ordinal_type o = 0;
  for(range_type::const_iterator it = r.begin(); it != r.end(); ++it, ++o)
    BOOST_CHECK_EQUAL(a1.ordinal( *it ), o); // check ordinal calculation and
                                             // order for fortran ordering.
}

BOOST_AUTO_TEST_CASE( resize )
{
  ArrayDim3 a1;
  a1.resize(s);
  BOOST_CHECK_EQUAL(a1.volume(), v); // check for correct resizing.
  BOOST_CHECK_EQUAL(a1.size(), s);
  BOOST_CHECK_EQUAL(a1.weight(), w);
}

BOOST_AUTO_TEST_SUITE_END()

struct DenseArrayStorageFixture : public ArrayDimFixture {
  typedef DenseArrayStorage<int, 3> DAS3;
  DenseArrayStorageFixture() : ArrayDimFixture(), da(s, 1)  {

  }

  ~DenseArrayStorageFixture() {

  }

  DAS3 da;
};

BOOST_FIXTURE_TEST_SUITE( dense_array_storage_suite, DenseArrayStorageFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(DAS3 a0); // check default constructor
  DAS3 a0;
  BOOST_CHECK_EQUAL(a0.volume(), 0); // check for zero size.
  BOOST_CHECK_THROW(a0.at(0), std::out_of_range); // check for data access error.

  BOOST_REQUIRE_NO_THROW(DAS3 a1(s, 1)); // check size constructor w/ initial value.
  DAS3 a1(s, 1);
  BOOST_CHECK_EQUAL(a1.volume(), v); // check for expected size.
  for(DAS3::ordinal_type i = 0; i < v; ++i)
    BOOST_CHECK_EQUAL(a1.at(i), 1); // check for expected values.

  boost::array<int, 24> val = {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}};
  BOOST_REQUIRE_NO_THROW(DAS3 a2(s, val.begin(), val.end())); // check size constructor w/ initialization list.
  DAS3 a2(s, val.begin(), val.end());
  BOOST_CHECK_EQUAL(a2.volume(), v); // check for expected size.
  int v2 = 0;
  for(DAS3::ordinal_type i = 0; i < v; ++i, ++v2)
    BOOST_CHECK_EQUAL(a2.at(i), v2); // check for expected values.

  BOOST_REQUIRE_NO_THROW(DAS3 a3(da));
  DAS3 a3(da);
  BOOST_CHECK_EQUAL(a3.volume(), v);
  for(DAS3::ordinal_type i = 0; i < v; ++i)
    BOOST_CHECK_EQUAL(a3.at(i), 1); // check for expected values.
}

BOOST_AUTO_TEST_CASE( accessor )
{
  for(DAS3::ordinal_type i = 0; i < v; ++i)
    BOOST_CHECK_EQUAL(da.at(i), 1);        // check ordinal access
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(da.at(v), std::out_of_range);// check for out of range error
  BOOST_CHECK_THROW(da.at(v), std::out_of_range);
#endif

  Range<std::size_t, 3> b(s);
  for(Range<std::size_t, 3>::const_iterator it = b.begin(); it != b.end(); ++it)
    BOOST_CHECK_EQUAL(da.at(* it), 1);        // check index access

  DAS3 a1(s, 1);
  BOOST_CHECK_EQUAL((a1.at(1) = 2), 2); // check for write access with at
  BOOST_CHECK_EQUAL(a1.at(1), 2);
  DAS3::index_type p1(0,0,2);
  BOOST_CHECK_EQUAL((a1.at(p1) = 2), 2);
  BOOST_CHECK_EQUAL(a1.at(p1), 2);

  DAS3::index_type p2(s);
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(da.at(p2), std::out_of_range);// check for out of range error
#endif

#ifdef NDEBUG // operator() calls at() when debugging so we don't need to run this test in that case.
  for(DAS3::ordinal_type i = 0; i < v; ++i)
    BOOST_CHECK_EQUAL(da.[i], 1);        // check ordinal access
  for(Range<std::size_t, 3>::const_iterator it = b.begin(); it != b.end(); ++it)
    BOOST_CHECK_EQUAL(da.[* it], 1);     // check index access

  DAS3 a2(s, 1);
  BOOST_CHECK_EQUAL((a2[1] = 2), 2); // check for write access with at
  BOOST_CHECK_EQUAL(a2[1], 2);
  DAS3::index_type p3(0,0,2);
  BOOST_CHECK_EQUAL((a2[p3] = 2), 2);
  BOOST_CHECK_EQUAL(a2[p3], 2);
#endif
}

BOOST_AUTO_TEST_CASE( iteration )
{
  boost::array<int, 24> val = {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}};
  DAS3 a1(s, val.begin(), val.end());
  DAS3::iterator it = a1.begin();
  BOOST_CHECK(iteration_test(a1, val.begin(), val.end())); // Check interator basic functionality
  BOOST_CHECK(const_iteration_test(a1, val.begin(), val.end())); // Check const iterator basic functionality
  BOOST_CHECK_EQUAL(a1.end() - a1.begin(), v); // check iterator difference operator
  BOOST_CHECK_EQUAL(*it, 0); // check dereference
  BOOST_CHECK_EQUAL(* (it + 2), 2); // check random access
  BOOST_CHECK_EQUAL((*it = 5), 5 ); // check iterator write access.
  BOOST_CHECK_EQUAL(*it, 5);
}

BOOST_AUTO_TEST_CASE( assignment )
{
  DAS3 a1;
  BOOST_CHECK_NO_THROW(a1 = da);
  BOOST_CHECK_EQUAL_COLLECTIONS(a1.begin(), a1.end(), da.begin(), da.end());
}

BOOST_AUTO_TEST_CASE( resize )
{
  DAS3::size_array s1 = {{3,2,2}};
  //                           {{0,  1,  2,  3, 10, 11, 12, 13, 20, 21, 22, 23,100,101,102,103,110,111,112,113,120,121,122,123}}
  boost::array<int, 24> val =  {{2,  2,  1,  1,  2,  2,  1,  1,  1,  1,  1,  1,  2,  2,  1,  1,  2,  2,  1,  1,  1,  1,  1,  1}};
  DAS3 a1(s1, 2);
  a1.resize(s,1);
  BOOST_CHECK_EQUAL(a1.size(), s);  // check that new dimensions are calculations.
  BOOST_CHECK_EQUAL(a1.volume(), v);
  BOOST_CHECK_EQUAL(a1.weight(), w);
  BOOST_CHECK_EQUAL_COLLECTIONS(a1.begin(), a1.end(), val.begin(), val.end()); // check that values from original are maintained.

}

BOOST_AUTO_TEST_CASE( permutation )
{
  Permutation<3> p(1,2,0);
  boost::array<int, 24> val =  {{0,  1,  2,  3, 10, 11, 12, 13, 20, 21, 22, 23,100,101,102,103,110,111,112,113,120,121,122,123}};
  //         destination       {{0,100,200,300,  1,101,201,301,  2,102,202,302, 10,110,210,310, 11,111,211,311, 12,112,212,312}}
  //         permuted index    {{0,  1,  2, 10, 11, 12,100,101,102,110,111,112,200,201,202,210,211,212,300,301,302,310,311,312}}
  boost::array<int, 24> pval = {{0, 10, 20,100,110,120,  1, 11, 21,101,111,121,  2, 12, 22,102,112,122,  3, 13, 23,103,113,123}};

  DAS3 a1(s, val.begin(), val.end());
  DAS3 a3(a1);
  DAS3 a2 = p ^ a1;
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.begin(), a2.end(), pval.begin(), pval.end()); // check permutation

  a3 ^= p;
  BOOST_CHECK_EQUAL_COLLECTIONS(a3.begin(), a3.end(), pval.begin(), pval.end()); // check in place permutation
}

BOOST_AUTO_TEST_SUITE_END()
