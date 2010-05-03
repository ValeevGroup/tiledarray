#include "TiledArray/utility.h"
#include "TiledArray/dense_array.h"
#include "TiledArray/coordinates.h"
#include "TiledArray/permutation.h"
#include <numeric>
#include <algorithm>
#include <iterator>
#include "unit_test_config.h"
#include "array_fixtures.h"

using namespace TiledArray;



DenseArrayFixture::DenseArrayFixture() : ArrayDimFixture(), da3(s, 1) {

  PermutationN::Array pp = {{0, 2, 1, 4, 3}};
  p0 = pp;
  DenseArrayN::size_array n = {{3, 5, 7, 11, 13}};
  DenseArrayN::size_array n_p0 = p0 ^ n;
  daN.resize(n);
  daN_p0.resize(n_p0);

  typedef Range<DenseArrayN::ordinal_type, ndim, DenseArrayN::tag_type, DenseArrayN::coordinate_system> RangeN;
  typedef RangeN::const_iterator index_iter;
  typedef DenseArrayN::iterator iter;
  RangeN range(n);
  RangeN range_p0(n_p0);
  iter v = daN.begin();
  for(index_iter i=range.begin(); i!=range.end(); ++i, ++v) {
    *v = daN.ordinal(*i);
    daN_p0[p0 ^ *i] = *v;
  }

  //std::copy(daN.begin(), daN.end(), std::ostream_iterator<int>(std::cout, "\n"));
  //std::copy(daN_p0.begin(), daN_p0.end(), std::ostream_iterator<int>(std::cout, "\n"));
}


BOOST_FIXTURE_TEST_SUITE( dense_storage_suite, DenseArrayFixture )

BOOST_AUTO_TEST_CASE( array_dims )
{
  BOOST_CHECK_EQUAL(da3.size(), s);
  BOOST_CHECK_EQUAL(da3.weight(), w);
  BOOST_CHECK_EQUAL(da3.volume(), v);
}

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(DenseArray3 a0); // check default constructor
  DenseArray3 a0;
  BOOST_CHECK_EQUAL(a0.volume(), 0ul); // check for zero size.
  BOOST_CHECK_THROW(a0.at(0), std::runtime_error); // check for data access error.

  BOOST_REQUIRE_NO_THROW(DenseArray3 a1(s, 1)); // check size constructor w/ initial value.
  DenseArray3 a1(s, 1);
  BOOST_CHECK_EQUAL(a1.volume(), v); // check for expected size.
  for(DenseArray3::ordinal_type i = 0; i < v; ++i)
    BOOST_CHECK_EQUAL(a1.at(i), 1); // check for expected values.

  boost::array<int, 24> val = {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}};
  BOOST_REQUIRE_NO_THROW(DenseArray3 a2(s, val.begin(), val.end())); // check size constructor w/ initialization list.
  DenseArray3 a2(s, val.begin(), val.end());
  BOOST_CHECK_EQUAL(a2.volume(), v); // check for expected size.
  int v2 = 0;
  for(DenseArray3::ordinal_type i = 0; i < v; ++i, ++v2)
    BOOST_CHECK_EQUAL(a2.at(i), v2); // check for expected values.

  BOOST_REQUIRE_NO_THROW(DenseArray3 a3(da3));
  DenseArray3 a3(da3);
  BOOST_CHECK_EQUAL(a3.volume(), v);
  for(DenseArray3::ordinal_type i = 0; i < v; ++i)
    BOOST_CHECK_EQUAL(a3.at(i), 1); // check for expected values.

  BOOST_REQUIRE_NO_THROW(DenseArray3 a4(s, val.begin(), val.end() - 1)); // check size constructor w/ short initialization list.
  DenseArray3 a4(s, val.begin(), val.end() - 3);
  int v4 = 0;
  for(DenseArray3::ordinal_type i = 0; i < v - 3; ++i, ++v4)
    BOOST_CHECK_EQUAL(a4.at(i), v4);
  for(DenseArray3::ordinal_type i = v - 3; i < v; ++i)
    BOOST_CHECK_EQUAL(a4.at(i), int());
}

BOOST_AUTO_TEST_CASE( accessor )
{
  for(DenseArray3::ordinal_type i = 0; i < v; ++i)
    BOOST_CHECK_EQUAL(da3.at(i), 1);        // check ordinal access
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(da3.at(v), std::out_of_range);// check for out of range error
  BOOST_CHECK_THROW(da3.at(v), std::out_of_range);
#endif

  Range<std::size_t, 3> b(s);
  for(Range<std::size_t, 3>::const_iterator it = b.begin(); it != b.end(); ++it)
    BOOST_CHECK_EQUAL(da3.at(* it), 1);        // check index access

  DenseArray3 a1(s, 1);
  BOOST_CHECK_EQUAL((a1.at(1) = 2), 2); // check for write access with at
  BOOST_CHECK_EQUAL(a1.at(1), 2);
  DenseArray3::index_type p1(0,0,2);
  BOOST_CHECK_EQUAL((a1.at(p1) = 2), 2);
  BOOST_CHECK_EQUAL(a1.at(p1), 2);

  DenseArray3::index_type p2(s);
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(da3.at(p2), std::out_of_range);// check for out of range error
#endif

#ifdef NDEBUG // operator() calls at() when debugging so we don't need to run this test in that case.
  for(DenseArray3::ordinal_type i = 0; i < v; ++i)
    BOOST_CHECK_EQUAL(da3.[i], 1);        // check ordinal access
  for(Range<std::size_t, 3>::const_iterator it = b.begin(); it != b.end(); ++it)
    BOOST_CHECK_EQUAL(da3.[* it], 1);     // check index access

  DenseArray3 a2(s, 1);
  BOOST_CHECK_EQUAL((a2[1] = 2), 2); // check for write access with at
  BOOST_CHECK_EQUAL(a2[1], 2);
  DenseArray3::index_type p3(0,0,2);
  BOOST_CHECK_EQUAL((a2[p3] = 2), 2);
  BOOST_CHECK_EQUAL(a2[p3], 2);
#endif
}

BOOST_AUTO_TEST_CASE( iteration )
{
  boost::array<int, 24> val = {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}};
  DenseArray3 a1(s, val.begin(), val.end());
  DenseArray3::iterator it = a1.begin();
  BOOST_CHECK(iteration_test(a1, val.begin(), val.end())); // Check interator basic functionality
  BOOST_CHECK(const_iteration_test(a1, val.begin(), val.end())); // Check const iterator basic functionality
  BOOST_CHECK_EQUAL(a1.end() - a1.begin(), (std::ptrdiff_t)v); // check iterator difference operator
  BOOST_CHECK_EQUAL(*it, 0); // check dereference
  BOOST_CHECK_EQUAL(* (it + 2), 2); // check random access
  BOOST_CHECK_EQUAL((*it = 5), 5 ); // check iterator write access.
  BOOST_CHECK_EQUAL(*it, 5);
}

BOOST_AUTO_TEST_CASE( assignment )
{
  DenseArray3 a1;
  BOOST_CHECK_NO_THROW(a1 = da3);
  BOOST_CHECK_EQUAL_COLLECTIONS(a1.begin(), a1.end(), da3.begin(), da3.end());
}

BOOST_AUTO_TEST_CASE( resize )
{
  DenseArray3::size_array s1 = {{3,2,2}};
  //                           {{0,  1,  2,  3, 10, 11, 12, 13, 20, 21, 22, 23,100,101,102,103,110,111,112,113,120,121,122,123}}
  boost::array<int, 24> val =  {{2,  2,  1,  1,  2,  2,  1,  1,  1,  1,  1,  1,  2,  2,  1,  1,  2,  2,  1,  1,  1,  1,  1,  1}};
  DenseArray3 a1(s1, 2);
  a1.resize(s,1);
  BOOST_CHECK_EQUAL(a1.size(), s);  // check that new dimensions are calculations.
  BOOST_CHECK_EQUAL(a1.volume(), v);
  BOOST_CHECK_EQUAL(a1.weight(), w);
  BOOST_CHECK_EQUAL_COLLECTIONS(a1.begin(), a1.end(), val.begin(), val.end()); // check that values from original are maintained.

}

BOOST_AUTO_TEST_CASE( permutation )
{
  DenseArrayN a1(daN.size(), daN.begin(), daN.end());
  DenseArrayN a3(a1);
  BOOST_CHECK_EQUAL_COLLECTIONS(a1.begin(), a1.end(), daN.begin(), daN.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(a3.begin(), a3.end(), daN.begin(), daN.end());
  DenseArrayN a2 = p0 ^ a1;
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.begin(), a2.end(), daN_p0.begin(), daN_p0.end()); // check permutation

  a3 ^= p0;
  BOOST_CHECK_EQUAL_COLLECTIONS(a3.begin(), a3.end(), daN_p0.begin(), daN_p0.end()); // check in place permutation
}

BOOST_AUTO_TEST_SUITE_END()



