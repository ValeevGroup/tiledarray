//#include "TiledArray/utility.h"
#include "TiledArray/dense_array.h"
#include <vector>
#include <numeric>
#include <algorithm>
#include <iterator>
#include "unit_test_config.h"
#include "array_fixtures.h"

using namespace TiledArray;


DenseArrayFixture::DenseArrayFixture() : prange(&r, &no_delete), da(prange, 1)
{ }

BOOST_FIXTURE_TEST_SUITE( dense_storage_suite, DenseArrayFixture )

BOOST_AUTO_TEST_CASE( at_accessor )
{
  // check ordinal access
  for(ordinal_index i = 0; i < volume; ++i)
    BOOST_CHECK_EQUAL(da.at(i), 1);

  // check index access
  for(RangeN::const_iterator it = r.begin(); it != r.end(); ++it)
    BOOST_CHECK_EQUAL(da.at(* it), 1);

  // check for out of range error
  BOOST_CHECK_THROW(da.at(volume), std::out_of_range);
  BOOST_CHECK_THROW(da.at(p6), std::out_of_range);



  // check for write access with at() function with ordinal index
  ordinal_index i = DenseArrayN::coordinate_system::calc_ordinal(p1, r.weight());
  BOOST_CHECK_EQUAL((da.at(i) = 2), 2);
  BOOST_CHECK_EQUAL(da.at(i), 2);
  BOOST_CHECK_EQUAL(da.at(p1), 2);

  // check for write access with at() function with index
  BOOST_CHECK_EQUAL((da.at(p1) = 3), 3);
  BOOST_CHECK_EQUAL(da.at(i), 3);
  BOOST_CHECK_EQUAL(da.at(p1), 3);
}

BOOST_AUTO_TEST_CASE( array_operator_accessor )
{
  // check ordinal access
  for(ordinal_index i = 0; i < volume; ++i)
    BOOST_CHECK_EQUAL(da[i], 1);

  // check index access
  for(RangeN::const_iterator it = r.begin(); it != r.end(); ++it)
    BOOST_CHECK_EQUAL(da[* it], 1);

  // check for out of range error
#ifdef NDEBUG
  BOOST_CHECK_NO_THROW(da[volume]);
  BOOST_CHECK_NO_THROW(da[p6]);
#else
  BOOST_CHECK_THROW(da[volume], std::out_of_range);
  BOOST_CHECK_THROW(da[p6], std::out_of_range);
#endif


  // check for write access with at() function with ordinal index
  ordinal_index i = DenseArrayN::coordinate_system::calc_ordinal(p1, r.weight());
  BOOST_CHECK_EQUAL((da[i] = 2), 2);
  BOOST_CHECK_EQUAL(da[i], 2);
  BOOST_CHECK_EQUAL(da[p1], 2);

  // check for write access with at() function with index
  BOOST_CHECK_EQUAL((da[p1] = 3), 3);
  BOOST_CHECK_EQUAL(da[i], 3);
  BOOST_CHECK_EQUAL(da[p1], 3);
}

BOOST_AUTO_TEST_CASE( constructor )
{
  // Check default value construction
  DenseArrayN a0(prange);
  for(DenseArrayN::ordinal_index i = 0; i < volume; ++i)
    BOOST_CHECK_EQUAL(a0.at(i), 0); // check for expected values.

  // Check value constructor
  BOOST_REQUIRE_NO_THROW(DenseArrayN a1(prange, 1));
  DenseArrayN a1(prange, 1);
  for(DenseArrayN::ordinal_index i = 0; i < volume; ++i)
    BOOST_CHECK_EQUAL(a1.at(i), 1); // check for expected values.

  std::vector<int> val(volume, 0);
  for(unsigned int i = 0; i < volume; ++i)
    val[i] = i;

  // Check constructor w/ initialization list.
  BOOST_REQUIRE_NO_THROW(DenseArrayN a2(prange, val.begin(), val.end()));
  DenseArrayN a2(prange, val.begin(), val.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(val.begin(), val.end(), a2.begin(), a2.end());

  // Check error condition for a list with a size not equal to volume.
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(DenseArrayN a3(prange, val.begin(), val.end() - 3), std::runtime_error); // check for bad input
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( iteration )
{
  DenseArrayN::iterator it = da.begin();
  // check dereference
  BOOST_CHECK_EQUAL(*it, 1);
  BOOST_CHECK_EQUAL(* (it + 2), 2); // check random access

  // check assignement
  BOOST_CHECK_EQUAL( (*it = 2) , 2);
  BOOST_CHECK_EQUAL(*it, 2);

  // check iterator difference operator
  BOOST_CHECK_EQUAL(da.end() - da.begin(), std::ptrdiff_t(volume));

  std::vector<int> val(volume, 0);
  for(unsigned int i = 0; i < volume; ++i)
    val[i] = i;

  // Check that it copies correctly
  std::copy(val.begin(), val.end(), da.begin());
  BOOST_CHECK_EQUAL_COLLECTIONS(da.begin(), da.end(), val.begin(), val.end());
}

BOOST_AUTO_TEST_CASE( range_access )
{
  BOOST_CHECK_EQUAL(da.range(), r);
}

BOOST_AUTO_TEST_SUITE_END()



