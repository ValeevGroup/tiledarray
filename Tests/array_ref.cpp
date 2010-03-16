#include "TiledArray/array_ref.h"
#include <boost/array.hpp>
#include <vector>
#include "unit_test_config.h"

using namespace TiledArray;
using TiledArray::detail::ArrayRef;

struct ArrayRefFixture {

  // Fixture setup
  ArrayRefFixture() : a(), ca(a), r(a), cr(ca) {
    int i = 0;
    for(boost::array<int, 10>::iterator it = a.begin(); it != a.end(); ++it)
      *it = ++i;
  }

  // Fixture tear down
  ~ArrayRefFixture() { }

  boost::array<int, 10> a;
  const boost::array<int, 10>& ca;
  ArrayRef<int> r;
  ArrayRef<const int> cr;
};

BOOST_FIXTURE_TEST_SUITE( array_ref_suite , ArrayRefFixture )

BOOST_AUTO_TEST_CASE( constructors )
{
  BOOST_REQUIRE_NO_THROW(ArrayRef<int> r0); // Check default constructor
  ArrayRef<int> r0;
  BOOST_CHECK_EQUAL(r0.size(), 0);

  BOOST_REQUIRE_NO_THROW(ArrayRef<int> rc(r)); // Check copy constructor
  ArrayRef<int> rc(r);
  BOOST_CHECK_EQUAL_COLLECTIONS(rc.begin(), rc.end(), r.begin(), r.end());

  BOOST_REQUIRE_NO_THROW(ArrayRef<int> r1(a)); // Check boost array constructor
  ArrayRef<int> r1(a);
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.begin(), r1.end(), a.begin(), a.end());

  BOOST_REQUIRE_NO_THROW(ArrayRef<const int> r2(ca)); // Check const boost array constructor
  ArrayRef<const int> r2(ca);
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.begin(), r2.end(), ca.begin(), ca.end());

  BOOST_REQUIRE_NO_THROW(ArrayRef<int> r3(std::make_pair(a.data(), a.data() + 10))); // Check pointer pair constructor
  ArrayRef<int> r3(std::make_pair(a.data(), a.data() + 10));
  BOOST_CHECK_EQUAL_COLLECTIONS(r3.begin(), r3.end(), a.begin(), a.end());

  BOOST_REQUIRE_NO_THROW(ArrayRef<int> r4(a.data(), a.data() + 10)); // Check pointer constructor
  ArrayRef<int> r4(a.data(), a.data() + 10);
  BOOST_CHECK_EQUAL_COLLECTIONS(r4.begin(), r4.end(), a.begin(), a.end());

  BOOST_REQUIRE_NO_THROW(ArrayRef<const int> r5(ca.data(), ca.data() + 10)); // Check pointer constructor with const pointers
  ArrayRef<const int> r5(ca.data(), ca.data() + 10);
  BOOST_CHECK_EQUAL_COLLECTIONS(r5.begin(), r5.end(), a.begin(), a.end());
}

BOOST_AUTO_TEST_CASE( assignment )
{
  ArrayRef<int> rc; // Check copy assignment
  BOOST_CHECK_EQUAL(rc.size(), 0);
  rc = r;
  BOOST_CHECK_EQUAL_COLLECTIONS(rc.begin(), rc.end(), r.begin(), r.end());

  ArrayRef<int> r1; // Check copy assignment
  BOOST_CHECK_EQUAL(r1.size(), 0);
  r1 = a;
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.begin(), r1.end(), a.begin(), a.end());

  ArrayRef<int> r2; // Check boost array assignment
  BOOST_CHECK_EQUAL(r2.size(), 0);
  r2 = a;
  BOOST_CHECK_EQUAL_COLLECTIONS(r2.begin(), r2.end(), a.begin(), a.end());

  ArrayRef<const int> r3; // Check boost array assignment
  BOOST_CHECK_EQUAL(r3.size(), 0);
  r3 = ca;
  BOOST_CHECK_EQUAL_COLLECTIONS(r3.begin(), r3.end(), ca.begin(), ca.end());

  ArrayRef<int> r4; // Check boost array assignment
  BOOST_CHECK_EQUAL(r4.size(), 0);
  r4 = std::make_pair(a.data(), a.data() + 10);
  BOOST_CHECK_EQUAL_COLLECTIONS(r4.begin(), r4.end(), a.begin(), a.end());
}

BOOST_AUTO_TEST_CASE( type_conversion )
{
  boost::array<int, 10> a1(r); // check type conversion
  BOOST_CHECK_EQUAL_COLLECTIONS(a1.begin(), a1.end(), r.begin(), r.end());
}

BOOST_AUTO_TEST_CASE( iterators )
{
  BOOST_CHECK_EQUAL(r.begin(), a.begin()); // check iterators
  BOOST_CHECK_EQUAL(r.end(), a.end());
  BOOST_CHECK(r.begin() == a.begin()); // check that iterators are comparable with boost iterators
  BOOST_CHECK(r.end() == a.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(r.begin(), r.end(), a.begin(), a.end());

  BOOST_CHECK_EQUAL(&(* r.rbegin()), &(* a.rbegin())); // check iterators
  BOOST_CHECK_EQUAL(&(* r.rend()), &(* a.rend()));
  BOOST_CHECK(r.rbegin() == a.rbegin()); // check that iterators are comparable
  BOOST_CHECK(r.rend() == a.rend());     // with boost reverse iterators.
  BOOST_CHECK_EQUAL_COLLECTIONS(r.rbegin(), r.rend(), a.rbegin(), a.rend());
}

BOOST_AUTO_TEST_CASE( size )
{
  BOOST_CHECK_EQUAL(r.size(), a.size()); // check array size
}

BOOST_AUTO_TEST_CASE( empty )
{
  ArrayRef<int> r1;

  BOOST_CHECK(r1.empty());  // Check empty and non-empty arrays
  BOOST_CHECK(! r.empty());
}

BOOST_AUTO_TEST_CASE( max_size )
{
  BOOST_CHECK_EQUAL(r.max_size(), 10); // check max size
}

BOOST_AUTO_TEST_CASE( element_accessor )
{
  for(std::size_t i = 0; i < 10; ++i) { // Check element access
    BOOST_CHECK_EQUAL(r[i], a[i]);
    BOOST_CHECK_EQUAL(cr[i], ca[i]);
    BOOST_CHECK_EQUAL(r.at(i), a.at(i));
    BOOST_CHECK_EQUAL(cr.at(i), ca.at(i));
  }

  BOOST_CHECK_THROW(r.at(10), std::out_of_range); // check range checking
  BOOST_CHECK_NO_THROW(r[10]); // check that no range checking is done.

  BOOST_CHECK_EQUAL(r.front(), a.front());
  BOOST_CHECK_EQUAL(r.back(), a.back());
  BOOST_CHECK_EQUAL(cr.front(), ca.front());
  BOOST_CHECK_EQUAL(cr.back(), ca.back());
}

BOOST_AUTO_TEST_CASE( pointer_accessor )
{
  BOOST_CHECK_EQUAL(r.data(), a.data());
  BOOST_CHECK_EQUAL(cr.data(), a.data());

  BOOST_CHECK_EQUAL(r.c_array(), a.c_array());
}

BOOST_AUTO_TEST_CASE( swap )
{
  ArrayRef<int> r1;
  BOOST_CHECK_EQUAL(r1.size(), 0);
  BOOST_CHECK_EQUAL(r.size(), 10);
  BOOST_CHECK_EQUAL_COLLECTIONS(r.begin(), r.end(), a.begin(), a.end());

  r.swap(r1);

  BOOST_CHECK_EQUAL(r.size(), 0);
  BOOST_CHECK_EQUAL(r1.size(), 10);
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.begin(), r1.end(), a.begin(), a.end());

  detail::swap(r, r1);

  BOOST_CHECK_EQUAL(r1.size(), 0);
  BOOST_CHECK_EQUAL(r.size(), 10);
  BOOST_CHECK_EQUAL_COLLECTIONS(r.begin(), r.end(), a.begin(), a.end());
}

BOOST_AUTO_TEST_CASE( assign )
{
  int i = 0;
  for(ArrayRef<int>::const_iterator it = r.begin(); it != r.end(); ++it)
    BOOST_CHECK_EQUAL(*it, ++i);
  r.assign(11);
  for(ArrayRef<int>::const_iterator it = r.begin(); it != r.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 11);
}

BOOST_AUTO_TEST_SUITE_END()
