#include "variable_list.h"
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

using namespace TiledArray;
using TiledArray::detail::VariableList;

struct VariableListFixture {
  VariableListFixture() : v("a,b,c,d") { }

  VariableList v;
};

BOOST_FIXTURE_TEST_SUITE( variable_list_suite , VariableListFixture )

BOOST_AUTO_TEST_CASE( accessors )
{
  BOOST_CHECK_EQUAL(v.count(), 4); // check for variable count
  BOOST_CHECK_EQUAL(v.get(0), "a"); // check 1st variable access
  BOOST_CHECK_EQUAL(v.get(3), "d"); // check last variable access
  BOOST_CHECK_THROW(v.get(4), std::out_of_range); // check for out of range throw.
}

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(VariableList v0); // Check default constructor
  VariableList v0;
  BOOST_CHECK_EQUAL(v0.count(), 0);

  BOOST_REQUIRE_NO_THROW(VariableList v1("a,b,c,d")); // check string constructor
  VariableList v1("a,b,c,d");
  BOOST_CHECK_EQUAL(v1.count(), 4);
  BOOST_CHECK_EQUAL(v1.get(0), "a");  // check for corret data
  BOOST_CHECK_EQUAL(v1.get(1), "b");
  BOOST_CHECK_EQUAL(v1.get(2), "c");
  BOOST_CHECK_EQUAL(v1.get(3), "d");

  BOOST_REQUIRE_NO_THROW(VariableList v2(v)); // check string constructor
  VariableList v2(v);
  BOOST_CHECK_EQUAL(v2.count(), 4);
  BOOST_CHECK_EQUAL(v2.get(0), "a");  // check for corret data
  BOOST_CHECK_EQUAL(v2.get(1), "b");
  BOOST_CHECK_EQUAL(v2.get(2), "c");
  BOOST_CHECK_EQUAL(v2.get(3), "d");

  BOOST_CHECK_THROW(VariableList v3(",a,b,c"), std::runtime_error); // check invalid input
  BOOST_CHECK_THROW(VariableList v4("a,,b,c"), std::runtime_error);
  BOOST_CHECK_THROW(VariableList v5(" ,a,b"), std::runtime_error);
  BOOST_CHECK_THROW(VariableList v6("a,  b,   , c"), std::runtime_error);
  BOOST_CHECK_THROW(VariableList v8("a,b,a,c"), std::runtime_error);

  VariableList v7(" a , b, c, d , e e ,f f, g10,h, i "); // check input with various spacings.
  BOOST_CHECK_EQUAL(v7.get(0), "a");
  BOOST_CHECK_EQUAL(v7.get(1), "b");
  BOOST_CHECK_EQUAL(v7.get(2), "c");
  BOOST_CHECK_EQUAL(v7.get(3), "d");
  BOOST_CHECK_EQUAL(v7.get(4), "ee");
  BOOST_CHECK_EQUAL(v7.get(5), "ff");
  BOOST_CHECK_EQUAL(v7.get(6), "g10");
  BOOST_CHECK_EQUAL(v7.get(7), "h");
  BOOST_CHECK_EQUAL(v7.get(8), "i");

}

BOOST_AUTO_TEST_CASE( iterator )
{
  boost::array<std::string, 4> a1 = {{"a", "b", "c", "d"}};

  BOOST_CHECK_EQUAL_COLLECTIONS(v.begin(), v.end(), a1.begin(), a1.end()); // check that all values are equal
  boost::array<std::string, 4>::const_iterator it_a = a1.begin();
  for(VariableList::const_iterator it = v.begin(); it != v.end(); ++it, ++it_a) // check basic iterator functionality.
    BOOST_CHECK_EQUAL(*it, *it_a);

}

BOOST_AUTO_TEST_CASE( assignment )
{
  VariableList v1;
  v1 = v;
  BOOST_CHECK_EQUAL_COLLECTIONS(v1.begin(), v1.end(), v.begin(), v.end());

  VariableList v2;
  v2 = "a,b,c,d";
  BOOST_CHECK_EQUAL_COLLECTIONS(v2.begin(), v2.end(), v.begin(), v.end());
}

BOOST_AUTO_TEST_SUITE_END()
