#include "TiledArray/annotated_array.h"
#include "TiledArray/tile.h"
#include "TiledArray/array.h"
#include "unit_test_config.h"
#include "range_fixture.h"
#include "math_fixture.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

struct AnnotatedArrayFixture : public MathFixture {

  AnnotatedArrayFixture() : a(f1, VariableList(make_var_list())) { }

  array_annotation a;
}; // struct AnnotatedArrayFixture



BOOST_FIXTURE_TEST_SUITE( annotated_array_suite , AnnotatedArrayFixture )

BOOST_AUTO_TEST_CASE( range_accessor )
{
  BOOST_CHECK_EQUAL(a.range(), r);
}

BOOST_AUTO_TEST_CASE( vars_accessor )
{
  VariableList v(make_var_list());
  BOOST_CHECK_EQUAL(a.vars(), v);
}

BOOST_AUTO_TEST_CASE( tile_data )
{
  BOOST_CHECK_EQUAL_COLLECTIONS(a.begin(), a.end(), f1.begin(), f1.end());
}

BOOST_AUTO_TEST_CASE( constructors )
{
  BOOST_REQUIRE_NO_THROW(array_annotation at1(f1, vars));
  array_annotation at1(f1, vars);
  BOOST_CHECK_EQUAL_COLLECTIONS(at1.begin(), at1.end(), f1.begin(), f1.end());
  BOOST_CHECK_EQUAL(at1.range(), r);
  BOOST_CHECK_EQUAL(at1.vars(), vars);

  BOOST_REQUIRE_NO_THROW(array_annotation at2(f1, vars));
  array_annotation at2(f1, vars);
  BOOST_CHECK_EQUAL_COLLECTIONS(at2.begin(), at2.end(), f1.begin(), f1.end());
  BOOST_CHECK_EQUAL(at2.range(), r);
  BOOST_CHECK_EQUAL(at2.vars(), vars);

  BOOST_REQUIRE_NO_THROW(array_annotation at3(a));
  array_annotation at3(a);
  BOOST_CHECK_EQUAL_COLLECTIONS(at3.begin(), at3.end(), f1.begin(), f1.end());
  BOOST_CHECK_EQUAL(at3.range(), r);
  BOOST_CHECK_EQUAL(at3.vars(), vars);
}

BOOST_AUTO_TEST_SUITE_END()
