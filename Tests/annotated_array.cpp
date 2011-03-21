#include "TiledArray/annotated_array.h"
#include "TiledArray/tile.h"
#include "TiledArray/array.h"
#include "unit_test_config.h"
#include "range_fixture.h"
#include "array_fixtures.h"

using namespace TiledArray;
using namespace TiledArray::expressions;


AnnotatedArrayFixture::AnnotatedArrayFixture() : at(t, vars) { }

std::string AnnotatedArrayFixture::make_var_list() {
  const char* temp = "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z";
  std::string result(temp, 2 * GlobalFixture::element_coordinate_system::dim - 1 );
  return result;
}

const VariableList AnnotatedArrayFixture::vars(AnnotatedArrayFixture::make_var_list());
const AnnotatedArrayFixture::range_type AnnotatedArrayFixture::r(
    fill_index<AnnotatedArrayFixture::index>(0),
    fill_index<AnnotatedArrayFixture::index>(5));
const AnnotatedArrayFixture::array_type AnnotatedArrayFixture::t(r, 1);

BOOST_FIXTURE_TEST_SUITE( annotated_array_suite , AnnotatedArrayFixture )

BOOST_AUTO_TEST_CASE( range_accessor )
{
  BOOST_CHECK_EQUAL(at.range(), r);
}

BOOST_AUTO_TEST_CASE( iterators )
{
  BOOST_CHECK( at.begin() == t.begin() );
  BOOST_CHECK( at.end() == t.end() );
}

BOOST_AUTO_TEST_CASE( const_iterators )
{
  const array_type& ct = t;
  const fake_annotation cat = at;

  BOOST_CHECK( ct.begin() == cat.begin() );
  BOOST_CHECK( ct.end() == cat.end() );
}

BOOST_AUTO_TEST_CASE( tile_data )
{
  BOOST_CHECK_EQUAL_COLLECTIONS(at.begin(), at.end(), t.begin(), t.end());
}

BOOST_AUTO_TEST_CASE( constructors )
{
  BOOST_REQUIRE_NO_THROW(fake_annotation at1(t, vars));
  fake_annotation at1(t, vars);
  BOOST_CHECK_EQUAL_COLLECTIONS(at1.begin(), at1.end(), t.begin(), t.end());
  BOOST_CHECK_EQUAL(at1.range(), r);
  BOOST_CHECK_EQUAL(at1.vars(), vars);

  BOOST_REQUIRE_NO_THROW(fake_annotation at2(t, vars));
  fake_annotation at2(t, vars);
  BOOST_CHECK_EQUAL_COLLECTIONS(at2.begin(), at2.end(), t.begin(), t.end());
  BOOST_CHECK_EQUAL(at2.range(), r);
  BOOST_CHECK_EQUAL(at2.vars(), vars);

  BOOST_REQUIRE_NO_THROW(fake_annotation at3(at));
  fake_annotation at3(at);
  BOOST_CHECK_EQUAL_COLLECTIONS(at3.begin(), at3.end(), t.begin(), t.end());
  BOOST_CHECK_EQUAL(at3.range(), r);
  BOOST_CHECK_EQUAL(at3.vars(), vars);
}

BOOST_AUTO_TEST_SUITE_END()
