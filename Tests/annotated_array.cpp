#include "TiledArray/annotated_array.h"
#include "TiledArray/tile.h"
#include "TiledArray/array.h"
#include "unit_test_config.h"
#include "range_fixture.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

struct AnnotatedArrayFixture {
  typedef Tile<int, GlobalFixture::element_coordinate_system> TileN;
  typedef TileN::range_type RangeN;
  typedef AnnotatedArray<TileN> AnnotatedTileN;
  typedef AnnotatedTileN::index index;

  static const VariableList vars;
  static const boost::shared_ptr<RangeN> r;
  static const TileN t;

  AnnotatedArrayFixture() : at(t, vars) { }

  static std::string make_var_list() {
    const char* temp = "a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p,q,r,s,t,u,v,w,x,y,z";
    std::string result(temp, 2 * GlobalFixture::element_coordinate_system::dim - 1 );
    return result;
  }

  AnnotatedTileN at;
}; // struct AnnotatedArrayFixture

const VariableList AnnotatedArrayFixture::vars(AnnotatedArrayFixture::make_var_list());
const boost::shared_ptr<AnnotatedArrayFixture::RangeN> AnnotatedArrayFixture::r =
    boost::make_shared<AnnotatedArrayFixture::RangeN>(
    fill_index<AnnotatedArrayFixture::index>(0),
    fill_index<AnnotatedArrayFixture::index>(5));
const AnnotatedArrayFixture::TileN AnnotatedArrayFixture::t(r, 1);

BOOST_FIXTURE_TEST_SUITE( annotated_array_suite , AnnotatedArrayFixture )

BOOST_AUTO_TEST_CASE( range_accessor )
{
  BOOST_CHECK_EQUAL(at.range(), *r);
}

BOOST_AUTO_TEST_CASE( iterators )
{
  BOOST_CHECK( at.begin() == t.end() );
  BOOST_CHECK( at.end() == t.end() );
}

BOOST_AUTO_TEST_CASE( const_iterators )
{
  const TileN& ct = t;
  const AnnotatedTileN cat = at;

  BOOST_CHECK( ct.begin() == cat.end() );
  BOOST_CHECK( ct.end() == cat.end() );
}

BOOST_AUTO_TEST_CASE( tile_data )
{
  BOOST_CHECK_EQUAL_COLLECTIONS(at.begin(), at.end(), t.begin(), t.end());
}

BOOST_AUTO_TEST_CASE( constructors )
{
  BOOST_REQUIRE_NO_THROW(AnnotatedTileN at1(t, vars));
  AnnotatedTileN at1(t, vars);
  BOOST_CHECK_EQUAL_COLLECTIONS(at1.begin(), at1.end(), t.begin(), t.end());
  BOOST_CHECK_EQUAL(at1.range(), *r);
  BOOST_CHECK_EQUAL(at1.vars(), vars);

  BOOST_REQUIRE_NO_THROW(AnnotatedTileN at2(t, vars));
  AnnotatedTileN at2(t, vars);
  BOOST_CHECK_EQUAL_COLLECTIONS(at2.begin(), at2.end(), t.begin(), t.end());
  BOOST_CHECK_EQUAL(at2.range(), *r);
  BOOST_CHECK_EQUAL(at2.vars(), vars);

  BOOST_REQUIRE_NO_THROW(AnnotatedTileN at3(at));
  AnnotatedTileN at3(at);
  BOOST_CHECK_EQUAL_COLLECTIONS(at3.begin(), at3.end(), t.begin(), t.end());
  BOOST_CHECK_EQUAL(at3.range(), *r);
  BOOST_CHECK_EQUAL(at3.vars(), vars);

#ifdef __GXX_EXPERIMENTAL_CXX0X__
  BOOST_REQUIRE_NO_THROW(AnnotatedTileN at4(std::move(at)));
  AnnotatedTileN at4(std::move(at));
  BOOST_CHECK_EQUAL_COLLECTIONS(at4.begin(), at4.end(), t.begin(), t.end());
  BOOST_CHECK_EQUAL(at4.range(), *r);
  BOOST_CHECK_EQUAL(at4.vars(), vars);
#endif // __GXX_EXPERIMENTAL_CXX0X__
}

BOOST_AUTO_TEST_SUITE_END()
