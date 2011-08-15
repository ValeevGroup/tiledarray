#include "TiledArray/expressions.h"
#include "TiledArray/tile.h"
#include "TiledArray/permutation.h"
#include "TiledArray/expressions.h"
#include <math.h>
#include <utility>
#include "unit_test_config.h"
#include "range_fixture.h"
#include <world/bufar.h>

using namespace TiledArray;

struct ExpressionFixture {
  typedef TiledArray::expressions::Tile<int, GlobalFixture::element_coordinate_system> TileN;
  typedef TileN::index index;
  typedef TileN::volume_type volume_type;
  typedef TileN::size_array size_array;
  typedef TileN::range_type RangeN;

  static const RangeN r;

  ExpressionFixture() : t(r, 1) {
  }

  ~ExpressionFixture() { }

  TileN t;
};

const ExpressionFixture::RangeN ExpressionFixture::r = ExpressionFixture::RangeN(index(0), index(5));

BOOST_FIXTURE_TEST_SUITE( expressions_suite , ExpressionFixture )

BOOST_AUTO_TEST_CASE( permutation )
{
  typedef TiledArray::CoordinateSystem<3, 0> cs3;
  Permutation<3> p(1,2,0);
  Range<cs3> r1(Range<cs3>::index(0,0,0), Range<cs3>::index(2,3,4));
  Range<cs3> r3(r1);
  std::array<double, 24> val =  {{0,  1,  2,  3, 10, 11, 12, 13, 20, 21, 22, 23,100,101,102,103,110,111,112,113,120,121,122,123}};
  //         destination       {{0,100,200,300,  1,101,201,301,  2,102,202,302, 10,110,210,310, 11,111,211,311, 12,112,212,312}}
  //         permuted index    {{0,  1,  2, 10, 11, 12,100,101,102,110,111,112,200,201,202,210,211,212,300,301,302,310,311,312}}
  std::array<double, 24> pval = {{0, 10, 20,100,110,120,  1, 11, 21,101,111,121,  2, 12, 22,102,112,122,  3, 13, 23,103,113,123}};
  expressions::Tile<int, cs3> t1(r1, val.begin());
  expressions::Tile<int, cs3> t2 = (p ^ t1);
  BOOST_CHECK_EQUAL(t2.range(), p ^ t1.range()); // check that the dimensions were correctly permuted.
  BOOST_CHECK_EQUAL_COLLECTIONS(t2.begin(), t2.end(), pval.begin(), pval.end()); // check that the values were correctly permuted.

  expressions::Tile<int, cs3> t3(r3, val.begin());
  t3 ^= p;
  BOOST_CHECK_EQUAL(t3.range(), p ^ t1.range()); // check that the dimensions were correctly permuted.
  BOOST_CHECK_EQUAL_COLLECTIONS(t3.begin(), t3.end(), pval.begin(), pval.end()); // check that the values were correctly permuted.
}

BOOST_AUTO_TEST_CASE( addition )
{
  const TileN t1(r, 1);
  const TileN t2(r, 2);

  // Check + operator
  t = t1 + t2;
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 3);

  t = t1 + TileN();
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 1);

  t.resize(RangeN());
  t = TileN() + t1;
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 1);

  t = TileN() + TileN();
  BOOST_CHECK_EQUAL(t.range().volume(), 0ul);
}

BOOST_AUTO_TEST_CASE( subtraction )
{
  const TileN t1(r, 1);
  const TileN t2(r, 2);

  // Check + operator
  t = t1 - t2;
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, -1);

  t = t1 - TileN();
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 1);

  t.resize(RangeN());
  t = TileN() - t1;
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, -1);

  t = TileN() + TileN();
  BOOST_CHECK_EQUAL(t.range().volume(), 0);
}

BOOST_AUTO_TEST_CASE( scalar_addition )
{
  // Check + operator
  t = t + 2;
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 3);

  t = 2 + t;
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 5);

  t = TileN() + 1;
  BOOST_CHECK_EQUAL(t.range().volume(), 0ul);

  t = 1 + TileN();
  BOOST_CHECK_EQUAL(t.range().volume(), 0ul);
}

BOOST_AUTO_TEST_CASE( scalar_subtraction )
{
  // Check + operator
  t = t - 2;
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, -1);

  t = 3 - t;
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 4);

  t = TileN() - 1;
  BOOST_CHECK_EQUAL(t.range().volume(), 0ul);

  t = 1 - TileN();
  BOOST_CHECK_EQUAL(t.range().volume(), 0ul);
}

BOOST_AUTO_TEST_CASE( scalar_multiplication )
{
  t = 2 * t;
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 2);

  t = TileN() * 2;
  BOOST_CHECK_EQUAL(t.range().volume(), 0ul);

  t = 2 * TileN();
  BOOST_CHECK_EQUAL(t.range().volume(), 0ul);
}

BOOST_AUTO_TEST_CASE( negation )
{

  // Check that += operator
  TileN tn = -t;
  for(TileN::const_iterator it = tn.begin(); it != tn.end(); ++it)
    BOOST_CHECK_EQUAL(*it, -1);

  tn = - TileN();
  BOOST_CHECK_EQUAL(tn.range().volume(), 0ul);
}

BOOST_AUTO_TEST_SUITE_END()

