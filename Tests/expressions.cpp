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
  typedef TileN::value_type value_type;
  typedef TileN::index index;
  typedef TileN::volume_type volume_type;
  typedef TileN::size_array size_array;
  typedef TileN::range_type range_type;
  typedef Permutation<GlobalFixture::coordinate_system::dim> PermN;

  static const range_type r;

  ExpressionFixture() : t(r, 1) {
  }

  ~ExpressionFixture() { }


  // get a unique value for the given index
  static value_type get_value(const index i) {
    index::value_type x = 1;
    value_type result = 0;
    for(index::const_iterator it = i.begin(); it != i.end(); ++it, x *= 10)
      result += *it * x;

    return result;
  }

  // make a tile to be permuted
  static TileN make_tile() {
    index start(0);
    index finish(0);
    index::value_type i = 3;
    for(index::iterator it = finish.begin(); it != finish.end(); ++it, ++i)
      *it = i;

    range_type r(start, finish);
    TileN result(r);
    for(range_type::const_iterator it = r.begin(); it != r.end(); ++it)
      result[*it] = get_value(*it);

    return result;
  }

  // make permutation definition object
  static PermN make_perm() {
    std::array<std::size_t, GlobalFixture::coordinate_system::dim> temp;
    for(std::size_t i = 0; i < temp.size(); ++i)
      temp[i] = i + 1;

    temp.back() = 0;

    return PermN(temp.begin());
  }

  TileN t;
};

const ExpressionFixture::range_type ExpressionFixture::r = ExpressionFixture::range_type(index(0), index(5));

BOOST_FIXTURE_TEST_SUITE( expressions_suite , ExpressionFixture )

BOOST_AUTO_TEST_CASE( permutation )
{
  index i;
  TileN pt = t;
  Permutation<GlobalFixture::coordinate_system::dim> p = make_perm();
  pt ^= p;

  for(range_type::const_iterator it = t.range().begin(); it != t.range().end(); ++it) {
    // Check that each element is correct
    BOOST_CHECK_EQUAL(pt[p ^ *it], t[*it]);
  }
}

BOOST_AUTO_TEST_CASE( addition )
{
  const TileN t1(r, 1);
  const TileN t2(r, 2);

  // Check + operator
  t = t1 + t2;
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 3);

//  t = t1 + TileN();
//  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
//    BOOST_CHECK_EQUAL(*it, 1);
//
//  t.resize(range_type());
//  t = TileN() + t1;
//  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
//    BOOST_CHECK_EQUAL(*it, 1);

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

//  t = t1 - TileN();
//  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
//    BOOST_CHECK_EQUAL(*it, 1);
//
//  t.resize(range_type());
//  t = TileN() - t1;
//  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
//    BOOST_CHECK_EQUAL(*it, -1);

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

