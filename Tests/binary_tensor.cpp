#include "TiledArray/binary_tensor.h"
#include "TiledArray/tile.h"
#include "unit_test_config.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

struct BinaryTensorFixture {
  typedef Tile<int, GlobalFixture::coordinate_system> TileN;
  typedef TileN::range_type range_type;
  typedef TileN::index index;
  typedef std::plus<TileN::value_type> plus_op;
  typedef BinaryTensor<TileN,TileN,plus_op> BinaryT;
  typedef BinaryT::value_type value_type;

  BinaryTensorFixture() : bt(t2, t3, op) { }

  // make a tile to be permuted
  static TileN make_tile(TileN::value_type value) {
    index start(0);
    index finish(0);
    index::value_type i = 3;
    for(index::iterator it = finish.begin(); it != finish.end(); ++it, ++i)
      *it = i;

    range_type r(start, finish);

    return TileN(r, value);
  }


  static const TileN t2;
  static const TileN t3;
  static const plus_op op;

  BinaryT bt;
}; // struct BinaryTensorFixture


const BinaryTensorFixture::TileN BinaryTensorFixture::t2 = make_tile(2);
const BinaryTensorFixture::TileN BinaryTensorFixture::t3 = make_tile(3);

const BinaryTensorFixture::plus_op BinaryTensorFixture::op =
    BinaryTensorFixture::plus_op();

BOOST_FIXTURE_TEST_SUITE( binary_tensor_suite , BinaryTensorFixture )

BOOST_AUTO_TEST_CASE( dimension_accessor )
{
  BOOST_CHECK_EQUAL(bt.dim(), t2.dim());
  BOOST_CHECK_EQUAL_COLLECTIONS(bt.size().begin(), bt.size().end(), t2.size().begin(), t2.size().end());
  BOOST_CHECK_EQUAL(bt.volume(), t2.volume());
  BOOST_CHECK_EQUAL(bt.order(), t2.order());
}

BOOST_AUTO_TEST_CASE( constructor )
{
  // Test the default constructor
  BOOST_REQUIRE_NO_THROW(BinaryT x());

  // Test primary constructor
  {
    BOOST_REQUIRE_NO_THROW(BinaryT x(t2, t3, op));
    BinaryT x(t2, t3, op);
    BOOST_CHECK_EQUAL(x.dim(), t2.dim());
    BOOST_CHECK_EQUAL_COLLECTIONS(x.size().begin(), x.size().end(), t2.size().begin(), t2.size().end());
    BOOST_CHECK_EQUAL(x.volume(), t2.volume());
    BOOST_CHECK_EQUAL(x.order(), t2.order());
  }

  // test copy constructor
  {
    BOOST_REQUIRE_NO_THROW(BinaryT x(bt));
    BinaryT x(bt);
    BOOST_CHECK_EQUAL(x.dim(), t2.dim());
    BOOST_CHECK_EQUAL_COLLECTIONS(x.size().begin(), x.size().end(), t2.size().begin(), t2.size().end());
    BOOST_CHECK_EQUAL(x.volume(), t2.volume());
    BOOST_CHECK_EQUAL(x.order(), t2.order());
  }
}

BOOST_AUTO_TEST_CASE( element_accessor )
{

  for(BinaryT::size_type i = 0; i < bt.volume(); ++i) {
    // Check that each element is correct
    BOOST_CHECK_EQUAL(bt[i], op(t2[i], t3[i]));
  }
}

BOOST_AUTO_TEST_CASE( iterator )
{
  TileN::size_type i = 0;
  for(BinaryT::const_iterator it = bt.begin(); it != bt.end(); ++it, ++i) {
    // Check that iteration works correctly
    BOOST_CHECK_EQUAL(*it, op(t2[i], t3[i]));
  }
}

BOOST_AUTO_TEST_SUITE_END()
