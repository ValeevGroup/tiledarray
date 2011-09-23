#include "TiledArray/unary_tensor.h"
#include "TiledArray/tile.h"
#include "unit_test_config.h"

using namespace TiledArray;
using namespace TiledArray::expressions;

struct UnaryTensorFixture {
  typedef Tile<int, GlobalFixture::coordinate_system> TileN;
  typedef TileN::range_type range_type;
  typedef TileN::index index;
  typedef std::binder1st<std::multiplies<TileN::value_type> > scale_op;
  typedef UnaryTensor<TileN,scale_op> UnaryT;
  typedef UnaryT::value_type value_type;

  UnaryTensorFixture() : ut(t, op) { }

  // make a tile to be permuted
  static TileN make_tile() {
    index start(0);
    index finish(0);
    index::value_type i = 3;
    for(index::iterator it = finish.begin(); it != finish.end(); ++it, ++i)
      *it = i;

    range_type r(start, finish);

    return TileN(r, 3);
  }


  static const TileN t;
  static const scale_op op;

  UnaryT ut;
}; // struct UnaryTensorFixture


const UnaryTensorFixture::TileN UnaryTensorFixture::t = make_tile();
const UnaryTensorFixture::scale_op UnaryTensorFixture::op =
    scale_op(std::multiplies<UnaryTensorFixture::TileN::value_type>(), 2);


BOOST_FIXTURE_TEST_SUITE( unary_tensor_suite , UnaryTensorFixture )

BOOST_AUTO_TEST_CASE( dimension_accessor )
{
  BOOST_CHECK_EQUAL(ut.range(), t.range());
  BOOST_CHECK_EQUAL(ut.size(), t.size());
}

BOOST_AUTO_TEST_CASE( constructor )
{
  // Test default constructor
  BOOST_REQUIRE_NO_THROW(UnaryT x());

  // Test primary constructor
  {
    BOOST_REQUIRE_NO_THROW(UnaryT x(t, op));
    UnaryT x(t, op);
    BOOST_CHECK_EQUAL(x.range(), t.range());
    BOOST_CHECK_EQUAL(x.size(), t.size());
  }

  // test copy constructor
  {
    BOOST_REQUIRE_NO_THROW(UnaryT x(ut));
    UnaryT x(ut);
    BOOST_CHECK_EQUAL(x.range(), t.range());
    BOOST_CHECK_EQUAL(x.size(), t.size());
  }
}

BOOST_AUTO_TEST_CASE( element_accessor )
{

  for(UnaryT::size_type i = 0; i < ut.size(); ++i) {
    // Check that each element is correct
    BOOST_CHECK_EQUAL(ut[i], op(t[i]));
  }
}

BOOST_AUTO_TEST_CASE( iterator )
{
  TileN::size_type i = 0;
  for(UnaryT::const_iterator it = ut.begin(); it != ut.end(); ++it, ++i) {
    // Check that iteration works correctly
    BOOST_CHECK_EQUAL(*it, op(t[i]));
  }
}

BOOST_AUTO_TEST_SUITE_END()
