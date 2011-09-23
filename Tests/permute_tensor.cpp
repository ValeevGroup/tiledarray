#include "TiledArray/permute_tensor.h"
#include "TiledArray/tile.h"
#include "unit_test_config.h"

using namespace TiledArray;
using namespace TiledArray::expressions;


struct PermuteTensorFixture {
  typedef Tile<int, GlobalFixture::coordinate_system> TileN;
  typedef TileN::range_type range_type;
  typedef TileN::index index;
  typedef Permutation<GlobalFixture::coordinate_system::dim> PermN;
  typedef PermuteTensor<TileN,GlobalFixture::coordinate_system::dim> PermT;
  typedef PermT::value_type value_type;

  PermuteTensorFixture() : pt(t, p) { }

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

  static const TileN t;
  static const PermN p;

  PermT pt;
}; // struct PermuteTensorFixture

using madness::operator<<;

const PermuteTensorFixture::TileN PermuteTensorFixture::t(make_tile());
const PermuteTensorFixture::PermN PermuteTensorFixture::p(make_perm());


BOOST_FIXTURE_TEST_SUITE( permute_tensor_suite , PermuteTensorFixture )

BOOST_AUTO_TEST_CASE( dimension_accessor )
{
  range_type pr = p ^ t.range();
  BOOST_CHECK_EQUAL(pt.range(), pr);
  BOOST_CHECK_EQUAL(pt.size(), pr.volume());
}

BOOST_AUTO_TEST_CASE( constructor )
{

  BOOST_REQUIRE_NO_THROW(PermT x());

  // Test primary constructor
  {
    BOOST_REQUIRE_NO_THROW(PermT x(t,p));
    PermT x(t,p);
    BOOST_CHECK_EQUAL(x.range(), p ^ t.range());
    BOOST_CHECK_EQUAL(x.size(), t.size());
  }

  // test copy constructor
  {
    BOOST_REQUIRE_NO_THROW(PermT x(pt));
    PermT x(pt);
    BOOST_CHECK_EQUAL(x.range(), p ^ t.range());
    BOOST_CHECK_EQUAL(x.size(), t.size());
  }
}

BOOST_AUTO_TEST_CASE( element_accessor )
{
  for(range_type::const_iterator it = t.range().begin(); it != t.range().end(); ++it) {
    // Check that each element is correct
    BOOST_CHECK_EQUAL(pt[pt.range().ord(p ^ *it)], t[*it]);
  }
}

BOOST_AUTO_TEST_CASE( iterator )
{
  range_type::const_iterator rit = t.range().begin();
  for(PermT::const_iterator it = pt.begin(); it != pt.end(); ++it, ++rit) {
    // Check that iteration works correctly
    BOOST_CHECK_EQUAL(*it, t[-p ^ *rit]);
  }
}

BOOST_AUTO_TEST_SUITE_END()
