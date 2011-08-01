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

  PermuteTensorFixture() : pt(t, p) {

  }

  static value_type get_value(const index i) {
    index::value_type x = 1;
    value_type result = 0;
    for(index::const_iterator it = i.begin(); it != i.end(); ++it, x *= 10)
      result += *it * x;

    return result;
  }

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

const PermuteTensorFixture::TileN PermuteTensorFixture::t(make_tile());
const PermuteTensorFixture::PermN PermuteTensorFixture::p(make_perm());


BOOST_FIXTURE_TEST_SUITE( permute_tensor_suite , PermuteTensorFixture )

BOOST_AUTO_TEST_CASE( dimension_accessor )
{
  range_type pr = p ^ t.range();
  BOOST_CHECK_EQUAL(pt.dim(), pr.dim());
  BOOST_CHECK_EQUAL_COLLECTIONS(pt.size().begin(), pt.size().end(), pr.size().begin(), pr.size().end());
  BOOST_CHECK_EQUAL(pt.volume(), pr.volume());
  BOOST_CHECK_EQUAL(pt.order(), pr.order());
}

BOOST_AUTO_TEST_CASE( constructor )
{
  PermT::size_array s(t.size().begin(), t.size().end());
  s ^= p;

  {
    BOOST_CHECK_NO_THROW(PermT x(t,p));
    PermT x(t,p);
    BOOST_CHECK_EQUAL(x.dim(), t.dim());
    BOOST_CHECK_EQUAL_COLLECTIONS(x.size().begin(), x.size().end(), s.begin(), s.end());
    BOOST_CHECK_EQUAL(x.volume(), t.volume());
    BOOST_CHECK_EQUAL(x.order(), t.order());
  }

  {
    BOOST_CHECK_NO_THROW(PermT x(pt));
    PermT x(pt);
    BOOST_CHECK_EQUAL(x.dim(), t.dim());
    BOOST_CHECK_EQUAL_COLLECTIONS(x.size().begin(), x.size().end(), s.begin(), s.end());
    BOOST_CHECK_EQUAL(x.volume(), t.volume());
    BOOST_CHECK_EQUAL(x.order(), t.order());
  }
}

BOOST_AUTO_TEST_CASE( assignment_operator )
{
  TileN t1;
  PermT x(t1, PermN());

  // Check initial conditions
  BOOST_CHECK_EQUAL(x.dim(), t1.dim());
  BOOST_CHECK_EQUAL_COLLECTIONS(x.size().begin(), x.size().end(), t1.size().begin(), t1.size().end());
  BOOST_CHECK_EQUAL(x.volume(), 0ul);
  BOOST_CHECK(x.begin() == x.end());
  BOOST_CHECK_EQUAL(x.order(), t1.order());

  x = pt;

  // Check that the tensor was copied.
  BOOST_CHECK_EQUAL(x.dim(), pt.dim());
  BOOST_CHECK_EQUAL_COLLECTIONS(x.size().begin(), x.size().end(), pt.size().begin(), pt.size().end());
  BOOST_CHECK_EQUAL(x.volume(), pt.volume());
  BOOST_CHECK_EQUAL(x.order(), pt.order());
  BOOST_CHECK_EQUAL_COLLECTIONS(x.begin(), x.end(), pt.begin(), pt.end());
}

BOOST_AUTO_TEST_CASE( permute_data )
{
  index i;
  range_type pr = p ^ t.range();

  for(range_type::const_iterator it = t.range().begin(); it != t.range().end(); ++it) {
    i = p ^ *it;
    BOOST_CHECK_EQUAL(pt[pr.ord(i)], t[*it]);
  }
}

BOOST_AUTO_TEST_CASE( iterator )
{
  index start(0);
  index finish(0);
  std::copy(pt.size().begin(), pt.size().end(), finish.begin());

  range_type r(start, finish);

  range_type::const_iterator rit = r.begin();
  for(PermT::const_iterator it = pt.begin(); it != pt.end(); ++it, ++rit) {
    BOOST_CHECK_EQUAL(*it, t[-p ^ *rit]);
  }
}

BOOST_AUTO_TEST_SUITE_END()
