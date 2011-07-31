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
    range_type r(index(0), index(5));
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

//
//BOOST_AUTO_TEST_CASE( permutation )
//{
//  typedef TiledArray::CoordinateSystem<3, 0> cs3;
//  Permutation<3> p(1,2,0);
//  Range<cs3> r1(Range<cs3>::index(0,0,0), Range<cs3>::index(2,3,4));
//  Range<cs3> r3(r1);
//  std::array<double, 24> val =  {{0,  1,  2,  3, 10, 11, 12, 13, 20, 21, 22, 23,100,101,102,103,110,111,112,113,120,121,122,123}};
//  //         destination       {{0,100,200,300,  1,101,201,301,  2,102,202,302, 10,110,210,310, 11,111,211,311, 12,112,212,312}}
//  //         permuted index    {{0,  1,  2, 10, 11, 12,100,101,102,110,111,112,200,201,202,210,211,212,300,301,302,310,311,312}}
//  std::array<double, 24> pval = {{0, 10, 20,100,110,120,  1, 11, 21,101,111,121,  2, 12, 22,102,112,122,  3, 13, 23,103,113,123}};
//  expressions::Tile<int, cs3> t1(r1, val.begin());
//  expressions::Tile<int, cs3> t2 = (p ^ t1);
//  BOOST_CHECK_EQUAL(t2.range(), p ^ t1.range()); // check that the dimensions were correctly permuted.
//  BOOST_CHECK_EQUAL_COLLECTIONS(t2.begin(), t2.end(), pval.begin(), pval.end()); // check that the values were correctly permuted.
//
//  expressions::Tile<int, cs3> t3(r3, val.begin());
//  t3 ^= p;
//  BOOST_CHECK_EQUAL(t3.range(), p ^ t1.range()); // check that the dimensions were correctly permuted.
//  BOOST_CHECK_EQUAL_COLLECTIONS(t3.begin(), t3.end(), pval.begin(), pval.end()); // check that the values were correctly permuted.
//}

BOOST_AUTO_TEST_SUITE_END()
