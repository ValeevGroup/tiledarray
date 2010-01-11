#include "TiledArray/annotated_tile.h"
#include "TiledArray/tile.h"
#include "TiledArray/tiled_range1.h"
#include "TiledArray/coordinates.h"
#include "TiledArray/permutation.h"
#include <iostream>
#include <math.h>
#include <utility>
#include <boost/test/unit_test.hpp>
//#include <boost/test/output_test_stream.hpp>

using namespace TiledArray;
using TiledArray::expressions::VariableList;
using TiledArray::expressions::tile::AnnotatedTile;

struct TileFixture {
  typedef Tile<double, 3> Tile3;
  typedef Tile3::index_type index_type;
  typedef Tile3::volume_type volume_type;
  typedef Tile3::size_array size_array;
  typedef Tile3::range_type range_type;

  TileFixture() {
    r.resize(index_type(0,0,0), index_type(5,5,5));
    t.resize(r.size(), 1.0);
  }

  ~TileFixture() { }

  Tile3 t;
  range_type r;
  range_type rs;
};

struct AnnotatedTileFixture : public TileFixture {
  typedef AnnotatedTile<double> ATile;
  typedef AnnotatedTile<const double> ConstATile;

  AnnotatedTileFixture() : v("a,b,c"), a(t, v), ca(const_cast<const Tile3&>(t), v), s(t.size()) { }
  ~AnnotatedTileFixture() { }

  template<typename InIter, typename T>
  bool check_val(InIter first, InIter last, const T& v, const T& tol = 0.000001) {
    for(; first != last; ++first)
      if(*first > v + tol || *first < v - tol)
        return false;

    return true;
  }

  VariableList v;
  ATile a;
  ConstATile ca;
  Tile3::size_array s;
};

BOOST_FIXTURE_TEST_SUITE( annotated_tile_suite , AnnotatedTileFixture )

BOOST_AUTO_TEST_CASE( accessor )
{
  BOOST_CHECK_EQUAL(a.data(), t.data()); // check that the same data pointer is used.

  BOOST_CHECK_EQUAL_COLLECTIONS(a.size().begin(), a.size().end(),
      t.size().begin(), t.size().end());    // check size accessor
  BOOST_CHECK_EQUAL_COLLECTIONS(a.weight().begin(), a.weight().end(),
      t.weight().begin(), t.weight().end()); // check weight accessor
  BOOST_CHECK_EQUAL(a.vars(), v); // check variable list accessor
  BOOST_CHECK_EQUAL(a.dim(), t.dim()); // check dimension accessor
  BOOST_CHECK_EQUAL(a.volume(), t.volume());// check volume accessor
}

BOOST_AUTO_TEST_CASE( element_access )
{
  BOOST_CHECK_CLOSE(a.at(index_type(0,0,0)), 1.0, 0.000001); // check at() with array coordinate index
  BOOST_CHECK_CLOSE(a.at(index_type(4,4,4)), 1.0, 0.000001);
  BOOST_CHECK_CLOSE(a[index_type(0,0,0)], 1.0, 0.000001);    // check operator[] with array coordinate index
  BOOST_CHECK_CLOSE(a[index_type(4,4,4)], 1.0, 0.000001);
  BOOST_CHECK_CLOSE(a.at(0), 1.0, 0.000001);                 // check at() with ordinal index
  BOOST_CHECK_CLOSE(a.at(r.volume() - 1), 1.0, 0.000001);
  BOOST_CHECK_CLOSE(a[0], 1.0, 0.000001);                    // check operator[] with ordinal index
  BOOST_CHECK_CLOSE(a[r.volume() - 1], 1.0, 0.000001);
  BOOST_CHECK_THROW(a.at(r.finish()), std::out_of_range); // check out of range error
  BOOST_CHECK_THROW(a.at(r.volume()), std::out_of_range);
}

BOOST_AUTO_TEST_CASE( include )
{
  BOOST_CHECK(a.includes(index_type(0,0,0)));
  BOOST_CHECK(a.includes(index_type(4,4,4)));
  BOOST_CHECK(! a.includes(index_type(4,4,5)));
}

BOOST_AUTO_TEST_CASE( iteration )
{
  for(ATile::const_iterator it = a.begin(); it != a.end(); ++it)
    BOOST_CHECK_CLOSE(*it, 1.0, 0.000001);

  ATile a1(a);
  ATile::iterator it1 = a1.begin();
  *it1 = 2.0;
  BOOST_CHECK_CLOSE(*it1, 2.0, 0.000001); // check iterator assignment
  BOOST_CHECK_CLOSE(a1.at(0), 2.0, 0.000001);
}

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(ATile ac(a)); // check copy constructor
  ATile ac(a);
  BOOST_CHECK_EQUAL_COLLECTIONS(ac.size().begin(), ac.size().end(),
      a.size().begin(), a.size().end());
  BOOST_CHECK(check_val(t.begin(), t.end(), 1.0)); // Verify tile data
  BOOST_CHECK(check_val(a.begin(), a.end(), 1.0)); // verify annotated tile data.
  BOOST_CHECK_EQUAL(ac.begin(), a.begin()); // make sure iterators are the same
  BOOST_CHECK_EQUAL(ac.end(), a.end());
  BOOST_CHECK(check_val(ac.begin(), ac.end(), 1.0));

  BOOST_REQUIRE_NO_THROW(ATile a1(s, v, 2.0)); // check non-reference annotated tile with
  ATile a1(s, v, 2.0);                         // a constant initial value.
  BOOST_CHECK_EQUAL_COLLECTIONS(a1.size().begin(), a1.size().end(), s.begin(), s.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(a1.weight().begin(), a1.weight().end(), a.weight().begin(), a.weight().end());
  BOOST_CHECK_EQUAL(a1.volume(), 125u);
  BOOST_CHECK_EQUAL(a1.vars(), v);
  BOOST_CHECK(check_val(a1.begin(), a1.end(), 2.0));

  BOOST_REQUIRE_NO_THROW(ATile a2(s, v, t.begin(), t.end())); // check non-reference annotated tile with
  ATile a2(s, v, t.begin(), t.end());                         // a initialization list.
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.size().begin(), a2.size().end(), s.begin(), s.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.weight().begin(), a2.weight().end(), a.weight().begin(), a.weight().end());
  BOOST_CHECK_EQUAL(a2.volume(), 125u);
  BOOST_CHECK_EQUAL(a2.vars(), v);
  BOOST_CHECK(check_val(a2.begin(), a2.end(), 1.0));
}

BOOST_AUTO_TEST_CASE( element_assignment )
{
  ATile a1(s, v, 2.0);
  BOOST_CHECK_NE(a1.at(0), 1.0);                    // verify preassignment conditions
  BOOST_CHECK_CLOSE(a1.at(0) = 1.0, 1.0, 0.000001); // check that assignment returns itself.
  BOOST_CHECK_CLOSE(a1.at(0), 1.0, 0.000001);       // check for correct assignment.
  BOOST_CHECK_NE(a1[1], 1.0);                       // verify preassignment conditions
  BOOST_CHECK_CLOSE(a1[1] = 1.0, 1.0, 0.000001) ;   // check that assignment returns itself.
  BOOST_CHECK_CLOSE(a1[1], 1.0, 0.000001);          // check for correct assignment.
}

BOOST_AUTO_TEST_CASE( permutation )
{
  Permutation<3> p(1,2,0);
  Tile3::size_array s1 = {{2, 3, 4}};
  boost::array<double, 24> val =  {{0,  1,  2,  3, 10, 11, 12, 13, 20, 21, 22, 23,100,101,102,103,110,111,112,113,120,121,122,123}};
  //         destination          {{0,100,200,300,  1,101,201,301,  2,102,202,302, 10,110,210,310, 11,111,211,311, 12,112,212,312}}
  //         permuted index       {{0,  1,  2, 10, 11, 12,100,101,102,110,111,112,200,201,202,210,211,212,300,301,302,310,311,312}}
  boost::array<double, 24> pval = {{0, 10, 20,100,110,120,  1, 11, 21,101,111,121,  2, 12, 22,102,112,122,  3, 13, 23,103,113,123}};
  ATile a1(s1, v, val.begin(), val.end());
  ATile a2 = p ^ a1;

  BOOST_CHECK_EQUAL(a2.size()[0], 4u); // check that the dimensions were correctly permuted.
  BOOST_CHECK_EQUAL(a2.size()[1], 2u);
  BOOST_CHECK_EQUAL(a2.size()[2], 3u);
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.begin(), a2.end(), pval.begin(), pval.end()); // check that the values were correctly permuted.

  ATile a3(s1, v, val.begin(), val.end());
  a3 ^= p;
  BOOST_CHECK_EQUAL(a3.size()[0], 4u); // check that the dimensions were correctly permuted.
  BOOST_CHECK_EQUAL(a3.size()[1], 2u);
  BOOST_CHECK_EQUAL(a3.size()[2], 3u);
  BOOST_CHECK_EQUAL_COLLECTIONS(a3.begin(), a3.end(), pval.begin(), pval.end()); // check that the values were correctly permuted.
}

BOOST_AUTO_TEST_SUITE_END()
