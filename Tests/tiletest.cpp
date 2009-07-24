#include <tile.h>
#include <tile_math.h>
#include <array_storage.h>
#include <tiled_range1.h>
#include <coordinates.h>
#include <permutation.h>
#include <iostream>
#include <math.h>
#include <utility>
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

using namespace TiledArray;

// Element Generation object test.
template<typename T, typename Index>
class gen {
public:
  const T operator ()(const Index& i) {
    typedef typename Index::index index_t;
	index_t result = 0;
    index_t e = 0;
    for(unsigned int d = 0; d < Index::dim(); ++d) {
      e = i[d] * static_cast<index_t>(std::pow(10.0, static_cast<int>(Index::dim()-d-1)));
      result += e;
    }

    return result;
  }
};

struct TileFixture {
  typedef Tile<double, 3> Tile3;
  typedef Tile3::index_type index_type;
  typedef Tile3::volume_type volume_type;
  typedef Tile3::size_array size_array;
  typedef Tile3::range_type range_type;

  TileFixture() {

    r.resize(index_type(0,0,0), index_type(5,5,5));
    rs.resize(index_type(0,0,0), index_type(5,5,1));
    t.resize(r.size(), 1.0);

  }

  ~TileFixture() { }

  Tile3 t;
  range_type r;
  range_type rs;
};

template<typename InIter, typename T>
bool check_val(InIter first, InIter last, const T& v, const T& tol = 0.000001) {
  for(; first != last; ++first)
    if(*first > v + tol || *first < v - tol)
      return false;

  return true;

}

BOOST_FIXTURE_TEST_SUITE( tile_suite , TileFixture )

BOOST_AUTO_TEST_CASE( accessor )
{
  BOOST_CHECK_EQUAL(t.start(), r.start());  // check start accessor
  BOOST_CHECK_EQUAL(t.finish(), r.finish());// check finish accessor
  BOOST_CHECK_EQUAL(t.size(), r.size());    // check size accessor
  BOOST_CHECK_EQUAL(t.volume(), r.volume());// check volume accessor
  BOOST_CHECK_EQUAL(t.range(), r);          // check range accessof
}

BOOST_AUTO_TEST_CASE( element_access )
{
  BOOST_CHECK_CLOSE(t.at(index_type(0,0,0)), 1.0, 0.000001); // check at() with array coordinate index
  BOOST_CHECK_CLOSE(t.at(index_type(4,4,4)), 1.0, 0.000001);
  BOOST_CHECK_CLOSE(t[index_type(0,0,0)], 1.0, 0.000001);    // check operator[] with array coordinate index
  BOOST_CHECK_CLOSE(t[index_type(4,4,4)], 1.0, 0.000001);
  BOOST_CHECK_CLOSE(t.at(0), 1.0, 0.000001);                 // check at() with ordinal index
  BOOST_CHECK_CLOSE(t.at(r.volume() - 1), 1.0, 0.000001);
  BOOST_CHECK_CLOSE(t[0], 1.0, 0.000001);                    // check operator[] with ordinal index
  BOOST_CHECK_CLOSE(t[r.volume() - 1], 1.0, 0.000001);
  BOOST_CHECK_THROW(t.at(r.finish()), std::out_of_range); // check out of range error
  BOOST_CHECK_THROW(t.at(r.volume()), std::out_of_range);
#ifndef NDEBUG
  BOOST_CHECK_THROW(t[r.finish()], std::out_of_range);
  BOOST_CHECK_THROW(t[r.volume()], std::out_of_range);
#endif
}

BOOST_AUTO_TEST_CASE( iteration )
{
  for(Tile3::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_CLOSE(*it, 1.0, 0.000001);

  Tile3 t1(t);
  Tile3::iterator it1 = t1.begin();
  *it1 = 2.0;
  BOOST_CHECK_CLOSE(*it1, 2.0, 0.000001); // check iterator assignment
  BOOST_CHECK_CLOSE(t1.at(0), 2.0, 0.000001);
  Tile3 t2;
  BOOST_CHECK_EQUAL(t2.begin(), t2.end());
}

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(Tile3 t0); // check default constructor
  Tile3 t0;
  BOOST_CHECK_EQUAL(t0.volume(), 0);
  BOOST_CHECK_THROW(t0.at(index_type(0,0,0)), std::out_of_range);

  BOOST_REQUIRE_NO_THROW(Tile3 tc(t)); // check copy constructor
  Tile3 tc(t);
  BOOST_CHECK_EQUAL(tc.range(), t.range());
  BOOST_CHECK(check_val(tc.begin(), tc.end(), 1.0));

  BOOST_REQUIRE_NO_THROW(Tile3 t1(r)); // check constructing with a range
  Tile3 t1(r);
  BOOST_CHECK_EQUAL(t1.range(), t.range());
  BOOST_CHECK(check_val(t1.begin(), t1.end(), 0.0));

  BOOST_REQUIRE_NO_THROW(Tile3 t2(r, 1)); // check constructing with a range and initial value.
  Tile3 t2(r, 1);
  BOOST_CHECK_EQUAL(t2.range(), t.range());
  BOOST_CHECK(check_val(t2.begin(), t2.end(), 1.0));

  BOOST_REQUIRE_NO_THROW(Tile3 t3(r, t.begin(), t.end())); // check constructing with range and iterators.
  Tile3 t3(r, t.begin(), t.end());
  BOOST_CHECK_EQUAL(t3.range(), t.range());
  BOOST_CHECK(check_val(t3.begin(), t3.end(), 1.0));

  BOOST_REQUIRE_NO_THROW(Tile3 t11(r, t.begin(), t.end() - 3)); // check constructing with iterators that do not cover the range.
  Tile3 t11(r, t.begin(), t.end() - 3);
  BOOST_CHECK(check_val(t11.begin(), t11.end() - 3, 1.0));
  BOOST_CHECK(check_val(t11.end() - 3, t11.end(), double()));
}

BOOST_AUTO_TEST_CASE( element_assignment )
{
  Tile3 t1(r);
  BOOST_CHECK_NE(t1.at(0), 1.0);                    // verify preassignment conditions
  BOOST_CHECK_CLOSE(t1.at(0) = 1.0, 1.0, 0.000001); // check that assignment returns itself.
  BOOST_CHECK_CLOSE(t1.at(0), 1.0, 0.000001);       // check for correct assignment.
  BOOST_CHECK_NE(t1[1], 1.0);                       // verify preassignment conditions
  BOOST_CHECK_CLOSE(t1[1] = 1.0, 1.0, 0.000001) ;   // check that assignment returns itself.
  BOOST_CHECK_CLOSE(t1[1], 1.0, 0.000001);          // check for correct assignment.
}

BOOST_AUTO_TEST_CASE( resize )
{
  Tile3 t1;
  t1.resize(r.size());
  BOOST_CHECK_EQUAL(t1.range(), r); // check new dimensions.
  BOOST_CHECK(check_val(t1.begin(), t1.end(), double())); // check new element initialization

  Tile3 t2;
  t2.resize(r.size(), 1);
  BOOST_CHECK_EQUAL(t2.range(), r);
  BOOST_CHECK(check_val(t2.begin(), t2.end(), 1.0)); // check for new element initialization

  size_array s = {{6,6,6}};
  t2.resize(s, 0);
  BOOST_CHECK_EQUAL(t2.size(), s); // check new dimensions
  BOOST_CHECK_CLOSE(t2.at(index_type(0,0,0)), 1.0, 0.000001); // check that previous values are maintained.
  BOOST_CHECK_CLOSE(t2.at(index_type(4,0,0)), 1.0, 0.000001);
  BOOST_CHECK_CLOSE(t2.at(index_type(0,4,0)), 1.0, 0.000001);
  BOOST_CHECK_CLOSE(t2.at(index_type(4,4,0)), 1.0, 0.000001);
  BOOST_CHECK_CLOSE(t2.at(index_type(0,0,4)), 1.0, 0.000001);
  BOOST_CHECK_CLOSE(t2.at(index_type(4,0,4)), 1.0, 0.000001);
  BOOST_CHECK_CLOSE(t2.at(index_type(0,4,4)), 1.0, 0.000001);
  BOOST_CHECK_CLOSE(t2.at(index_type(4,4,4)), 1.0, 0.000001);
  BOOST_CHECK_CLOSE(t2.at(index_type(5,0,0)), 0.0, 0.000001); // check that previous values are maintained.
  BOOST_CHECK_CLOSE(t2.at(index_type(0,5,0)), 0.0, 0.000001);
  BOOST_CHECK_CLOSE(t2.at(index_type(5,5,0)), 0.0, 0.000001);
  BOOST_CHECK_CLOSE(t2.at(index_type(0,0,5)), 0.0, 0.000001);
  BOOST_CHECK_CLOSE(t2.at(index_type(5,0,5)), 0.0, 0.000001);
  BOOST_CHECK_CLOSE(t2.at(index_type(0,5,5)), 0.0, 0.000001);
  BOOST_CHECK_CLOSE(t2.at(index_type(5,5,5)), 0.0, 0.000001);
}

BOOST_AUTO_TEST_CASE( set_origin )
{
  Tile3 t1(t);
  t1.set_origin(index_type(1,1,1));
  range_type r1(index_type(1,1,1), index_type(6,6,6));
  BOOST_CHECK_EQUAL(t1.range(), r1); // check new dimensions
  BOOST_CHECK(check_val(t1.begin(), t1.end(), 1)); // check that values are maintained
}

BOOST_AUTO_TEST_CASE( permutation )
{
  Permutation<3> p(1,2,0);
  range_type r1(index_type(0,0,0), index_type(2,3,4));
  boost::array<double, 24> val =  {{0,  1,  2,  3, 10, 11, 12, 13, 20, 21, 22, 23,100,101,102,103,110,111,112,113,120,121,122,123}};
  //         destination       {{0,100,200,300,  1,101,201,301,  2,102,202,302, 10,110,210,310, 11,111,211,311, 12,112,212,312}}
  //         permuted index    {{0,  1,  2, 10, 11, 12,100,101,102,110,111,112,200,201,202,210,211,212,300,301,302,310,311,312}}
  boost::array<double, 24> pval = {{0, 10, 20,100,110,120,  1, 11, 21,101,111,121,  2, 12, 22,102,112,122,  3, 13, 23,103,113,123}};
  Tile3 t1(r1, val.begin(), val.end());
  Tile3 t2 = p ^ t1;
  BOOST_CHECK_EQUAL(t2.range(), p ^ r1); // check that the dimensions were correctly permuted.
  BOOST_CHECK_EQUAL_COLLECTIONS(t2.begin(), t2.end(), pval.begin(), pval.end()); // check that the values were correctly permuted.

  Tile3 t3(r1, val.begin(), val.end());
  t3 ^= p;
  BOOST_CHECK_EQUAL(t3.range(), p ^ r1); // check that the dimensions were correctly permuted.
  BOOST_CHECK_EQUAL_COLLECTIONS(t3.begin(), t3.end(), pval.begin(), pval.end()); // check that the values were correctly permuted.
}

BOOST_AUTO_TEST_CASE( ostream )
{
  Tile3 t1(range_type(index_type(0,0,0), index_type(3,3,3)), 1);
  boost::test_tools::output_test_stream output;
  output << t1;
  BOOST_CHECK( !output.is_empty( false ) ); // check for correct output.
  BOOST_CHECK( output.check_length( 80, false ) );
  BOOST_CHECK( output.is_equal("{{{1 1 1 }{1 1 1 }{1 1 1 }}{{1 1 1 }{1 1 1 }{1 1 1 }}{{1 1 1 }{1 1 1 }{1 1 1 }}}") );
}

BOOST_AUTO_TEST_CASE( addition )
{
  Tile3 t1(t);
  Tile3 t2(t.range(), 2);
  Tile3 t3;
  t3("a,b,c") = t1("a,b,c") + t2("a,b,c");
  BOOST_CHECK(check_val(t3.begin(), t3.end(), 3.0));//  check that the values were added correctly.

//  Tile3 t4(t);
//  t4("a,b,c") += t2("a,b,c");
//  BOOST_CHECK(check_val(t4.begin(), t4.end(), 3.0));//  check that the values were added correctly.
}

BOOST_AUTO_TEST_CASE( subtract )
{
  Tile3 t1(t);
  Tile3 t2(t.range(), 2);
  Tile3 t3;
  t3("a,b,c") = t2("a,b,c") - t1("a,b,c");
  BOOST_CHECK(check_val(t3.begin(), t3.end(), 1.0));//  check that the values were added correctly.

//  Tile3 t4(t2);
//  t4("a,b,c") -= t1("a,b,c");
//  BOOST_CHECK(check_val(t4.begin(), t4.end(), 1.0));//  check that the values were added correctly.
}
/*
BOOST_AUTO_TEST_CASE( addition_scalar )
{
  Tile3 t1(t);
  Tile3 t2 = t1 + 2;
  BOOST_CHECK(check_val(t2.begin(), t2.end(), 3.0));//  check that the values were added correctly.

  Tile3 t3(t);
  t3 += 2;
  BOOST_CHECK(check_val(t3.begin(), t3.end(), 3.0));//  check that the values were added correctly.
}

BOOST_AUTO_TEST_CASE( subtract_scalar )
{
  Tile3 t1(r, 2);
  Tile3 t2 = t1 - 1;
  BOOST_CHECK(check_val(t2.begin(), t2.end(), 1.0));//  check that the values were added correctly.

  Tile3 t3(r, 2);
  t3 -= 1;
  BOOST_CHECK(check_val(t3.begin(), t3.end(), 1.0));//  check that the values were added correctly.
}

BOOST_AUTO_TEST_CASE( multiply_scalar )
{
  Tile3 t1(r, 1);
  Tile3 t2 = t1 * 3;
  BOOST_CHECK(check_val(t2.begin(), t2.end(), 3.0));//  check that the values were added correctly.
  Tile3 t3 = 3 * t1;
  BOOST_CHECK(check_val(t3.begin(), t3.end(), 3.0));

  Tile3 t4(r, 1);
  t4 *= 3;
  BOOST_CHECK(check_val(t4.begin(), t4.end(), 3.0));//  check that the values were added correctly.
}

BOOST_AUTO_TEST_CASE( negate )
{
  Tile3 t1(r, 1);
  Tile3 t2 = -t1;
  BOOST_CHECK(check_val(t2.begin(), t2.end(), -1.0));//  check that the values were added correctly.
}


BOOST_AUTO_TEST_CASE( contract_1x1 )
{
  Tile<double,1>::range_type r1(make_tile_range1<std::size_t>(0,5));
  Tile<double,1> t1(r1, 3.0);
  Tile<double,1> t2(r1, 2.0);
  double tr = 0.0;
  detail::contract(t1, t2, tr);
  BOOST_CHECK_EQUAL(tr, 30.0);
}

BOOST_AUTO_TEST_CASE( contract_3x3 )
{
  Tile3 t1(r, 3);
  Tile3 t2(r, 2);
  Tile<double,4>::range_type rr(Tile<double,4>::index_type(0,0,0,0), Tile<double,4>::index_type(5,5,5,5));
  Tile<double,4> tr(rr, 0.0);
  detail::contract_aic_x_bid(t1, t2, tr);
  BOOST_CHECK(check_val(tr.begin(), tr.end(), 30.0));
}

BOOST_AUTO_TEST_CASE( variable_list )
{
  detail::AnnotatedTile<Tile3> t1 = t("a,b,c");
  BOOST_CHECK_EQUAL_COLLECTIONS(t1.tile().begin(), t1.tile().end(), t.begin(), t.end());
  BOOST_CHECK_EQUAL(t1.vars().get(0), "a");
  BOOST_CHECK_EQUAL(t1.vars().get(1), "b");
  BOOST_CHECK_EQUAL(t1.vars().get(2), "c");

  BOOST_CHECK_THROW(t("a,b,c,d"), std::runtime_error);
  BOOST_CHECK_THROW(t("a,b"), std::runtime_error);

}
*/
BOOST_AUTO_TEST_SUITE_END()

