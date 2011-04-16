#include "TiledArray/tile.h"
#include "TiledArray/permutation.h"
#include <math.h>
#include <utility>
#include "unit_test_config.h"
#include "range_fixture.h"
#include <world/bufar.h>

//using namespace TiledArray;

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
  typedef TiledArray::Tile<int, GlobalFixture::element_coordinate_system> TileN;
  typedef TileN::index index;
  typedef TileN::volume_type volume_type;
  typedef TileN::size_array size_array;
  typedef TileN::range_type RangeN;

  static const RangeN r;

  TileFixture() : t(r, 1) {
  }

  ~TileFixture() { }

  TileN t;
};

const TileFixture::RangeN TileFixture::r = TileFixture::RangeN(index(0), index(5));


template<typename InIter, typename T>
bool check_val(InIter first, InIter last, const T& v, const T& tol) {
  for(; first != last; ++first)
    if(*first > v + tol || *first < v - tol)
      return false;

  return true;

}

BOOST_FIXTURE_TEST_SUITE( tile_suite , TileFixture )

BOOST_AUTO_TEST_CASE( range_accessor )
{
  BOOST_CHECK_EQUAL(t.range().start(), r.start());  // check start accessor
  BOOST_CHECK_EQUAL(t.range().finish(), r.finish());// check finish accessor
  BOOST_CHECK_EQUAL(t.range().size(), r.size());    // check size accessor
  BOOST_CHECK_EQUAL(t.range().volume(), r.volume());// check volume accessor
  BOOST_CHECK_EQUAL(t.range(), r);          // check range accessof
}

BOOST_AUTO_TEST_CASE( element_access )
{
  // check at() with array coordinate index
  BOOST_CHECK_EQUAL(t.at(index(0)), 1);
  BOOST_CHECK_EQUAL(t.at(index(4)), 1);

  // check operator[] with array coordinate index
  BOOST_CHECK_EQUAL(t[index(0)], 1);
  BOOST_CHECK_EQUAL(t[index(4)], 1);

  // check at() with ordinal index
  BOOST_CHECK_EQUAL(t.at(0), 1);
  BOOST_CHECK_EQUAL(t.at(r.volume() - 1), 1);

  // check operator[] with ordinal index
  BOOST_CHECK_EQUAL(t[0], 1);
  BOOST_CHECK_EQUAL(t[r.volume() - 1], 1);

  // check out of range error
  BOOST_CHECK_THROW(t.at(r.finish()), std::out_of_range);
  BOOST_CHECK_THROW(t.at(r.volume()), std::out_of_range);
#ifndef NDEBUG
  BOOST_CHECK_THROW(t[r.finish()], std::out_of_range);
  BOOST_CHECK_THROW(t[r.volume()], std::out_of_range);
#endif
}

BOOST_AUTO_TEST_CASE( iteration )
{
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_CLOSE(*it, 1.0, 0.000001);

  TileN t1(t);
  TileN::iterator it1 = t1.begin();
  *it1 = 2.0;

  // check iterator assignment
  BOOST_CHECK_CLOSE(*it1, 2.0, 0.000001);
  BOOST_CHECK_CLOSE(t1.at(0), 2.0, 0.000001);
  TileN t2;
  BOOST_CHECK_EQUAL(t2.begin(), t2.end());
}

BOOST_AUTO_TEST_CASE( constructor )
{
  // check default constructor
  BOOST_REQUIRE_NO_THROW(TileN t0);
  TileN t0;
  BOOST_CHECK_EQUAL(t0.range().volume(), 0u);
  BOOST_CHECK_THROW(t0.at(index(0,0,0)), std::out_of_range);
  BOOST_CHECK_EQUAL(t0.begin(), t0.end());

  // check copy constructor
  {
    BOOST_REQUIRE_NO_THROW(TileN tc(t));
    TileN tc(t);
    BOOST_CHECK_EQUAL(tc.range(), t.range());
    for(TileN::const_iterator it = tc.begin(); it != tc.end(); ++it)
      BOOST_CHECK_EQUAL(*it, 1);
  }

  // check constructing with a range
  {
    BOOST_REQUIRE_NO_THROW(TileN t1(r));
    TileN t1(r);
    BOOST_CHECK_EQUAL(t1.range(), t.range());
    for(TileN::const_iterator it = t1.begin(); it != t1.end(); ++it)
      BOOST_CHECK_EQUAL(*it, int());
  }

  // check constructing with a range and initial value.
  {
    BOOST_REQUIRE_NO_THROW(TileN t2(r, 1));
    TileN t2(r, 1);
    BOOST_CHECK_EQUAL(t2.range(), t.range());
    for(TileN::const_iterator it = t2.begin(); it != t2.end(); ++it)
      BOOST_CHECK_EQUAL(*it, 1);
  }

  // check constructing with range and iterators.
  {
    BOOST_REQUIRE_NO_THROW(TileN t3(r, t.begin(), t.end()));
    TileN t3(r, t.begin(), t.end());
    BOOST_CHECK_EQUAL(t3.range(), t.range());
    for(TileN::const_iterator it = t3.begin(); it != t3.end(); ++it)
      BOOST_CHECK_EQUAL(*it, 1);
  }
}

BOOST_AUTO_TEST_CASE( element_assignment )
{
  // verify preassignment conditions
  BOOST_CHECK_NE(t.at(0), 2);
  // check that assignment returns itself.
  BOOST_CHECK_EQUAL(t.at(0) = 2, 2);
  // check for correct assignment.
  BOOST_CHECK_EQUAL(t.at(0), 2);

  // verify preassignment conditions
  BOOST_CHECK_NE(t[1], 2);
  // check that assignment returns itself.
  BOOST_CHECK_EQUAL(t[1] = 2, 2) ;
  // check for correct assignment.
  BOOST_CHECK_EQUAL(t[1], 2);
}

BOOST_AUTO_TEST_CASE( resize )
{
  TileN t1;
  BOOST_CHECK_EQUAL(t1.range().volume(), 0u);
  t1.resize(r);
  // check new dimensions.
  BOOST_CHECK_EQUAL(t1.range(), r);
  // check new element initialization
  BOOST_CHECK_EQUAL(std::find_if(t1.begin(), t1.end(), std::bind1st(std::not_equal_to<int>(), int())), t1.end());

  TileN t2;
  BOOST_CHECK_EQUAL(std::distance(t2.begin(), t2.end()), 0);
  t2.resize(r, 1);
  BOOST_CHECK_EQUAL(t2.range(), r);
  BOOST_CHECK_EQUAL(std::distance(t2.begin(), t2.end()), static_cast<long>(r.volume()));
  // check for new element initialization
  BOOST_CHECK_EQUAL(std::find_if(t2.begin(), t2.end(), std::bind1st(std::not_equal_to<int>(), 1)), t2.end());

  // Check that the common elements are maintained in resize operation.
  RangeN r2(index(0), index(6));
  t2.resize(r2, 2);
  BOOST_CHECK_EQUAL(t2.range(), r2); // check new dimensions
  BOOST_CHECK_EQUAL(static_cast<std::size_t>(std::distance(t2.begin(), t2.end())), r2.volume());
  for(RangeN::const_iterator it = r2.begin(); it != r2.end(); ++it) {
    if(r.includes(*it))
      BOOST_CHECK_EQUAL(t2[*it], 1);
    else
      BOOST_CHECK_EQUAL(t2[*it], 2);
  }
}

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
  Tile<int, cs3> t1(r1, val.begin(), val.end());
  Tile<int, cs3> t2 = p ^ t1;
  BOOST_CHECK_EQUAL(t2.range(), p ^ t1.range()); // check that the dimensions were correctly permuted.
  BOOST_CHECK_EQUAL_COLLECTIONS(t2.begin(), t2.end(), pval.begin(), pval.end()); // check that the values were correctly permuted.

  Tile<int, cs3> t3(r3, val.begin(), val.end());
  t3 ^= p;
  BOOST_CHECK_EQUAL(t3.range(), p ^ t1.range()); // check that the dimensions were correctly permuted.
  BOOST_CHECK_EQUAL_COLLECTIONS(t3.begin(), t3.end(), pval.begin(), pval.end()); // check that the values were correctly permuted.
}

BOOST_AUTO_TEST_CASE( ostream )
{
  typedef TiledArray::CoordinateSystem<3> cs3;
  Range<cs3> r1(Range<cs3>::index(0), Range<cs3>::index(3));
  Tile<int, cs3> t1(r1, 1);
  boost::test_tools::output_test_stream output;
  output << t1;
  BOOST_CHECK( !output.is_empty( false ) ); // check for correct output.
  BOOST_CHECK( output.check_length( 80, false ) );
  BOOST_CHECK( output.is_equal("{{{1 1 1 }{1 1 1 }{1 1 1 }}{{1 1 1 }{1 1 1 }{1 1 1 }}{{1 1 1 }{1 1 1 }{1 1 1 }}}") );
}

BOOST_AUTO_TEST_CASE( serialization )
{
  std::size_t buf_size = (t.range().volume() * sizeof(int) + sizeof(TileN))*2;
  unsigned char* buf = new unsigned char[buf_size];
  madness::archive::BufferOutputArchive oar(buf, buf_size);
  oar & t;
  std::size_t nbyte = oar.size();
  oar.close();

  TileN ts;
  madness::archive::BufferInputArchive iar(buf,nbyte);
  iar & ts;
  iar.close();

  delete [] buf;

  BOOST_CHECK_EQUAL(t.range(), ts.range());
  BOOST_CHECK_EQUAL_COLLECTIONS(t.begin(), t.end(), ts.begin(), ts.end());
}

BOOST_AUTO_TEST_CASE( addition )
{
  const TileN t1(r, 1);
  const TileN t2(r, 2);

  // Check that += operator
  t += t1;
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 2);

  t += TileN();
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 2);

  t.resize(RangeN());
  t += t1;
  BOOST_CHECK_EQUAL(t.range(), t1.range());
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 1);

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
}

BOOST_AUTO_TEST_CASE( subtraction )
{
  const TileN t1(r, 1);
  const TileN t2(r, 2);

  // Check that += operator
  t -= t2;
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, -1);

  t -= TileN();
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, -1);

  t.resize(RangeN());
  t -= t1;
  BOOST_CHECK_EQUAL(t.range(), t1.range());
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, -1);

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
}

BOOST_AUTO_TEST_CASE( scalar_addition )
{

  // Check that += operator
  t += 1;
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 2);


  t.resize(RangeN());
  t += 1;
  BOOST_CHECK_EQUAL(t.range().volume(), 0ul);


  // Check + operator
  t.resize(r, 1);
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

  // Check that += operator
  t += 1;
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 2);


  t.resize(RangeN());
  t += 1;
  BOOST_CHECK_EQUAL(t.range().volume(), 0ul);


  // Check + operator
  t.resize(r, 1);
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

BOOST_AUTO_TEST_CASE( scalar_multiplication )
{

  // Check that += operator
  t *= 2;
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 2);


  t.resize(RangeN());
  t *= 2;
  BOOST_CHECK_EQUAL(t.range().volume(), 0ul);


  // Check + operator
  t.resize(r, 1);
  t = t * 2;
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 2);

  t = 2 * t;
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 4);

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
}

BOOST_AUTO_TEST_SUITE_END()

