#include "TiledArray/tile.h"
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
  typedef TiledArray::expressions::Tile<int, GlobalFixture::element_coordinate_system> TileN;
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
  BOOST_CHECK_THROW(t.at(r.finish()), Exception);
  BOOST_CHECK_THROW(t.at(r.volume()), Exception);
#ifndef NDEBUG
  BOOST_CHECK_THROW(t[r.finish()], Exception);
  BOOST_CHECK_THROW(t[r.volume()], Exception);
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
  BOOST_CHECK_THROW(t0.at(index(0,0,0)), Exception);
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
    std::vector<int> data;
    int v = r.volume();
    for(int i = 0; i < v; ++i)
      data.push_back(i);

    BOOST_REQUIRE_NO_THROW(TileN t3(r, data.begin()));
    TileN t3(r, data.begin());
    BOOST_CHECK_EQUAL(t3.range(), r);
    BOOST_CHECK_EQUAL_COLLECTIONS(t3.begin(), t3.end(), data.begin(), data.end());
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

BOOST_AUTO_TEST_CASE( ostream )
{
  typedef TiledArray::CoordinateSystem<3> cs3;
  Range<cs3> r1(Range<cs3>::index(0), Range<cs3>::index(3));
  expressions::Tile<int, cs3> t1(r1, 1);
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
  t -= t2;
  BOOST_CHECK_EQUAL(t.range(), t1.range());
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, -2);
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
}

BOOST_AUTO_TEST_CASE( scalar_subtraction )
{

  // Check that += operator
  t -= 1;
  for(TileN::const_iterator it = t.begin(); it != t.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 0);


  t.resize(RangeN());
  t -= 1;
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
}

BOOST_AUTO_TEST_SUITE_END()

