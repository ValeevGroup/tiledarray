#include "tile_slice.h"
#include "tile.h"
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

using namespace TiledArray;

struct TileSliceBaseFixture {
  typedef Tile<double, 3> Tile3;
  typedef Tile3::index_type index_type;
  typedef Tile3::volume_type volume_type;
  typedef Tile3::size_array size_array;
  typedef Tile3::range_type range_type;

  TileSliceBaseFixture() {

    r.resize(index_type(0,0,0), index_type(5,5,5));
    rs.resize(index_type(0,0,0), index_type(5,5,1));
    t.resize(r.size(), 1.0);

  }

  ~TileSliceBaseFixture() { }

  Tile3 t;
  range_type r;
  range_type rs;
};

struct TileSliceFixture : public TileSliceBaseFixture {
  typedef TileSlice<TileSliceBaseFixture::Tile3> TileSlice3;

  TileSliceFixture() : ts(t, rs) {

  }

  ~TileSliceFixture() { }

  TileSlice3 ts;
};

BOOST_FIXTURE_TEST_SUITE( tile_slice_suite , TileSliceFixture )

BOOST_AUTO_TEST_CASE( accessors )
{
  BOOST_CHECK_EQUAL(ts.range(), rs);
  BOOST_CHECK_EQUAL(ts.start(), rs.start());
  BOOST_CHECK_EQUAL(ts.finish(), rs.finish());
  BOOST_CHECK_EQUAL(ts.size(), rs.size());
  BOOST_CHECK_EQUAL(ts.volume(), rs.volume());

  BOOST_CHECK_NE(ts.range(), t.range());
}

BOOST_AUTO_TEST_CASE( constructor )
{
  range_type rs(index_type(0,0,0), index_type(5,5,1));
  BOOST_REQUIRE_NO_THROW(TileSlice3 ts1(t,rs)); // primary constructor
  TileSlice3 ts1(t,rs);
  BOOST_CHECK_EQUAL(ts1.range(), rs);
  BOOST_CHECK_CLOSE(ts1.at(index_type(0,0,0)), 1.0, 0.000001);

  BOOST_REQUIRE_NO_THROW(TileSlice3 ts2(ts1)); // copy constructor
  TileSlice3 ts2(ts1);
  BOOST_CHECK_EQUAL(ts2.range(), rs);
  BOOST_CHECK_CLOSE(ts2.at(index_type(0,0,0)), 1.0, 0.000001);

#ifdef __GXX_EXPERIMENTAL_CXX0X__
  BOOST_REQUIRE_NO_THROW(TileSlice3 ts3(std::forward<TileSlice3>(TileSlice3(t,rs)))); // move constructor
  TileSlice3 ts3(std::forward<TileSlice3>(TileSlice3(t,rs)));
  BOOST_CHECK_EQUAL(ts3.range(), rs);
  BOOST_CHECK_CLOSE(ts3.at(index_type(0,0,0)), 1.0, 0.000001);
#endif // __GXX_EXPERIMENTAL_CXX0X__

  const Tile3 tc(r, 1.0);
  BOOST_REQUIRE_NO_THROW(TileSlice<const Tile3> ts4(tc,rs)); // primary constructor w/ const tile
  TileSlice<const Tile3> ts4(tc,rs);
  BOOST_CHECK_EQUAL(ts4.range(), rs);
  BOOST_CHECK_CLOSE(ts4.at(index_type(0,0,0)), 1.0, 0.000001);

  BOOST_REQUIRE_NO_THROW(TileSlice3 ts5 = t.slice(rs)); // copy constructor
  TileSlice3 ts5 = t.slice(rs);
  BOOST_CHECK_EQUAL(ts5.range(), rs);
  BOOST_CHECK_CLOSE(ts5.at(index_type(0,0,0)), 1.0, 0.000001);
}

BOOST_AUTO_TEST_CASE( includes )
{
  BOOST_CHECK(t.includes(index_type(3,3,3)));   // check a point that is in tile and not in slice.
  BOOST_CHECK(! ts.includes(index_type(3,3,3)));
  BOOST_CHECK(t.includes(index_type(3,3,0)));   // check a point that is in both tile and slice.
  BOOST_CHECK(ts.includes(index_type(3,3,0)));
}

BOOST_AUTO_TEST_CASE( assignment )
{
  Tile3 t1(r, 0.0);
  TileSlice3 ts1(t1, rs);
  ts1 = ts;
  BOOST_CHECK(std::equal(ts1.begin(), ts1.end(), ts.begin()));

#ifdef __GXX_EXPERIMENTAL_CXX0X__
  TileSlice3 ts2(t1, rs);
  ts2 = TileSlice3(t,rs);
  BOOST_CHECK(std::equal(ts2.begin(), ts2.end(), ts.begin()));
#endif // __GXX_EXPERIMENTAL_CXX0X__
}

BOOST_AUTO_TEST_CASE( iteration )
{
  for(TileSlice3::iterator it = ts.begin(); it != ts.end(); ++it) { // check iteration
    BOOST_CHECK_CLOSE(*it, t.at(it.index()), 0.000001); // check for correct value
    BOOST_CHECK_EQUAL(&(*it), &(t.at(it.index()))); // check that the slice element is referencing the same place in memory
  }

  Tile3 t1(t);
  TileSlice3 ts1(t1, rs);
  TileSlice3::iterator it1 = ts1.begin();
  BOOST_CHECK_CLOSE(*it1, 1.0, 0.000001);
  *it1 = 5.0;
  BOOST_CHECK_CLOSE(*it1, 5.0, 0.000001);

  const TileSlice3& ts2 = ts;
  for(TileSlice3::const_iterator it = ts2.begin(); it != ts2.end(); ++it) { // check const iteration
    BOOST_CHECK_CLOSE(*it, t.at(it.index()), 0.000001); // check for the correct value
    BOOST_CHECK_EQUAL(&(*it), &(t.at(it.index()))); // check that the slice element is referencing the same place in memory
  }
}

BOOST_AUTO_TEST_CASE( element_access )
{
  Tile3 t1(r);
  double d = 0.0;
  for(Tile3::iterator it = t1.begin(); it != t1.end(); ++it)
    *it = d++;

  range_type rs1 = range_type(index_type(1,0,0), index_type(2,5,5));
  TileSlice3 ts1(t1,rs1);
  Tile3::iterator t1_begin = t1.begin() + 25;
  Tile3::iterator t1_end = t1.end() - (3 * 25);
  BOOST_CHECK_EQUAL_COLLECTIONS(ts1.begin(), ts1.end(), t1_begin, t1_end);

  for(range_type::const_iterator it = ts1.range().begin(); it != ts1.range().end(); ++it) {
    BOOST_CHECK_CLOSE(ts1.at(*it), t1.at(*it), 0.000001);  // check that the values for the same index are the same w/ at()
    BOOST_CHECK_CLOSE(ts1[*it], t1[*it], 0.000001);        // and with operator []()
  }

  BOOST_CHECK_CLOSE(ts1.at(index_type(1,3,3)), 43.0, 0.000001);
  ts1.at(index_type(1,3,3)) = 500.0; // check at write access
  BOOST_CHECK_CLOSE(ts1.at(index_type(1,3,3)), 500.0, 0.000001);
  BOOST_CHECK_CLOSE(t1.at(index_type(1,3,3)), 500.0, 0.000001);
  ts1[index_type(1,3,3)] = 200.0; // check operator []() write access
  BOOST_CHECK_CLOSE(ts1.at(index_type(1,3,3)), 200.0, 0.000001);
  BOOST_CHECK_CLOSE(t1.at(index_type(1,3,3)), 200.0, 0.000001);

  const TileSlice3& ts2 = ts1;
  BOOST_CHECK_CLOSE(ts2.at(index_type(1,0,0)), 25.0, 0.000001); // check at() const
  BOOST_CHECK_CLOSE(ts2[index_type(1,0,0)], 25.0, 0.000001);    // check operator[] const

  const Tile3 t2(r, 1.0);
  TileSlice<const Tile3> ts3(t, rs);
  BOOST_CHECK_CLOSE(ts3.at(index_type(1,0,0)), 1.0, 0.000001);  // check const tile access for at()
  BOOST_CHECK_CLOSE(ts3[index_type(1,0,0)], 1.0, 0.000001);     // check const tile access for operator[]()
}


BOOST_AUTO_TEST_SUITE_END()
