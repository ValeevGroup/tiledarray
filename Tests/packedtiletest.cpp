#include "packed_tile.h"
#include "tile.h"
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

using namespace TiledArray;

struct PackedTileBaseFixture {
  typedef Tile<double, 6> Tile6;
  typedef Tile6::index_type index6_type;
  typedef Tile6::volume_type volume6_type;
  typedef Tile6::size_array size6_array;
  typedef Tile6::range_type range6_type;

  PackedTileBaseFixture() {

    r.resize(index6_type(0,0,0,0,0,0), index6_type(3,3,3,3,3,3));
    t.resize(r.size(), 1.0);
    b[0] = 0;
    b[1] = 2;
    b[2] = 4;
    b[3] = 6;
  }

  ~PackedTileBaseFixture() { }

  Tile6 t;
  range6_type r;
  boost::array<std::size_t, 4> b;
};

struct PackedTileFixture : public PackedTileBaseFixture {
  typedef PackedTile<Tile6, 3> PackedTile3;
  typedef PackedTile<const Tile6, 3> PackedConstTile3;
  typedef PackedTile3::index_type index3_type;
  typedef PackedTile3::volume_type volume3_type;
  typedef PackedTile3::size_array size3_array;
  typedef PackedTile3::range_type range3_type;

  PackedTileFixture() : pt(t, b.begin(), b.end()), pr(index3_type(0,0,0), index3_type(9,9,9)) {

  }

  ~PackedTileFixture() { }

  PackedTile3 pt;
  range3_type pr;
};

BOOST_FIXTURE_TEST_SUITE( packed_tile_suite , PackedTileFixture )

BOOST_AUTO_TEST_CASE( accessors )
{
  BOOST_CHECK_EQUAL(pt.range(), pr);
  BOOST_CHECK_EQUAL(pt.start(), pr.start());
  BOOST_CHECK_EQUAL(pt.finish(), pr.finish());
  BOOST_CHECK_EQUAL(pt.size(), pr.size());
  BOOST_CHECK_EQUAL(pt.volume(), pr.volume());

  size3_array pw = {{81,9,1}};
  BOOST_CHECK_EQUAL(pt.weight(), pw);
}

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(PackedTile3 pt1(t, b.begin(), b.end())); // primary constructor
  PackedTile3 pt1(t,b.begin(), b.end());
  BOOST_CHECK_EQUAL(pt1.range(), pr);
  BOOST_CHECK_CLOSE(pt1.at(index3_type(0,0,0)), 1.0, 0.000001);

  BOOST_REQUIRE_NO_THROW(PackedTile3 pt2(pt)); // copy constructor
  PackedTile3 pt2(pt);
  BOOST_CHECK_EQUAL(pt2.range(), pr);
  BOOST_CHECK_CLOSE(pt2.at(index3_type(0,0,0)), 1.0, 0.000001);

#ifdef __GXX_EXPERIMENTAL_CXX0X__
  BOOST_REQUIRE_NO_THROW(PackedTile3 pt3(std::forward<PackedTile3>(PackedTile3(t, b.begin(), b.end())))); // move constructor
  PackedTile3 pt3(std::forward<PackedTile3>(PackedTile3(t, b.begin(), b.end())));
  BOOST_CHECK_EQUAL(pt3.range(), pr);
  BOOST_CHECK_CLOSE(pt3.at(index3_type(0,0,0)), 1.0, 0.000001);
#endif // __GXX_EXPERIMENTAL_CXX0X__


  const Tile6 ct(r, 1.0);
  BOOST_REQUIRE_NO_THROW( PackedConstTile3 pt4(ct, b.begin(), b.end()) ); // primary constructor w/ const tile
  PackedConstTile3 pt4(ct, b.begin(), b.end());
  BOOST_CHECK_EQUAL(pt4.range(), pr);
  BOOST_CHECK_CLOSE(pt4.at(index3_type(0,0,0)), 1.0, 0.000001);

  BOOST_REQUIRE_NO_THROW(PackedTile3 pt5(t, b.begin(), b.end(), index3_type(1,1,1))); // primary constructor w/ origin offset
  PackedTile3 pt5(t, b.begin(), b.end(), index3_type(1,1,1));
  BOOST_CHECK_EQUAL(pt5.range(), range3_type(index3_type(1,1,1), index3_type(10,10,10)));
  BOOST_CHECK_CLOSE(pt5.at(index3_type(0,0,0)), 1.0, 0.000001);
}

BOOST_AUTO_TEST_CASE( includes )
{
  BOOST_CHECK(pt.includes(index3_type(0,0,0)));   // check a point that is in both tile and slice.
  BOOST_CHECK(pt.includes(index3_type(8,8,8)));
}

BOOST_AUTO_TEST_CASE( assignment )
{
  Tile6 t1(r, 0.0);
  PackedTile3 pt1(t1, b.begin(), b.end());
  pt1 = pt;
  BOOST_CHECK(std::equal(pt1.begin(), pt1.end(), pt.begin()));

#ifdef __GXX_EXPERIMENTAL_CXX0X__
  PackedTile3 pt2(t1, b.begin(), b.end());
  pt2 = PackedTile3(t, b.begin(), b.end());
  BOOST_CHECK(std::equal(pt2.begin(), pt2.end(), pt.begin()));
#endif // __GXX_EXPERIMENTAL_CXX0X__
}

BOOST_AUTO_TEST_CASE( iteration )
{
  for(PackedTile3::iterator it = pt.begin(); it != pt.end(); ++it) { // check iteration
    BOOST_CHECK_CLOSE(*it, 1.0, 0.000001); // check for correct value
    BOOST_CHECK_EQUAL(&(*it), &(t.at(it - pt.begin()))); // check that the slice element is referencing the same place in memory
  }

  Tile6 t1(t);
  PackedTile3 pt1(t1, b.begin(), b.end());
  PackedTile3::iterator it1 = pt1.begin();
  BOOST_CHECK_CLOSE(*it1, 1.0, 0.000001);
  *it1 = 5.0;
  BOOST_CHECK_CLOSE(*it1, 5.0, 0.000001);

  const PackedTile3& pt2 = pt;
  for(PackedTile3::const_iterator it = pt2.begin(); it != pt2.end(); ++it) { // check const iteration
    BOOST_CHECK_CLOSE(*it, 1.0, 0.000001); // check for the correct value
    BOOST_CHECK_EQUAL(&(*it), &(t.at(it - pt.begin()))); // check that the slice element is referencing the same place in memory
  }
}

BOOST_AUTO_TEST_CASE( element_access )
{
  Tile6 t1(r);
  double d = 0.0;
  for(Tile6::iterator it = t1.begin(); it != t1.end(); ++it)
    *it = d++;

  PackedTile3 pt1(t1, b.begin(), b.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(pt1.begin(), pt1.end(), t1.begin(), t1.end());

  range6_type::const_iterator t_it = t1.range().begin();
  for(range3_type::const_iterator it = pt1.range().begin(); it != pt1.range().end(); ++it, ++t_it) {
    BOOST_CHECK_CLOSE(pt1.at(*it), t1.at(*t_it), 0.000001);  // check that the values for the same index are the same w/ at()
    BOOST_CHECK_CLOSE(pt1[*it], t1[*t_it], 0.000001);        // and with operator []()
  }

  BOOST_CHECK_CLOSE(pt1.at(index3_type(1,3,3)), 111.0, 0.000001);
  pt1.at(index3_type(1,3,3)) = 500.0; // check at write access
  BOOST_CHECK_CLOSE(pt1.at(index3_type(1,3,3)), 500.0, 0.000001);
  BOOST_CHECK_CLOSE(t1.at(index6_type(0,1,1,0,1,0)), 500.0, 0.000001);
  pt1[index3_type(1,3,3)] = 200.0; // check operator []() write access
  BOOST_CHECK_CLOSE(pt1.at(index3_type(1,3,3)), 200.0, 0.000001);
  BOOST_CHECK_CLOSE(t1.at(index6_type(0,1,1,0,1,0)), 200.0, 0.000001);

  const PackedTile3& pt2 = pt1;
  BOOST_CHECK_CLOSE(pt2.at(index3_type(1,0,0)), 81.0, 0.000001); // check at() const
  BOOST_CHECK_CLOSE(pt2[index3_type(1,0,0)], 81.0, 0.000001);    // check operator[] const

  const Tile6 t2(r, 1.0);
  PackedConstTile3 pt3(t, b.begin(), b.end());
  BOOST_CHECK_CLOSE(pt3.at(index3_type(1,0,0)), 1.0, 0.000001);  // check const tile access for at()
  BOOST_CHECK_CLOSE(pt3[index3_type(1,0,0)], 1.0, 0.000001);     // check const tile access for operator[]()
}

BOOST_AUTO_TEST_CASE( origin )
{
  PackedTile3 pt1(pt);
  pt1.set_origin(index3_type(1,1,1));
  BOOST_CHECK_EQUAL(pt1.range(), range3_type(index3_type(1,1,1), index3_type(10,10,10)));
}

BOOST_AUTO_TEST_SUITE_END()
