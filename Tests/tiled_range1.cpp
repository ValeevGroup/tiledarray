#include "TiledArray/tiled_range1.h"
#include "unit_test_config.h"
#include "range_fixture.h"
#include "TiledArray/coordinates.h"

using namespace TiledArray;


const boost::array<std::size_t, 6> Range1Fixture::a = Range1Fixture::init_tiling<6>();
const Range1Fixture::range1_type::range_type Range1Fixture::tiles
    = make_range1<range1_type::coordinate_system>(0,5);
const Range1Fixture::range1_type::tile_range_type Range1Fixture::elements
    = make_range1<range1_type::tile_coordinate_system>(0,50);

BOOST_FIXTURE_TEST_SUITE( range1_suite, Range1Fixture )

BOOST_AUTO_TEST_CASE( block_accessor )
{
  BOOST_CHECK_EQUAL(tr1.tiles(), tiles);
  BOOST_CHECK_EQUAL(tr1.elements(), elements);
  BOOST_CHECK_EQUAL(tr1.tile(0), tile[0]);
  BOOST_CHECK_EQUAL(tr1.tile(1), tile[1]);
  BOOST_CHECK_EQUAL(tr1.tile(2), tile[2]);
  BOOST_CHECK_EQUAL(tr1.tile(3), tile[3]);
  BOOST_CHECK_EQUAL(tr1.tile(4), tile[4]);
  BOOST_CHECK_THROW(tr1.tile(5), std::out_of_range);
}

BOOST_AUTO_TEST_CASE( block_info )
{
  boost::array<std::size_t, 1> s1 = {{ 5 }};
  boost::array<std::size_t, 1> s2 = {{ 50 }};
  boost::array<std::size_t, 1> s3 = {{ 3 }};
  BOOST_CHECK_EQUAL(tr1.tiles().size(), s1);
  BOOST_CHECK_EQUAL(tr1.tiles().start(), 0ul);
  BOOST_CHECK_EQUAL(tr1.tiles().finish(), 5ul);
  BOOST_CHECK_EQUAL(tr1.elements().size(), s2);
  BOOST_CHECK_EQUAL(tr1.elements().start(), 0ul);
  BOOST_CHECK_EQUAL(tr1.elements().finish(), 50ul);
  BOOST_CHECK_EQUAL(tr1.tile(0).size(), s3);
  BOOST_CHECK_EQUAL(tr1.tile(0).start(), 0ul);
  BOOST_CHECK_EQUAL(tr1.tile(0).finish(), 3ul);
}

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(range1_type r0);
  range1_type r0;                  // check default construction and range info.
  BOOST_CHECK_EQUAL(r0.tiles(), make_range1<range1_type::coordinate_system>(0,0));
  BOOST_CHECK_EQUAL(r0.elements(), make_range1<range1_type::tile_coordinate_system>(0,0));
  BOOST_CHECK_EQUAL(r0.tile(ordinal_index(0)), make_range1<range1_type::tile_coordinate_system>(0,0));

  BOOST_REQUIRE_NO_THROW(range1_type r1(a.begin(), a.end()));
  range1_type r1(a.begin(), a.end());           // check construction with a
  BOOST_CHECK_EQUAL(r1.tiles(), tiles);         // iterators and the range info.
  BOOST_CHECK_EQUAL(r1.elements(), elements);
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.begin(), r1.end(), tile.begin(), tile.end());

  BOOST_REQUIRE_NO_THROW(range1_type r2(tr1));
  range1_type r2(tr1);                                   // check copy constructor
  BOOST_CHECK_EQUAL(r1.tiles(), tiles);
  BOOST_CHECK_EQUAL(r1.elements(), elements);
  BOOST_CHECK_EQUAL_COLLECTIONS(r1.begin(), r1.end(), tile.begin(), tile.end());

  BOOST_REQUIRE_NO_THROW(range1_type r3(a.begin(), a.end(), 2));
  range1_type r3(a.begin(), a.end(), 2);// check construction with a with a tile offset.
  BOOST_CHECK_EQUAL(r3.tiles(), make_range1<range1_type::coordinate_system>(2,7));
  BOOST_CHECK_EQUAL(r3.elements(), elements);
  BOOST_CHECK_EQUAL_COLLECTIONS(r3.begin(), r3.end(), tile.begin(), tile.end());

  BOOST_REQUIRE_NO_THROW(range1_type r4(a.begin() + 1, a.end()));
  range1_type r4(a.begin() + 1, a.end());// check construction with a with a element offset.
  BOOST_CHECK_EQUAL(r4.tiles(), make_range1<range1_type::coordinate_system>(0,4));
  BOOST_CHECK_EQUAL(r4.elements(), make_range1<range1_type::tile_coordinate_system>(3,50));
  BOOST_CHECK_EQUAL_COLLECTIONS(r4.begin(), r4.end(), tile.begin() + 1, tile.end());
}

BOOST_AUTO_TEST_CASE( ostream )
{
  boost::test_tools::output_test_stream output;
  output << tr1;
  BOOST_CHECK( !output.is_empty( false ) );
  BOOST_CHECK( output.check_length( 42, false ) );
  BOOST_CHECK( output.is_equal( "( tiles = [ 0, 5 ), elements = [ 0, 50 ) )" ) );
}

BOOST_AUTO_TEST_CASE( element2tile )
{
  boost::array<std::size_t, 50> e = {{0,0,0,1,1,1,1,2,2,2,3,3,3,3,3,3,3,3,3,3,
      4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4,4}};
  boost::array<std::size_t, 50> c;

  for(std::size_t i = 0; i < 50; ++i)
    c[i] = tr1.element2tile(i);

  BOOST_CHECK_EQUAL_COLLECTIONS(c.begin(), c.end(), e.begin(), e.end());
}

BOOST_AUTO_TEST_CASE( resize )
{
  range1_type r1;
  BOOST_CHECK_EQUAL(r1.resize(a.begin(), a.end(), 0), tr1); // check retiling function
  BOOST_CHECK_EQUAL(r1, tr1);
}

BOOST_AUTO_TEST_CASE( comparison )
{
  range1_type r1(tr1);
  BOOST_CHECK(r1 == tr1);     // check equality operator
  BOOST_CHECK(! (r1 != tr1)); // check not-equal operator
  r1.resize(a.begin(), a.end(), 3);
  BOOST_CHECK(! (r1 == tr1)); // check for inequality with different start  point for tiles
  BOOST_CHECK(r1 != tr1);
  boost::array<std::size_t, 6> a1 = a;
  a1[2] = 8;
  r1.resize(a1.begin(), a1.end(), 0);
  BOOST_CHECK(! (r1 == tr1)); // check for inequality with different tile boundaries.
  BOOST_CHECK(r1 != tr1);
  a1[2] = 7;
  a1[4] = 50;
  r1.resize(a1.begin(), a1.end() - 1, 0);
  BOOST_CHECK(! (r1 == tr1)); // check for inequality with different number of tiles.
  BOOST_CHECK(r1 != tr1);
}

BOOST_AUTO_TEST_CASE( iteration )
{
  BOOST_CHECK_EQUAL(const_iteration_test(tr1, tile.begin(), tile.end()), 5u);
                                    // check for proper iteration functionality.
  BOOST_CHECK_EQUAL( * tr1.find(11), tile[3]); // check that find returns an
                                             // iterator to the correct tile.

  BOOST_CHECK( tr1.find(55) == tr1.end()); // check that the iterator points to
                           // the end() iterator if the element is out of range.
}

BOOST_AUTO_TEST_CASE( assignment )
{
  range1_type r1;
  BOOST_CHECK_NE( r1, tr1);
  BOOST_CHECK_EQUAL((r1 = tr1), tr1); // check operator=
  BOOST_CHECK_EQUAL(r1, tr1);
}

BOOST_AUTO_TEST_SUITE_END()
