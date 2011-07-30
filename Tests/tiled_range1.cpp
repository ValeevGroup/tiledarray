#include "TiledArray/tiled_range1.h"
#include "unit_test_config.h"
#include "range_fixture.h"
#include "TiledArray/coordinates.h"
#include <sstream>

using namespace TiledArray;


const std::array<std::size_t, 6> Range1Fixture::a = Range1Fixture::init_tiling<6>();
const Range1Fixture::range1_type::range_type Range1Fixture::tiles(0, Range1Fixture::a.size() - 1);
const Range1Fixture::range1_type::tile_range_type Range1Fixture::elements(Range1Fixture::a.front(), Range1Fixture::a.back());

BOOST_FIXTURE_TEST_SUITE( range1_suite, Range1Fixture )

BOOST_AUTO_TEST_CASE( range_accessor )
{
  BOOST_CHECK_EQUAL(tr1.tiles(), tiles);
  BOOST_CHECK_EQUAL(tr1.elements(), elements);

  // Check individual tiles
  for(std::size_t i = 0; i < a.size() - 1; ++i)
    BOOST_CHECK_EQUAL(tr1.tile(i), range1_type::tile_range_type(a[i], a[i + 1]));
}

BOOST_AUTO_TEST_CASE( range_info )
{
  BOOST_CHECK_EQUAL(tr1.tiles().size(), a.size() - 1);
  BOOST_CHECK_EQUAL(tr1.tiles().start(), 0ul);
  BOOST_CHECK_EQUAL(tr1.tiles().finish(), a.size() - 1);
  BOOST_CHECK_EQUAL(tr1.elements().size(), a.back());
  BOOST_CHECK_EQUAL(tr1.elements().start(), 0ul);
  BOOST_CHECK_EQUAL(tr1.elements().finish(), a.back());
  for(std::size_t i = 0; i < a.size() - 1; ++i) {
    BOOST_CHECK_EQUAL(tr1.tile(i).start(), a[i]);
    BOOST_CHECK_EQUAL(tr1.tile(i).finish(), a[i + 1]);
    BOOST_CHECK_EQUAL(tr1.tile(i).size(), a[i + 1] - a[i]);
    BOOST_CHECK_EQUAL(tr1.tile(i).volume(), a[i + 1] - a[i]);
    BOOST_CHECK_EQUAL(tr1.tile(i).weight(), 1ul);
  }
}

BOOST_AUTO_TEST_CASE( constructor )
{
  // check default construction and range info.
  {
    BOOST_REQUIRE_NO_THROW(range1_type r);
    range1_type r;
    BOOST_CHECK_EQUAL(r.tiles(), range1_type::range_type(0,0));
    BOOST_CHECK_EQUAL(r.elements(), range1_type::tile_range_type(0,0));
#ifdef TA_EXCEPTION_ERROR
    BOOST_CHECK_THROW(r.tile(0), Exception);
#endif
  }

  // check construction with a iterators and the range info.
  {
    BOOST_REQUIRE_NO_THROW(range1_type r(a.begin(), a.end()));
    range1_type r(a.begin(), a.end());
    BOOST_CHECK_EQUAL(r.tiles(), tiles);
    BOOST_CHECK_EQUAL(r.elements(), elements);
    for(std::size_t i = 0; i < a.size() - 1; ++i) {
      BOOST_CHECK_EQUAL(r.tile(i).start(), a[i]);
      BOOST_CHECK_EQUAL(r.tile(i).finish(), a[i + 1]);
      BOOST_CHECK_EQUAL(r.tile(i).size(), a[i + 1] - a[i]);
      BOOST_CHECK_EQUAL(r.tile(i).volume(), a[i + 1] - a[i]);
      BOOST_CHECK_EQUAL(r.tile(i).weight(), 1ul);
    }
  }


  // check copy constructor
  {
    BOOST_REQUIRE_NO_THROW(range1_type r(tr1));
    range1_type r(tr1);
    BOOST_CHECK_EQUAL(r.tiles(), tiles);
    BOOST_CHECK_EQUAL(r.elements(), elements);
    for(std::size_t i = 0; i < a.size() - 1; ++i) {
      BOOST_CHECK_EQUAL(r.tile(i).start(), a[i]);
      BOOST_CHECK_EQUAL(r.tile(i).finish(), a[i + 1]);
      BOOST_CHECK_EQUAL(r.tile(i).size(), a[i + 1] - a[i]);
      BOOST_CHECK_EQUAL(r.tile(i).volume(), a[i + 1] - a[i]);
      BOOST_CHECK_EQUAL(r.tile(i).weight(), 1ul);
    }
  }

  // check construction with a with a tile offset.
  {
    BOOST_REQUIRE_NO_THROW(range1_type r(a.begin(), a.end(), 2));
    range1_type r(a.begin(), a.end(), 2);
    BOOST_CHECK_EQUAL(r.tiles(), range1_type::range_type(2, 1 + a.size()));
    BOOST_CHECK_EQUAL(r.elements(), elements);
    for(std::size_t i = 0; i < a.size() - 1; ++i) {
      BOOST_CHECK_EQUAL(r.tile(i + 2).start(), a[i]);
      BOOST_CHECK_EQUAL(r.tile(i + 2).finish(), a[i + 1]);
      BOOST_CHECK_EQUAL(r.tile(i + 2).size(), a[i + 1] - a[i]);
      BOOST_CHECK_EQUAL(r.tile(i + 2).volume(), a[i + 1] - a[i]);
      BOOST_CHECK_EQUAL(r.tile(i + 2).weight(), 1ul);
    }
  }


  // check construction with a with a element offset.
  {
    BOOST_REQUIRE_NO_THROW(range1_type r(a.begin() + 1, a.end()));
    range1_type r(a.begin() + 1, a.end());
    BOOST_CHECK_EQUAL(r.tiles(), range1_type::range_type(0,a.size() - 2));
    BOOST_CHECK_EQUAL(r.elements(), range1_type::tile_range_type(a[1],a.back()));
    for(std::size_t i = 1; i < a.size() - 1; ++i) {
      BOOST_CHECK_EQUAL(r.tile(i - 1).start(), a[i]);
      BOOST_CHECK_EQUAL(r.tile(i - 1).finish(), a[i + 1]);
      BOOST_CHECK_EQUAL(r.tile(i - 1).size(), a[i + 1] - a[i]);
      BOOST_CHECK_EQUAL(r.tile(i - 1).volume(), a[i + 1] - a[i]);
      BOOST_CHECK_EQUAL(r.tile(i - 1).weight(), 1ul);
    }
  }
}

BOOST_AUTO_TEST_CASE( ostream )
{
  std::stringstream stm;
  stm << "( tiles = " << range1_type::range_type(0, a.size() - 1) <<
      ", elements = " << range1_type::tile_range_type(a.front(), a.back()) << " )";

  boost::test_tools::output_test_stream output;
  output << tr1;
  BOOST_CHECK( !output.is_empty( false ) );
  BOOST_CHECK( output.check_length( stm.str().size(), false ) );
  BOOST_CHECK( output.is_equal( stm.str().c_str() ) );
}

BOOST_AUTO_TEST_CASE( element2tile )
{
  // construct a map that should match the element to tile map for tr1.
  std::vector<std::size_t> e;
  for(range1_type::range_type::index t = tr1.tiles().start(); t < tr1.tiles().finish(); ++t)
    for(ordinal_index i = tr1.tile(t).start(); i < tr1.tile(t).finish(); ++i)
      e.push_back(t);

  // Construct a map that matches the internal element to tile map for tr1.
  std::vector<std::size_t> c;
  for(std::size_t i = tr1.elements().start(); i < tr1.elements().finish(); ++i)
    c.push_back(tr1.element2tile(i));

  // Check that the expected and internal element to tile maps match.
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
  std::array<std::size_t, 6> a1 = a;
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
  // check for proper iteration functionality.
  std::size_t count = 0;
  for(range1_type::const_iterator it = tr1.begin(); it != tr1.end(); ++it, ++count) {
    BOOST_CHECK_EQUAL(*it, range1_type::tile_range_type(a[count], a[count + 1]));
  }
  BOOST_CHECK_EQUAL(count, tr1.tiles().volume());
}

BOOST_AUTO_TEST_CASE( find )
{
  // check that find returns an iterator to the correct tile.
  BOOST_CHECK_EQUAL( * tr1.find(tr1.tile(3).start() + 1), range1_type::tile_range_type(a[3], a[4]));

  // check that the iterator points to the end() iterator if the element is out of range.
  BOOST_CHECK( tr1.find(a.back() + 10) == tr1.end());

}

BOOST_AUTO_TEST_CASE( assignment )
{
  range1_type r1;
  BOOST_CHECK_NE( r1, tr1);
  BOOST_CHECK_EQUAL((r1 = tr1), tr1); // check operator=
  BOOST_CHECK_EQUAL(r1, tr1);
}

BOOST_AUTO_TEST_SUITE_END()
