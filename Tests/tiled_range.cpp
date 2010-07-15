#include "TiledArray/tiled_range.h"
#include "TiledArray/permutation.h"
#include "unit_test_config.h"
#include "range_fixture.h"

#include "TiledArray/coordinates.h"

using namespace TiledArray;
/*
struct TiledRangeFixture {
  typedef TiledRange<GlobalFixture::coordinate_system> TRangeN;
  typedef TRangeN::index index;

  TiledRangeFixture() {
    d[0] = 0; d[1] = 10; d[2] = 20; d[3] = 30;

    start000 = TRangeN::tile_index::make(d0[0], d1[0], d2[0]);
    tile_start = TRangeN::index::make(0,0,0);
    element_start = TRangeN::tile_index::make(d0[0], d1[0], d2[0]);
    finish000 = TRangeN::tile_index::make(d0[1], d1[1], d2[1]);
    tile_finish = TRangeN::index::make(d0.size() - 1, d1.size() - 1, d2.size() - 1);
    element_finish = TRangeN::tile_index::make(d0[d0.size() - 1], d1[d1.size() - 1], d2[d2.size() - 1]);
    size000 = finish000.data();
    tile_size = tile_finish.data();
    element_size = element_finish.data();

    tile_range.resize(tile_start, tile_finish);
    element_range.resize(element_start, element_finish);
    tile000_range.resize(start000, finish000);

    vol000 = tile000_range.volume();
    tile_vol = tile_range.volume();
    element_vol = element_range.volume();

    tr.resize(dims.begin(), dims.end());
  }
  ~TiledRangeFixture() { }

  boost::array<std::size_t, 4> d0;
  boost::array<std::size_t, 5> d1;
  boost::array<std::size_t, 6> d2;
  boost::array<TRangeN::tiled_range1_type, 3> dims;
  TRangeN tr;

  TRangeN::tile_index start000;
  TRangeN::index tile_start;
  TRangeN::tile_index element_start;
  TRangeN::tile_index finish000;
  TRangeN::index tile_finish;
  TRangeN::tile_index element_finish;
  TRangeN::size_array size000;
  TRangeN::size_array tile_size;
  TRangeN::size_array element_size;
  TRangeN::volume_type vol000;
  TRangeN::volume_type tile_vol;
  TRangeN::volume_type element_vol;
  TRangeN::range_type tile_range;
  TRangeN::element_range_type element_range;
  TRangeN::tile_range_type tile000_range;

};
*/


const TiledRangeFixture::TRangeN::range_type
TiledRangeFixture::tile_range(RangeFixture::fill_index<TiledRangeFixture::index>(0),
    RangeFixture::fill_index<TiledRangeFixture::index>(5));
const TiledRangeFixture::TRangeN::tile_range_type
TiledRangeFixture::element_range(RangeFixture::fill_index<TiledRangeFixture::tile_index>(0),
    RangeFixture::fill_index<TiledRangeFixture::tile_index>(a[5]));

BOOST_FIXTURE_TEST_SUITE( tiled_range_suite, TiledRangeFixture )

BOOST_AUTO_TEST_CASE( accessor )
{
  BOOST_CHECK_EQUAL(tr.tiles(), tile_range);
  BOOST_CHECK_EQUAL(tr.elements(), element_range);
}

BOOST_AUTO_TEST_CASE( constructor )
{
  // check default constructor
  BOOST_REQUIRE_NO_THROW(TRangeN r0);
  TRangeN r0;
  TRangeN::size_array s0 = {{0,0,0}};
  BOOST_CHECK_EQUAL(r0.tiles().size(), s0);
  BOOST_CHECK_EQUAL(r0.elements().size(), s0);

  // check ranges constructor
  BOOST_REQUIRE_NO_THROW(TRangeN r1(dims.begin(), dims.end()));
  TRangeN r1(dims.begin(), dims.end());
  BOOST_CHECK_EQUAL(r1.tiles(), tile_range);
  BOOST_CHECK_EQUAL(r1.elements(), element_range);

  std::vector<TRangeN::tiled_range1_type> dims2;
  for(std::size_t i = 0; i < GlobalFixture::coordinate_system::dim; ++i)
    dims2.push_back(TRangeN::tiled_range1_type(a.begin(), a.end(), 1));
  TRangeN::range_type t2(p1, p6);

  // check ranges constructor w/ offset tile origin.
  BOOST_REQUIRE_NO_THROW(TRangeN r2(dims2.begin(), dims2.end()));
  TRangeN r2(dims2.begin(), dims2.end());
  BOOST_CHECK_EQUAL(r2.tiles(), t2);
  BOOST_CHECK_EQUAL(r2.elements(), element_range);

  boost::array<std::size_t, 6> a3;
  std::copy(GlobalFixture::primes.begin(), GlobalFixture::primes.begin() + 6, a3.begin());
  std::vector<TRangeN::tiled_range1_type> dims3(GlobalFixture::coordinate_system::dim, TRangeN::tiled_range1_type(a3.begin(), a3.end()));
  TRangeN::tile_range_type e3(fill_index<tile_index>(a3[0]), fill_index<tile_index>(a3[5]));

  // check ranges constructor w/ offset element origin.
  BOOST_REQUIRE_NO_THROW(TRangeN r3(dims3.begin(), dims3.end()));
  TRangeN r3(dims3.begin(), dims3.end());
  BOOST_CHECK_EQUAL(r3.tiles(), tile_range);
  BOOST_CHECK_EQUAL(r3.elements(), e3);

  // check copy constructor
  BOOST_REQUIRE_NO_THROW(TRangeN r4(tr));
  TRangeN r4(tr);
  BOOST_CHECK_EQUAL(r4.tiles(), tr.tiles());
  BOOST_CHECK_EQUAL(r4.elements(), tr.elements());

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(TRangeN r5(dims3.begin(), dims3.end() - 1), std::runtime_error);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( ostream )
{
  boost::test_tools::output_test_stream output;
  output << tr;
  BOOST_CHECK( !output.is_empty( false ) );
  BOOST_CHECK( output.check_length( 75, false ) );
  BOOST_CHECK( output.is_equal( "( tiles = [ (0, 0, 0), (3, 4, 5) ) elements = [ (0, 0, 0), (30, 20, 15) ) )" ) );
}

BOOST_AUTO_TEST_CASE( comparison ) {
  TRangeN r1(tr);

  // check equality operator for identical ranges
  BOOST_CHECK(r1 == tr);
  // check that the inequality operator for identical ranges
  BOOST_CHECK(! (r1 != tr));

  std::vector<TRangeN::tiled_range1_type> dims2;
  for(std::size_t i = 0; i < GlobalFixture::coordinate_system::dim; ++i)
    dims2.push_back(TRangeN::tiled_range1_type(a.begin(), a.end(), 1));
  TRangeN r2(dims2.begin(), dims2.end());

  // comparison w/ offset tile origin.
  BOOST_CHECK(! (r2 == tr));
  BOOST_CHECK(r2 != tr);

  boost::array<std::size_t, 6> a3;
  std::copy(GlobalFixture::primes.begin(), GlobalFixture::primes.begin() + 6, a3.begin());
  std::vector<TRangeN::tiled_range1_type> dims3(GlobalFixture::coordinate_system::dim,
      TRangeN::tiled_range1_type(a3.begin(), a3.end()));

  TRangeN r3(dims3.begin(), dims3.end());

  // comparison operators w/ offset elements and different tiling
  BOOST_CHECK(! (r3 == tr));
  BOOST_CHECK(r3 != tr);
}

BOOST_AUTO_TEST_CASE( assignment )
{
  TRangeN r1;

  // verify they are not equal before assignment.
  BOOST_CHECK_NE(r1, tr);

  // check that assignment returns itself.
  BOOST_CHECK_EQUAL((r1 = tr), tr);

  // check that assignment is valid.
  BOOST_CHECK_EQUAL(r1, tr);
}

BOOST_AUTO_TEST_CASE( permutation )
{
  Permutation<3> p(2,0,1);
  TRangeN r1 = p ^ tr;
  BOOST_CHECK_EQUAL(r1.tiles(), p ^ tr.tiles()); // check that tile data was permuted properly.
  BOOST_CHECK_EQUAL(r1.elements(), p ^ tr.elements()); // check that element data was permuted properly.

  TRangeN r2(tr);
  BOOST_CHECK_EQUAL((r2 ^= p), r1); // check that permutation returns itself.
  BOOST_CHECK_EQUAL(r2, r1);// check that the permutation was assigned correctly.
}

BOOST_AUTO_TEST_CASE( make_tile_range )
{
  tile_index start;
  tile_index finish;

  // iterate over all the tile indexes in the tiled range.
  TRangeN::ordinal_index i = 0;
  for(RangeN::const_iterator it = r.begin(); it != r.end(); ++it, ++i) {
    // get the start and finish indexes of the current range.
    for(unsigned int d = 0; d < GlobalFixture::coordinate_system::dim; ++d) {
      start[d] = a[ (*it)[d] ];
      finish[d] = a[ (*it)[d] ];
    }

    // construct a range object that should match the range constructed by TiledRange.
    TRangeN::tile_range_type range(start, finish);

    // Get the two ranges to be tested.
    TRangeN::tile_range_type range_index = tr.make_tile_range(*it);
    TRangeN::tile_range_type range_ordinal = tr.make_tile_range(i);

    BOOST_CHECK_EQUAL(range_index, range);
    BOOST_CHECK_EQUAL(range_ordinal, range);
  }
}

BOOST_AUTO_TEST_SUITE_END()

