#include "TiledArray/tiled_range.h"
#include "TiledArray/permutation.h"
#include "unit_test_config.h"
#include "range_fixture.h"

#include "TiledArray/coordinates.h"

using namespace TiledArray;

const TiledRangeFixture::TRangeN::range_type
TiledRangeFixture::tile_range(fill_index<TiledRangeFixture::index>(0),
    fill_index<TiledRangeFixture::index>(5));
const TiledRangeFixture::TRangeN::tile_range_type
TiledRangeFixture::element_range(fill_index<TiledRangeFixture::tile_index>(0),
    fill_index<TiledRangeFixture::tile_index>(a[5]));

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

  std::stringstream stm;
  stm << "( tiles = " << TRangeN::range_type(tr.tiles().start(), tr.tiles().finish()) <<
      ", elements = " << TRangeN::tile_range_type(tr.elements().start(), tr.elements().finish()) << " )";

  boost::test_tools::output_test_stream output;
  output << tr;
  BOOST_CHECK( !output.is_empty( false ) );
  BOOST_CHECK( output.check_length( stm.str().size(), false ) );
  BOOST_CHECK( output.is_equal( stm.str().c_str() ) );
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
  for(RangeN::const_iterator it = tr.tiles().begin(); it != tr.tiles().end(); ++it, ++i) {
    // get the start and finish indexes of the current range.
    for(unsigned int d = 0; d < GlobalFixture::coordinate_system::dim; ++d) {
      start[d] = a[ (*it)[d] ];
      finish[d] = a[ (*it)[d] + 1 ];
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

