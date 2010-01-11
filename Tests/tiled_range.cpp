#include "TiledArray/tiled_range.h"
#include "TiledArray/permutation.h"
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>
#include "iteration_test.h"

using namespace TiledArray;

struct TiledRangeFixture {
  typedef TiledRange<std::size_t, 3> TRange3;
  typedef TRange3::index_type index_type;

  TiledRangeFixture() {
    d0[0] = 0; d0[1] = 10; d0[2] = 20; d0[3] = 30;
    d1[0] = 0; d1[1] = 5; d1[2] = 10; d1[3] = 15; d1[4] = 20;
    d2[0] = 0; d2[1] = 3; d2[2] = 6; d2[3] = 9; d2[4] = 12; d2[5] = 15;
    dims[0] = TRange3::tiled_range1_type(d0.begin(), d0.end());
    dims[1] = TRange3::tiled_range1_type(d1.begin(), d1.end());
    dims[2] = TRange3::tiled_range1_type(d2.begin(), d2.end());

    start000 = TRange3::tile_index_type::make(d0[0], d1[0], d2[0]);
    tile_start = TRange3::index_type::make(0,0,0);
    element_start = TRange3::tile_index_type::make(d0[0], d1[0], d2[0]);
    finish000 = TRange3::tile_index_type::make(d0[1], d1[1], d2[1]);
    tile_finish = TRange3::index_type::make(d0.size() - 1, d1.size() - 1, d2.size() - 1);
    element_finish = TRange3::tile_index_type::make(d0[d0.size() - 1], d1[d1.size() - 1], d2[d2.size() - 1]);
    size000 = finish000.data();
    tile_size = tile_finish.data();
    element_size = element_finish.data();

    tile_range.resize(tile_start, tile_finish);
    element_range.resize(element_start, element_finish);
    tile000_range.resize(start000, finish000);

    vol000 = tile000_range.volume();
    tile_vol = tile_range.volume();
    element_vol = element_range.volume();

    r.resize(dims.begin(), dims.end());
  }
  ~TiledRangeFixture() { }

  boost::array<std::size_t, 4> d0;
  boost::array<std::size_t, 5> d1;
  boost::array<std::size_t, 6> d2;
  boost::array<TRange3::tiled_range1_type, 3> dims;
  TRange3 r;

  TRange3::tile_index_type start000;
  TRange3::index_type tile_start;
  TRange3::tile_index_type element_start;
  TRange3::tile_index_type finish000;
  TRange3::index_type tile_finish;
  TRange3::tile_index_type element_finish;
  TRange3::size_array size000;
  TRange3::size_array tile_size;
  TRange3::size_array element_size;
  TRange3::volume_type vol000;
  TRange3::volume_type tile_vol;
  TRange3::volume_type element_vol;
  TRange3::range_type tile_range;
  TRange3::element_range_type element_range;
  TRange3::tile_range_type tile000_range;

};


BOOST_FIXTURE_TEST_SUITE( tiled_range_suite, TiledRangeFixture )

BOOST_AUTO_TEST_CASE( accessor )
{
  BOOST_CHECK_EQUAL(r.tiles(), tile_range);
  BOOST_CHECK_EQUAL(r.elements(), element_range);
  BOOST_CHECK_EQUAL(r.tile(tile_start), tile000_range);
  BOOST_CHECK_EQUAL(r.tile(0ul), tile000_range);
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(r.tile(tile_finish), std::out_of_range);
  BOOST_CHECK_THROW(r.tile(60ul), std::out_of_range);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(TRange3 r0); // check default constructor
  TRange3 r0;
  TRange3::size_array s0 = {{0,0,0}};
  BOOST_CHECK_EQUAL(r0.tiles().size(), s0);
  BOOST_CHECK_EQUAL(r0.elements().size(), s0);
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(r0.tile(tile_start), std::out_of_range);
#endif // TA_EXCEPTION_ERROR

  BOOST_REQUIRE_NO_THROW(TRange3 r1(dims.begin(), dims.end())); // check ranges constructor
  TRange3 r1(dims.begin(), dims.end());
  BOOST_CHECK_EQUAL(r1.tiles(), tile_range);
  BOOST_CHECK_EQUAL(r1.elements(), element_range);
  BOOST_CHECK_EQUAL(r1.tile(tile_start), tile000_range);

  boost::array<TRange3::tiled_range1_type, 3> dims2 =
      {{ TRange3::tiled_range1_type(d0.begin(), d0.end(), 1),
      TRange3::tiled_range1_type(d1.begin(), d1.end(), 2),
      TRange3::tiled_range1_type(d2.begin(), d2.end(), 3) }};
  TRange3::range_type t2(TRange3::index_type(1,2,3), TRange3::index_type(1,2,3) + tile_size);

  BOOST_REQUIRE_NO_THROW(TRange3 r2(dims2.begin(), dims2.end())); // check ranges constructor
  TRange3 r2(dims2.begin(), dims2.end());                         // w/ offset tile origin.
  BOOST_CHECK_EQUAL(r2.tiles(), t2);
  BOOST_CHECK_EQUAL(r2.elements(), element_range);
  BOOST_CHECK_EQUAL(r2.tile(t2.start()), tile000_range);

  boost::array<TRange3::tiled_range1_type, 3> dims3 =
      {{ TRange3::tiled_range1_type(d0.begin() + 1 , d0.end()),
      TRange3::tiled_range1_type(d1.begin() + 1, d1.end()),
      TRange3::tiled_range1_type(d2.begin() + 1, d2.end()) }};
  TRange3::tile_range_type t3(TRange3::tile_index_type(d0[1], d1[1], d2[1]),
      TRange3::tile_index_type(d0[2], d1[2], d2[2]));
  TRange3::range_type tr3(TRange3::index_type(0, 0, 0), TRange3::index_type(2, 3, 4));
  TRange3::element_range_type er3(TRange3::element_index_type(10, 5, 3), TRange3::element_index_type(30, 20, 15));

  BOOST_REQUIRE_NO_THROW(TRange3 r3(dims3.begin(), dims3.end())); // check ranges constructor
  TRange3 r3(dims3.begin(), dims3.end());                         // w/ offset element origin.
  BOOST_CHECK_EQUAL(r3.tiles(), tr3);
  BOOST_CHECK_EQUAL(r3.elements(), er3);
  BOOST_CHECK_EQUAL(r3.tile(tile_start), t3);

  BOOST_REQUIRE_NO_THROW(TRange3 r4(r)); // check copy constructor
  TRange3 r4(r);
  BOOST_CHECK_EQUAL(r4.tiles(), r.tiles());
  BOOST_CHECK_EQUAL(r4.elements(), r.elements());
  BOOST_CHECK_EQUAL(r4.tile(tile_start), r.tile(tile_start));

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(TRange3 r5(dims3.begin(), dims3.end() - 1), std::runtime_error);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( iteration ) {
  TRange3::tile_index_type s;
  TRange3::tile_index_type f;
  TRange3::tile_range_type tr;
  TRange3::range_type::const_iterator t_it = r.tiles().begin();
  for(TRange3::const_iterator it = r.begin(); it != r.end(); ++it, ++t_it) {
    // check that the basic iteration functionality works, that is begin,  end, and increment.
    for(unsigned int d = 0; d < 3; ++d) {
      s[d] = dims[d].tile((*t_it)[d]).start()[0];
      f[d] = dims[d].tile((*t_it)[d]).finish()[0];
    }

    tr.resize(s, f);
    BOOST_CHECK_EQUAL(*it, tr); // check that the tile range has the expected dimensions.
        // check that it iterates in the expected order.
        // Check dereference.

  }
}

BOOST_AUTO_TEST_CASE( find )
{
  TRange3::tile_index_type s(10,5,3);
  TRange3::tile_index_type f(20,10,6);
  TRange3::tile_index_type t(12,7,4);
  TRange3::tile_range_type tr(s, f);

  BOOST_CHECK_EQUAL(* r.find(t), tr); // check that the correct tile is found.
  BOOST_CHECK_EQUAL(r.find(element_finish), r.end()); // check that the iterator
                       // points to the end when the tile index is out of range.
}

BOOST_AUTO_TEST_CASE( ostream )
{
  boost::test_tools::output_test_stream output;
  output << r;
  BOOST_CHECK( !output.is_empty( false ) );
  BOOST_CHECK( output.check_length( 75, false ) );
  BOOST_CHECK( output.is_equal( "( tiles = [ (0, 0, 0), (3, 4, 5) ) elements = [ (0, 0, 0), (30, 20, 15) ) )" ) );
}

BOOST_AUTO_TEST_CASE( comparison ) {
  TRange3 r1(r);
  BOOST_CHECK(r1 == r);     // check equality operator
  BOOST_CHECK(! (r1 != r)); // check that the inequality operator

  boost::array<TRange3::tiled_range1_type, 3> dims2 =
      {{ TRange3::tiled_range1_type(d0.begin(), d0.end(), 1),
      TRange3::tiled_range1_type(d1.begin(), d1.end(), 2),
      TRange3::tiled_range1_type(d2.begin(), d2.end(), 3) }};
  TRange3 r2(dims2.begin(), dims2.end());

  BOOST_CHECK(! (r2 == r));                 // comparison w/ offset tile origin.
  BOOST_CHECK(r2 != r);

  boost::array<TRange3::tiled_range1_type, 3> dims3 =
      {{ TRange3::tiled_range1_type(d0.begin() + 1 , d0.end()),
      TRange3::tiled_range1_type(d1.begin() + 1, d1.end()),
      TRange3::tiled_range1_type(d2.begin() + 1, d2.end()) }};
  TRange3 r3(dims3.begin(), dims3.end());

  BOOST_CHECK(! (r3 == r)); // comparison w/ offset elements and different tiling
  BOOST_CHECK(r3 != r);
}

BOOST_AUTO_TEST_CASE( assignment )
{
  TRange3 r1;
  BOOST_CHECK_NE(r1, r);          // verify they are not equal before assignment.
  BOOST_CHECK_EQUAL((r1 = r), r); // check that assignment returns itself.
  BOOST_CHECK_EQUAL(r1, r);       // check that assignment is valid.
}

BOOST_AUTO_TEST_CASE( resize )
{
  TRange3 r1;
  BOOST_CHECK_NE(r1, r);          // verify they are not equal before resize.
  BOOST_CHECK_EQUAL(r1.resize(dims.begin(), dims.end()), r); // check that resize() returns the object.
  BOOST_CHECK_EQUAL(r1, r);       // check that resize was done correctly.
}

BOOST_AUTO_TEST_CASE( permutation )
{
  Permutation<3> p(2,0,1);
  TRange3 r1 = p ^ r;
  BOOST_CHECK_EQUAL(r1.tiles(), p ^ r.tiles()); // check that tile data was permuted properly.
  BOOST_CHECK_EQUAL(r1.elements(), p ^ r.elements()); // check that element data was permuted properly.
  for(TRange3::range_type::const_iterator it = r.tiles().begin(); it != r.tiles().end(); ++it) {
    BOOST_CHECK_EQUAL(r1.tile(p ^ *it), p ^ r.tile(*it)); // check that tiles and tile data was permuted correctly.
  }

  TRange3 r2(r);
  BOOST_CHECK_EQUAL((r2 ^= p), r1); // check that permutation returns itself.
  BOOST_CHECK_EQUAL(r2, r1);// check that the permutation was assigned correctly.
}

BOOST_AUTO_TEST_SUITE_END()

