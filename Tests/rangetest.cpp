#include <range.h>
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>
#include "iterationtest.h"

using namespace TiledArray;
/*
struct RangeFixture {
  typedef Range<std::size_t, 3> Range3;
  typedef Range3::index_type index_type;

  RangeFixture() {
    d0[0] = 0; d0[1] = 10; d0[2] = 20; d0[3] = 30;
    d1[0] = 0; d1[1] = 5; d1[2] = 10; d1[3] = 15; d1[4] = 20;
    d2[0] = 0; d2[1] = 3; d2[2] = 6; d2[3] = 9; d2[4] = 12; d2[5] = 15;
    dims[0] = Range3::range1_type(d0);
    dims[1] = Range3::range1_type(d1);
    dims[2] = Range3::range1_type(d2);

    r.resize(dims.begin(), dims.end());
  }
  ~RangeFixture() { }

  boost::array<std::size_t, 4> d0;
  boost::array<std::size_t, 5> d1;
  boost::array<std::size_t, 6> d2;
  boost::array<Range3::range1_type, 3 > dims;
  Range3 r;
};


BOOST_FIXTURE_TEST_SUITE( range_suite, RangeFixture )

BOOST_AUTO_TEST_CASE( constructor )
{

}

BOOST_AUTO_TEST_SUITE_END()
*/

/*
void
RangeTest()
{
  typedef Range<std::size_t, 4> Range4;
  typedef Range4::tile_index_type::index eindex;
  typedef Range4::index_type::index tindex;

  std::cout << "Range Tests:" << std::endl;

  std::cout << "Constructor tests: ";
  // Test default constructor.
  Range<std::size_t, 4> ortho1;

  // Test with C-style Range Array constructor.
  eindex dim0[] = {0,10,20,30};
  eindex dim1[] = {0,5,10,15,20};
  eindex dim2[] = {0,3,6,9,12,15};
  eindex dim3[] = {0,2,4,6,8,10,12};
  tindex tiles[4] = {4, 5, 6, 7};

  Range4::range1_type rng_set[4] = {Range4::range1_type(dim0, dim0 + tiles[0]),
      Range4::range1_type(dim1, dim1 + tiles[1]),
      Range4::range1_type(dim2, dim2 + tiles[2]),
      Range4::range1_type(dim3, dim3 + tiles[3])};

  std::vector<Range4::range1_type> rng_vector(rng_set, rng_set + 4);

  Range4 ortho2(rng_set, rng_set + 4);
  Range4 ortho3(rng_vector.begin(),rng_vector.end());

  std::cout << "PASSED" << std::endl;

  std::cout << "Ranges: ";
  std::cout << "ortho1 = " << ortho1 << std::endl;
  std::cout << "ortho2 = " << ortho2 << std::endl;
  std::cout << "ortho3 = " << ortho3 << std::endl;

  std::cout << "comparison tests: " << std::endl;
  std::cout << "ortho1 == ortho1: " << (ortho1==ortho1 ? "true" : "false") << std::endl;
  std::cout << "ortho1 == ortho3: " << (ortho1==ortho3 ? "true" : "false") << std::endl;
  std::cout << "ortho2 == ortho3: " << (ortho2==ortho3 ? "true" : "false") << std::endl;
  std::cout << "ortho1 != ortho1: " << (ortho1!=ortho1 ? "true" : "false") << std::endl;
  std::cout << "ortho1 != ortho3: " << (ortho1!=ortho3 ? "true" : "false") << std::endl;
  std::cout << "ortho2 != ortho3: " << (ortho2!=ortho3 ? "true" : "false") << std::endl;

  std::cout << "ordinal tests: " << std::endl;
  typedef Range<std::size_t, 4>::tile_index_type tile_index_type;
  typedef Range<std::size_t, 4>::index_type index_type;
  index_type t1(0, 2, 4, 1);
  tile_index_type e1(1, 3, 3, 2);
  tile_index_type e2(29, 19, 14, 12);
  std::cout << "tile_index: t1=" << t1 << std::endl;
  std::cout << "ortho2.includes(t1) = " << ortho2.tiles().includes(t1) << std::endl;
  std::cout << "element_index: e1=" << e1 << std::endl;
  std::cout << "ortho2.includes(e1) = " << ortho2.elements().includes(e1) << std::endl;
  std::cout << "ortho2.find(e1) = " << * ortho2.find(e1) << std::endl;
  std::cout << "element_index: e2=" << e2 << std::endl;
  std::cout << "ortho2.includes(e2) = " << ortho2.elements().includes(e2) << std::endl;
  std::cout << "ortho2.find(e2) = " << * ortho2.find(e2) << std::endl;

  std::cout << "tile iterator tests: " << std::endl;
  for(Range<std::size_t, 4>::block_type::const_iterator t=ortho2.tiles().begin(); t!=ortho2.tiles().end(); ++t) {
    std::cout << "t = " << *t << std::endl;
  }

  std::cout << "End Range Tests" << std::endl << std::endl;
}
*/
