#include "array.h"
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>
#include "../madness_fixture.h"

using namespace TiledArray;

template <typename A>
struct TiledRangeFixture {
  typedef typename A::tiled_range_type TRange3;
  typedef typename TRange3::tiled_range1_type TRange1;

  TiledRangeFixture() {
    const std::size_t d0[] = {0, 10, 20, 30};
    const std::size_t d1[] = {0, 5, 10, 15, 20};
    const std::size_t d2[] = {0, 3, 6, 9, 12, 15};
    const TRange1 dim0(d0, d0 + 4);
    const TRange1 dim1(d1, d1 + 5);
    const TRange1 dim2(d2, d2 + 6);
    const TRange1 dims[3] = {dim0, dim1, dim2};
    trng.resize(dims, dims+3);
  }

  ~TiledRangeFixture() { }

  TRange3 trng;
}; // struct TiledRangeFixture

struct ArrayFixture : public TiledRangeFixture<Array<int, 3> > {
  typedef Array<int, 3> Array3;

  ArrayFixture() : world(*MadnessFixture::world), a(world, trng) {

  }

  ~ArrayFixture() { }

  madness::World& world;
  Array<int, 3> a;

}; // struct ArrayFixture


BOOST_FIXTURE_TEST_SUITE( array_suite , ArrayFixture )

BOOST_AUTO_TEST_CASE( find )
{

}

BOOST_AUTO_TEST_SUITE_END()
