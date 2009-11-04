#include "array.h"
#include "utility.h"
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>
#include <boost/functional.hpp>
#include <algorithm>
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
    trng.resize(dims, dims + 3);
  }

  ~TiledRangeFixture() { }

  TRange3 trng;
}; // struct TiledRangeFixture

struct ArrayFixture : public TiledRangeFixture<Array<int, 3> > {
  typedef Array<int, 3> Array3;
  typedef Array3::index_type index_type;
  typedef Array3::tile_index_type tile_index_type;
  typedef Array3::tile_type tile_type;
  typedef std::vector<std::pair<index_type, tile_type> > data_array;

  ArrayFixture() : world(MadnessFixture::world), a(*world, trng), ca(a) {
    int v = 0;
    for(TRange3::range_type::const_iterator it = a.tiles().begin(); it != a.tiles().end(); ++it) {
      a.insert(*it, v);
      data.push_back(Array3::value_type(*it, tile_type(a.tile(*it), v++)));
    }
    world->gop.fence();
  }

  ~ArrayFixture() { }

  // Sum the first elements of each tile on all nodes.
  double sum_first(const Array3& a) {
    double sum = 0.0;
    for(Array3::const_iterator it = a.begin(); it != a.end(); ++it)
      sum += it->second.at(0);

    world->mpi.comm().Allreduce(MPI_IN_PLACE, &sum, 1, MPI::DOUBLE, MPI::SUM);

    return sum;
  }

  // Count the number of tiles in the array on all nodes.
  std::size_t tile_count(const Array3& a) {
    int n = static_cast<int>(a.volume(true));
    world->mpi.comm().Allreduce(MPI_IN_PLACE, &n, 1, MPI::INT, MPI::SUM);

    return n;
  }

  data_array::const_iterator find_data_element(const index_type& i) const {
    return std::find_if(data.begin(), data.end(),
            detail::make_unary_transform(boost::bind1st(std::equal_to<Array3::index_type>(), i),
            detail::pair_first<Array3::value_type>()));
  }

  madness::World* world;
  Array3 a;
  const Array3& ca;
  data_array data;

}; // struct ArrayFixture

BOOST_FIXTURE_TEST_SUITE( array_suite , ArrayFixture )

BOOST_AUTO_TEST_CASE( array_dims )
{
  BOOST_CHECK_EQUAL(a.start(), trng.tiles().start());
  BOOST_CHECK_EQUAL(a.finish(), trng.tiles().finish());
  BOOST_CHECK_EQUAL(a.size(), trng.tiles().size());
//  BOOST_CHECK_EQUAL(a.weight(), trng.tiles().weight());
  BOOST_CHECK_EQUAL(a.volume(), trng.tiles().volume());
  BOOST_CHECK_EQUAL(a.tiles(), trng.tiles());
  BOOST_CHECK_EQUAL(a.elements(), trng.elements());
  for(TRange3::range_type::const_iterator it = trng.tiles().begin(); it != trng.tiles().end(); ++it) {
    BOOST_CHECK_EQUAL(a.tile(*it), trng.tile(*it));
  }
  BOOST_CHECK(a.includes(trng.tiles().start()));
  BOOST_CHECK(a.includes(index_type(2, 3, 4)));
  BOOST_CHECK(!a.includes(trng.tiles().finish()));
}

BOOST_AUTO_TEST_CASE( iterator )
{
  for(Array3::iterator it = a.begin(); it != a.end(); ++it) { // check non-const iterator functionality
    data_array::const_iterator d_it = find_data_element(it->first);
    BOOST_CHECK(d_it != data.end());
    BOOST_CHECK_EQUAL(it->second.at(0), d_it->second.at(0));
  }

  for(Array3::const_iterator it = a.begin(); it != a.end(); ++it) { // check const/non-const iterator functionality
    data_array::const_iterator d_it = find_data_element(it->first);
    BOOST_CHECK(d_it != data.end());
    BOOST_CHECK_EQUAL(it->second.at(0), d_it->second.at(0));
  }

  for(Array3::const_iterator it = ca.begin(); it != ca.end(); ++it) { // check const iterator functionality
    data_array::const_iterator d_it = find_data_element(it->first);
    BOOST_CHECK(d_it != data.end());
    BOOST_CHECK_EQUAL(it->second.at(0), d_it->second.at(0));
  }
}

BOOST_AUTO_TEST_CASE( accessors )
{
  Array3::accessor acc;
  Array3::const_accessor const_acc;
  data_array::const_iterator d_it;
  for(TRange3::range_type::const_iterator it = trng.tiles().begin(); it != trng.tiles().end(); ++it) {
    if(a.is_local(*it)) {
      data_array::const_iterator d_it = find_data_element(*it);
      BOOST_CHECK(a.find(acc,*it));
      BOOST_CHECK_EQUAL(acc->second.at(0), d_it->second.at(0));
      acc.release();

      BOOST_CHECK(ca.find(const_acc,*it));
      BOOST_CHECK_EQUAL(const_acc->second.at(0), d_it->second.at(0));
      const_acc.release();
    }
  }
}

BOOST_AUTO_TEST_CASE( tile_construction )
{
  Array3::const_accessor acc;
  for(TRange3::range_type::const_iterator it = trng.tiles().begin(); it != trng.tiles().end(); ++it) {
    if(a.find(acc,*it)) {
      BOOST_CHECK_EQUAL(acc->second.start(), trng.tile(*it).start());
      BOOST_CHECK_EQUAL(acc->second.finish(), trng.tile(*it).finish());
      BOOST_CHECK_EQUAL(acc->second.size(), trng.tile(*it).size());
      BOOST_CHECK_EQUAL(acc->second.volume(), trng.tile(*it).volume());
    }
    acc.release();
  }
}

BOOST_AUTO_TEST_SUITE_END()
