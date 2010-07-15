#include "TiledArray/array.h"
#include "TiledArray/utility.h"
#include <boost/functional.hpp>
#include <algorithm>
#include "unit_test_config.h"

using namespace TiledArray;

struct ArrayFixture {
/*
  typedef Array<int, 3> Array3;
  typedef Array3::index_type index_type;
  typedef Array3::tile_index tile_index_type;
  typedef Array3::tile_type tile_type;
  typedef std::vector<std::pair<index_type, tile_type> > data_array;

  ArrayFixture() : world(GlobalFixture::world), a(*world, trng), ca(a) {
    int v = 1;
    int tv = 1;
    for(TRangeN::range_type::const_iterator it = a.tiles().begin(); it != a.tiles().end(); ++it) {
      tile_type t(a.tile(*it), v);
      tv = v++;
      for(tile_type::iterator t_it = t.begin(); t_it != t.end(); ++t_it)
        *t_it = tv++;
      a.insert(*it, t);
      data.push_back(std::pair<index_type, tile_type>(*it, t));
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
            detail::make_unary_transform(std::bind1st(std::equal_to<Array3::index_type>(), i),
            detail::pair_first<Array3::value_type>()));
  }

  madness::World* world;
  Array3 a;
  const Array3& ca;
  data_array data;
*/
}; // struct ArrayFixture

BOOST_FIXTURE_TEST_SUITE( array_suite , ArrayFixture )
/*
BOOST_AUTO_TEST_CASE( array_dims )
{
  BOOST_CHECK_EQUAL(a.start(), trng.tiles().start());
  BOOST_CHECK_EQUAL(a.finish(), trng.tiles().finish());
  BOOST_CHECK_EQUAL(a.size(), trng.tiles().size());
  Array3::size_array w = {{20, 5, 1}};
  BOOST_CHECK_EQUAL(a.weight(), w);
  BOOST_CHECK_EQUAL(a.volume(), trng.tiles().volume());
  BOOST_CHECK_EQUAL(a.tiles(), trng.tiles());
  BOOST_CHECK_EQUAL(a.elements(), trng.elements());
  for(TRangeN::range_type::const_iterator it = trng.tiles().begin(); it != trng.tiles().end(); ++it) {
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
  for(TRangeN::range_type::const_iterator it = trng.tiles().begin(); it != trng.tiles().end(); ++it) {
    if(a.is_local(*it)) {
      d_it = find_data_element(*it);
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
  for(TRangeN::range_type::const_iterator it = trng.tiles().begin(); it != trng.tiles().end(); ++it) {
    if(a.find(acc,*it)) {
      BOOST_CHECK_EQUAL(acc->second.start(), trng.tile(*it).start());
      BOOST_CHECK_EQUAL(acc->second.finish(), trng.tile(*it).finish());
      BOOST_CHECK_EQUAL(acc->second.size(), trng.tile(*it).size());
      BOOST_CHECK_EQUAL(acc->second.volume(), trng.tile(*it).volume());
      acc.release();
    }
  }
}

BOOST_AUTO_TEST_CASE( clear )
{
  BOOST_CHECK_EQUAL(sum_first(a), 1830);
  BOOST_CHECK_EQUAL(tile_count(a), 60ul);
  a.clear();
  BOOST_CHECK_EQUAL(sum_first(a), 0);
  BOOST_CHECK_EQUAL(tile_count(a), 0ul);
}

BOOST_AUTO_TEST_CASE( insert_erase )
{
  a.clear();

  BOOST_CHECK_EQUAL(sum_first(a), 0);
  BOOST_CHECK_EQUAL(tile_count(a), 0ul);
  for(data_array::iterator it = data.begin(); it != data.end(); ++it) { // check inserting pairs
    if(a.is_local(it->first))
      a.insert(*it);
  }
  world->gop.fence();
  BOOST_CHECK_EQUAL(sum_first(a), 1830);
  BOOST_CHECK_EQUAL(tile_count(a), 60ul);

  for(data_array::iterator it = data.begin(); it != data.end(); ++it) { // check erasure of element
    if(a.is_local(it->first))
      a.erase(it->first);
  }
  world->gop.fence();
  BOOST_CHECK_EQUAL(sum_first(a), 0);
  BOOST_CHECK_EQUAL(tile_count(a), 0ul);

  for(data_array::iterator it = data.begin(); it != data.end(); ++it) { // check insert of element at index
    if(a.is_local(it->first))
      a.insert(it->first, it->second);
  }
  world->gop.fence();
  BOOST_CHECK_EQUAL(sum_first(a), 1830);
  BOOST_CHECK_EQUAL(tile_count(a), 60ul);

  a.erase(a.begin(), a.end()); // check erasing everything with iterators
  world->gop.fence();
  BOOST_CHECK_EQUAL(sum_first(a), 0);
  BOOST_CHECK_EQUAL(tile_count(a), 0ul);

  for(data_array::iterator it = data.begin(); it != data.end(); ++it) { // check insert of element
    if(a.is_local(it->first))                                   // w/ tile iterator initialization
      a.insert(it->first, it->second.begin(), it->second.end());
  }
  world->gop.fence();
  BOOST_CHECK_EQUAL(sum_first(a), 1830);
  BOOST_CHECK_EQUAL(tile_count(a), 60ul);

  a.clear();
  world->gop.fence();
  for(data_array::iterator it = data.begin(); it != data.end(); ++it) { // check insert of element
    if(a.is_local(it->first))                                   // w/ tile constant initialization
      a.insert(it->first, it->second.at(0));
  }
  world->gop.fence();
  BOOST_CHECK_EQUAL(sum_first(a), 1830);
  BOOST_CHECK_EQUAL(tile_count(a), 60ul);

  a.clear();
  world->gop.fence();
  if(world->mpi.comm().rank() == 0)
    for(data_array::iterator it = data.begin(); it != data.end(); ++it) // check communicating insert
      a.insert(*it);
  world->gop.fence();
  BOOST_CHECK_EQUAL(sum_first(a), 1830);
  BOOST_CHECK_EQUAL(tile_count(a), 60ul);
}

BOOST_AUTO_TEST_CASE( clone )
{
  Array3 a1(*world, trng);
  BOOST_CHECK_EQUAL(sum_first(a1), 0);
  BOOST_CHECK_EQUAL(tile_count(a1), 0ul);
  a1.clone(a); // Test a deep copy.
  BOOST_CHECK_EQUAL(sum_first(a1), 1830);
  BOOST_CHECK_EQUAL(tile_count(a1), 60ul);
}

BOOST_AUTO_TEST_CASE( swap_array )
{
  Array3 a1(*world, trng);
  a1.clone(a);
  Array3 a2(*world, trng);

  BOOST_CHECK_EQUAL(sum_first(a1), 1830); // verify initial conditions
  BOOST_CHECK_EQUAL(tile_count(a1), 60ul);
  BOOST_CHECK(a2.begin() == a2.end());
  TiledArray::swap(a1, a2);

  BOOST_CHECK_EQUAL(sum_first(a2), 1830); // check that the arrays were
  BOOST_CHECK_EQUAL(tile_count(a2), 60ul);           // swapped correctly.
  BOOST_CHECK(a1.begin() == a1.end());
}

BOOST_AUTO_TEST_CASE( find )
{
  typedef madness::Future<Array3::iterator> future_iter;

  if(world->mpi.comm().rank() == 0) {
    data_array::const_iterator d_it;
    for(TRangeN::range_type::const_iterator it = trng.tiles().begin(); it != trng.tiles().end(); ++it) {
      future_iter v = a.find(*it);  // check find function with coordinate index
      d_it = find_data_element(*it);
      BOOST_CHECK_EQUAL(v.get()->second.at(0), d_it->second.at(0));
    }
  }

  future_iter v = a.find(a.finish());
  BOOST_CHECK(v.get() == a.end());
}

BOOST_AUTO_TEST_CASE( permute )
{
  Permutation<3> p(2, 0, 1);
  a ^= p;
  Array3::const_accessor acc;
  for(data_array::const_iterator it = data.begin(); it != data.end(); ++it) {
    if(a.find(acc, p ^ it->first)) {
      tile_type pt =  p ^ it->second;
      BOOST_CHECK_EQUAL(acc->second.at(0), pt.at(0));
      BOOST_CHECK_EQUAL(acc->second.range(), pt.range());
      BOOST_CHECK_EQUAL_COLLECTIONS(acc->second.begin(), acc->second.end(), pt.begin(), pt.end());
    }
  }
}

BOOST_AUTO_TEST_CASE( resize )
{
  const std::size_t d0[] = {0, 10, 20};
  const std::size_t d1[] = {0, 5, 10, 15};
  const std::size_t d2[] = {0, 3, 6, 9, 12};
  const TRange1 dim0(d0, d0 + 3);
  const TRange1 dim1(d1, d1 + 4);
  const TRange1 dim2(d2, d2 + 5);
  const TRange1 dims[3] = {dim0, dim1, dim2};
  TRangeN trng1(dims, dims + 3);

  a.resize(trng1);
  BOOST_CHECK_EQUAL(sum_first(a), 0);   // check for no data
  BOOST_CHECK_EQUAL(tile_count(a), 0ul);

  BOOST_CHECK_EQUAL(a.start(), trng1.tiles().start());     // check the range dimensions for correctness
  BOOST_CHECK_EQUAL(a.finish(), trng1.tiles().finish());
  BOOST_CHECK_EQUAL(a.size(), trng1.tiles().size());
  Array3::size_array w = {{12, 4, 1}};
  BOOST_CHECK_EQUAL(a.weight(), w);
  BOOST_CHECK_EQUAL(a.volume(), trng1.tiles().volume());
  BOOST_CHECK_EQUAL(a.tiles(), trng1.tiles());
  BOOST_CHECK_EQUAL(a.elements(), trng1.elements());
  for(TRangeN::range_type::const_iterator it = trng1.tiles().begin(); it != trng1.tiles().end(); ++it) {
    BOOST_CHECK_EQUAL(a.tile(*it), trng1.tile(*it));
  }

  for(data_array::iterator it = data.begin(); it != data.end(); ++it) {
    if(a.is_local(it->first) && a.includes(it->first))
      a.insert(it->first, it->second);
  }
  world->gop.fence();
  BOOST_CHECK_EQUAL(sum_first(a), 420);  // check that the correct tiles were added
  BOOST_CHECK_EQUAL(tile_count(a), 24ul);
}
*/
BOOST_AUTO_TEST_SUITE_END()
