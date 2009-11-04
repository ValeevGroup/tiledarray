#include "utility.h"
#include "array_storage.h"
#include "coordinates.h"
#include "iterationtest.h"
#include "permutation.h"
#include "../madness_fixture.h"
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>
#include <boost/functional.hpp>
#include <numeric>
#include <algorithm>
#include <iterator>

using namespace TiledArray;

struct ArrayDimFixture {
  typedef detail::ArrayDim<std::size_t, 3, LevelTag<0> > ArrayDim3;
  typedef ArrayDim3::index_type index_type;
  typedef ArrayDim3::size_array size_array;

  ArrayDimFixture() {
    s[0] = 2;
    s[1] = 3;
    s[2] = 4;

    w[0] = 12;
    w[1] = 4;
    w[2] = 1;

    v = 24;

    d.resize(s);
  }
  ~ArrayDimFixture() { }

  size_array s;
  size_array w;
  std::size_t v;
  ArrayDim3 d;
};


BOOST_FIXTURE_TEST_SUITE( array_dim_suite, ArrayDimFixture )

BOOST_AUTO_TEST_CASE( access )
{
  BOOST_CHECK_EQUAL( d.volume(), v); // check volume calculation
  BOOST_CHECK_EQUAL( d.size(), s);    // check the size accessor
  BOOST_CHECK_EQUAL( d.weight(), w);  // check weight accessor
}

BOOST_AUTO_TEST_CASE( includes )
{
  Range<std::size_t, 3> r(s);
  for(Range<std::size_t, 3>::const_iterator it = r.begin(); it != r.end(); ++it)
    BOOST_CHECK(d.includes( *it )); // check that all the expected indexes are
                                     // included.

  std::vector<index_type> p;
  p.push_back(index_type(2,2,3));
  p.push_back(index_type(1,3,3));
  p.push_back(index_type(1,2,4));
  p.push_back(index_type(2,3,4));

  for(std::vector<index_type>::const_iterator it = p.begin(); it != p.end(); ++it)
    BOOST_CHECK(! d.includes(*it));  // Check that elements outside the range
                                     // are not included.
}

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(ArrayDim3 d0); // Check default construction
  ArrayDim3 d0;
  BOOST_CHECK_EQUAL(d0.volume(), 0u);        // check for zero size with default
  ArrayDim3::size_array s0 = {{0,0,0}}; // construction.
  BOOST_CHECK_EQUAL(d0.size(), s0);
  BOOST_CHECK_EQUAL(d0.weight(), s0);

  BOOST_REQUIRE_NO_THROW(ArrayDim3 d1(s)); // check size constructor
  ArrayDim3 d1(s);
  BOOST_CHECK_EQUAL(d1.volume(), v);  // check that a has correct va
  BOOST_CHECK_EQUAL(d1.size(), s);
  BOOST_CHECK_EQUAL(d1.weight(), w);

  BOOST_REQUIRE_NO_THROW(ArrayDim3 d2(d));
  ArrayDim3 d2(d);
  BOOST_CHECK_EQUAL(d2.volume(), d.volume());
  BOOST_CHECK_EQUAL(d2.size(), d.size());
  BOOST_CHECK_EQUAL(d2.weight(), d.weight());
}

BOOST_AUTO_TEST_CASE( ordinal )
{
  Range<std::size_t, 3> r(s);
  ArrayDim3::ordinal_type o = 0;
  for(Range<std::size_t, 3>::const_iterator it = r.begin(); it != r.end(); ++it, ++o)
    BOOST_CHECK_EQUAL(d.ordinal( *it ), o); // check ordinal calculation and order

#ifdef TA_EXCEPTION_ERROR
  index_type p(2,3,4);
  BOOST_CHECK_THROW(d.ordinal(p), std::out_of_range); // check for throw with
                                                  // an out of range element.
#endif
}

BOOST_AUTO_TEST_CASE( ordinal_fortran )
{
  typedef detail::ArrayDim<std::size_t, 3, LevelTag<0>, CoordinateSystem<3, detail::increasing_dimension_order> > FArrayDim3;
  typedef Range<std::size_t, 3, LevelTag<0>, CoordinateSystem<3, detail::increasing_dimension_order> > range_type;

  FArrayDim3 d1(s);
  range_type r(s);
  FArrayDim3::ordinal_type o = 0;
  for(range_type::const_iterator it = r.begin(); it != r.end(); ++it, ++o)
    BOOST_CHECK_EQUAL(d1.ordinal( *it ), o); // check ordinal calculation and
                                             // order for fortran ordering.
}

BOOST_AUTO_TEST_CASE( resize )
{
  ArrayDim3 d1;
  d1.resize(s);
  BOOST_CHECK_EQUAL(d1.volume(), v); // check for correct resizing.
  BOOST_CHECK_EQUAL(d1.size(), s);
  BOOST_CHECK_EQUAL(d1.weight(), w);
}

BOOST_AUTO_TEST_SUITE_END()

struct DenseArrayStorageFixture : public ArrayDimFixture {
  static const std::size_t ndim = 5;
  typedef DenseArrayStorage<int, 3> DenseArray3;
  typedef DenseArrayStorage<int, ndim> DenseArrayN;
  typedef Permutation<ndim> PermutationN;

  DenseArrayStorageFixture() : ArrayDimFixture(), da3(s, 1) {

    PermutationN::Array pp = {{0, 2, 1, 4, 3}};
    p0 = pp;
    DenseArrayN::size_array n = {{3, 5, 7, 11, 13}};
    DenseArrayN::size_array n_p0 = p0 ^ n;
    daN.resize(n);
    daN_p0.resize(n_p0);

    typedef Range<DenseArrayN::ordinal_type, ndim, DenseArrayN::tag_type, DenseArrayN::coordinate_system> RangeN;
    typedef RangeN::const_iterator index_iter;
    typedef DenseArrayN::iterator iter;
    RangeN range(n);
    RangeN range_p0(n_p0);
    iter v = daN.begin();
    for(index_iter i=range.begin(); i!=range.end(); ++i, ++v) {
      *v = daN.ordinal(*i);
      daN_p0[p0 ^ *i] = *v;
    }

    //std::copy(daN.begin(), daN.end(), std::ostream_iterator<int>(std::cout, "\n"));
    //std::copy(daN_p0.begin(), daN_p0.end(), std::ostream_iterator<int>(std::cout, "\n"));
  }

  ~DenseArrayStorageFixture() {
  }

  DenseArray3 da3;
  DenseArrayN daN;
  DenseArrayN daN_p0;
  PermutationN p0;
};

BOOST_FIXTURE_TEST_SUITE( dense_storage_suite, DenseArrayStorageFixture )

BOOST_AUTO_TEST_CASE( array_dims )
{
  BOOST_CHECK_EQUAL(da3.size(), s);
  BOOST_CHECK_EQUAL(da3.weight(), w);
  BOOST_CHECK_EQUAL(da3.volume(), v);
}

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(DenseArray3 a0); // check default constructor
  DenseArray3 a0;
  BOOST_CHECK_EQUAL(a0.volume(), 0ul); // check for zero size.
  BOOST_CHECK_THROW(a0.at(0), std::out_of_range); // check for data access error.

  BOOST_REQUIRE_NO_THROW(DenseArray3 a1(s, 1)); // check size constructor w/ initial value.
  DenseArray3 a1(s, 1);
  BOOST_CHECK_EQUAL(a1.volume(), v); // check for expected size.
  for(DenseArray3::ordinal_type i = 0; i < v; ++i)
    BOOST_CHECK_EQUAL(a1.at(i), 1); // check for expected values.

  boost::array<int, 24> val = {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}};
  BOOST_REQUIRE_NO_THROW(DenseArray3 a2(s, val.begin(), val.end())); // check size constructor w/ initialization list.
  DenseArray3 a2(s, val.begin(), val.end());
  BOOST_CHECK_EQUAL(a2.volume(), v); // check for expected size.
  int v2 = 0;
  for(DenseArray3::ordinal_type i = 0; i < v; ++i, ++v2)
    BOOST_CHECK_EQUAL(a2.at(i), v2); // check for expected values.

  BOOST_REQUIRE_NO_THROW(DenseArray3 a3(da3));
  DenseArray3 a3(da3);
  BOOST_CHECK_EQUAL(a3.volume(), v);
  for(DenseArray3::ordinal_type i = 0; i < v; ++i)
    BOOST_CHECK_EQUAL(a3.at(i), 1); // check for expected values.

  BOOST_REQUIRE_NO_THROW(DenseArray3 a4(s, val.begin(), val.end() - 1)); // check size constructor w/ short initialization list.
  DenseArray3 a4(s, val.begin(), val.end() - 3);
  int v4 = 0;
  for(DenseArray3::ordinal_type i = 0; i < v - 3; ++i, ++v4)
    BOOST_CHECK_EQUAL(a4.at(i), v4);
  for(DenseArray3::ordinal_type i = v - 3; i < v; ++i)
    BOOST_CHECK_EQUAL(a4.at(i), int());
}

BOOST_AUTO_TEST_CASE( accessor )
{
  for(DenseArray3::ordinal_type i = 0; i < v; ++i)
    BOOST_CHECK_EQUAL(da3.at(i), 1);        // check ordinal access
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(da3.at(v), std::out_of_range);// check for out of range error
  BOOST_CHECK_THROW(da3.at(v), std::out_of_range);
#endif

  Range<std::size_t, 3> b(s);
  for(Range<std::size_t, 3>::const_iterator it = b.begin(); it != b.end(); ++it)
    BOOST_CHECK_EQUAL(da3.at(* it), 1);        // check index access

  DenseArray3 a1(s, 1);
  BOOST_CHECK_EQUAL((a1.at(1) = 2), 2); // check for write access with at
  BOOST_CHECK_EQUAL(a1.at(1), 2);
  DenseArray3::index_type p1(0,0,2);
  BOOST_CHECK_EQUAL((a1.at(p1) = 2), 2);
  BOOST_CHECK_EQUAL(a1.at(p1), 2);

  DenseArray3::index_type p2(s);
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(da3.at(p2), std::out_of_range);// check for out of range error
#endif

#ifdef NDEBUG // operator() calls at() when debugging so we don't need to run this test in that case.
  for(DenseArray3::ordinal_type i = 0; i < v; ++i)
    BOOST_CHECK_EQUAL(da3.[i], 1);        // check ordinal access
  for(Range<std::size_t, 3>::const_iterator it = b.begin(); it != b.end(); ++it)
    BOOST_CHECK_EQUAL(da3.[* it], 1);     // check index access

  DenseArray3 a2(s, 1);
  BOOST_CHECK_EQUAL((a2[1] = 2), 2); // check for write access with at
  BOOST_CHECK_EQUAL(a2[1], 2);
  DenseArray3::index_type p3(0,0,2);
  BOOST_CHECK_EQUAL((a2[p3] = 2), 2);
  BOOST_CHECK_EQUAL(a2[p3], 2);
#endif
}

BOOST_AUTO_TEST_CASE( iteration )
{
  boost::array<int, 24> val = {{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23}};
  DenseArray3 a1(s, val.begin(), val.end());
  DenseArray3::iterator it = a1.begin();
  BOOST_CHECK(iteration_test(a1, val.begin(), val.end())); // Check interator basic functionality
  BOOST_CHECK(const_iteration_test(a1, val.begin(), val.end())); // Check const iterator basic functionality
  BOOST_CHECK_EQUAL(a1.end() - a1.begin(), (std::ptrdiff_t)v); // check iterator difference operator
  BOOST_CHECK_EQUAL(*it, 0); // check dereference
  BOOST_CHECK_EQUAL(* (it + 2), 2); // check random access
  BOOST_CHECK_EQUAL((*it = 5), 5 ); // check iterator write access.
  BOOST_CHECK_EQUAL(*it, 5);
}

BOOST_AUTO_TEST_CASE( assignment )
{
  DenseArray3 a1;
  BOOST_CHECK_NO_THROW(a1 = da3);
  BOOST_CHECK_EQUAL_COLLECTIONS(a1.begin(), a1.end(), da3.begin(), da3.end());
}

BOOST_AUTO_TEST_CASE( resize )
{
  DenseArray3::size_array s1 = {{3,2,2}};
  //                           {{0,  1,  2,  3, 10, 11, 12, 13, 20, 21, 22, 23,100,101,102,103,110,111,112,113,120,121,122,123}}
  boost::array<int, 24> val =  {{2,  2,  1,  1,  2,  2,  1,  1,  1,  1,  1,  1,  2,  2,  1,  1,  2,  2,  1,  1,  1,  1,  1,  1}};
  DenseArray3 a1(s1, 2);
  a1.resize(s,1);
  BOOST_CHECK_EQUAL(a1.size(), s);  // check that new dimensions are calculations.
  BOOST_CHECK_EQUAL(a1.volume(), v);
  BOOST_CHECK_EQUAL(a1.weight(), w);
  BOOST_CHECK_EQUAL_COLLECTIONS(a1.begin(), a1.end(), val.begin(), val.end()); // check that values from original are maintained.

}

BOOST_AUTO_TEST_CASE( permutation )
{
  DenseArrayN a1(daN.size(), daN.begin(), daN.end());
  DenseArrayN a3(a1);
  BOOST_CHECK_EQUAL_COLLECTIONS(a1.begin(), a1.end(), daN.begin(), daN.end());
  BOOST_CHECK_EQUAL_COLLECTIONS(a3.begin(), a3.end(), daN.begin(), daN.end());
  DenseArrayN a2 = p0 ^ a1;
  BOOST_CHECK_EQUAL_COLLECTIONS(a2.begin(), a2.end(), daN_p0.begin(), daN_p0.end()); // check permutation

  a3 ^= p0;
  BOOST_CHECK_EQUAL_COLLECTIONS(a3.begin(), a3.end(), daN_p0.begin(), daN_p0.end()); // check in place permutation
}

BOOST_AUTO_TEST_SUITE_END()



struct DistributedArrayStorageFixture : public ArrayDimFixture {
  typedef detail::ArrayDim<std::size_t, 3, LevelTag<1> > ArrayDim3;
  typedef std::vector<double> data_type;
  typedef DistributedArrayStorage<data_type, 3> DistArray3;
  typedef DistArray3::index_type index_type;
  typedef boost::array<std::pair<DistArray3::index_type, data_type>, 24> data_array;
  typedef Range<std::size_t, 3, LevelTag<1>, CoordinateSystem<3> > range_type;

  DistributedArrayStorageFixture() : world(MadnessFixture::world), r(s),
      a(*world, s), ca(a), d(s) {
    double val = 0.0;
    range_type r(index_type(0,0,0), index_type(2,3,4));
    range_type::const_iterator r_it = r.begin();
    for(data_array::iterator it = data.begin(); it != data.end(); ++it) {
      it->first = *r_it++;
      it->second.resize(24, ++val);
      if(a.is_local(it->first))
        a.insert(*it);
    }

    world->gop.fence();
  }

  double sum_first(const DistArray3& a) {
    double sum = 0.0;
    for(DistArray3::const_iterator it = a.begin(); it != a.end(); ++it)
      sum += it->second.front();

    world->mpi.comm().Allreduce(MPI_IN_PLACE, &sum, 1, MPI::DOUBLE, MPI::SUM);

    return sum;
  }

  std::size_t tile_count(const DistArray3& a) {
    int n = static_cast<int>(a.volume(true));
    world->mpi.comm().Allreduce(MPI_IN_PLACE, &n, 1, MPI::INT, MPI::SUM);

    return n;
  }

  madness::World* world;
  range_type r;
  data_array data;
  DistArray3 a;
  const DistArray3& ca;
  ArrayDim3 d;
}; // struct DistributedArrayStorageFixture

template<typename D>
struct dim_ord : public std::unary_function<typename D::index_type, typename D::ordinal_type> {
  dim_ord(const D& d) : d_(d) { }

  typename D::ordinal_type operator()(const typename D::index_type& i) { return d_.ord(i); }
private:
  const D d_;
};

BOOST_FIXTURE_TEST_SUITE( distributed_storage_suite, DistributedArrayStorageFixture )

BOOST_AUTO_TEST_CASE( array_dims )
{
  BOOST_CHECK_EQUAL(a.size(), s);
  BOOST_CHECK_EQUAL(a.weight(), w);
  BOOST_CHECK_EQUAL(a.volume(), v);
  BOOST_CHECK(a.includes(r.start())); // check index includes check
  BOOST_CHECK(a.includes(index_type(1,2,3)));
  BOOST_CHECK(!a.includes(r.finish()));
}

BOOST_AUTO_TEST_CASE( iterator )
{
  for(DistArray3::iterator it = a.begin(); it != a.end(); ++it){
    data_array::const_iterator d_it = std::find_if(data.begin(), data.end(),
        detail::make_unary_transform(boost::bind1st(std::equal_to<DistArray3::index_type>(), it->first),
        detail::pair_first<DistArray3::value_type>()));
    BOOST_CHECK(d_it != data.end());
    BOOST_CHECK_CLOSE(it->second.front(), d_it->second.front(), 0.000001);
  }

  for(DistArray3::const_iterator it = a.begin(); it != a.end(); ++it) { // check const/non-const iterator functionality
    data_array::const_iterator d_it = std::find_if(data.begin(), data.end(),
        detail::make_unary_transform(boost::bind1st(std::equal_to<DistArray3::index_type>(), it->first),
        detail::pair_first<DistArray3::value_type>()));
    BOOST_CHECK(d_it != data.end());
    BOOST_CHECK_CLOSE(it->second.front(), d_it->second.front(), 0.000001);
  }

  const DistArray3& a_ref = a;
  for(DistArray3::const_iterator it = a_ref.begin(); it != a_ref.end(); ++it) { // check const iterator functionality
    data_array::const_iterator d_it = std::find_if(data.begin(), data.end(),
        detail::make_unary_transform(boost::bind1st(std::equal_to<DistArray3::index_type>(), it->first),
        detail::pair_first<DistArray3::value_type>()));
    BOOST_CHECK(d_it != data.end());
    BOOST_CHECK_CLOSE(it->second.front(), d_it->second.front(), 0.000001);
  }
}

BOOST_AUTO_TEST_CASE( accessors )
{
  for(range_type::const_iterator it = r.begin(); it != r.end(); ++it) {
    if(a.is_local(*it)) {
      {
        DistArray3::accessor acc;
        BOOST_CHECK(a.find(acc,*it));
        BOOST_CHECK_CLOSE(acc->second.front(), data.at(d.ord(*it)).second.front(), 0.000001);
      }

      {
        DistArray3::const_accessor const_acc;
        BOOST_CHECK(ca.find(const_acc,*it));
        BOOST_CHECK_CLOSE(const_acc->second.front(), data.at(d.ord(*it)).second.front(), 0.000001);
      }
    }
  }
}

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(DistArray3 a1(*world));
  DistArray3 a1(*world);
  BOOST_CHECK_EQUAL(a1.volume(true), 0ul);
  BOOST_CHECK(a1.begin() == a1.end()); // Check that the array is empty

  BOOST_REQUIRE_NO_THROW(DistArray3 a2(*world, s));
  DistArray3 a2(*world, s);
  BOOST_CHECK(a2.begin() == a2.end()); // Check that the array is empty

  BOOST_REQUIRE_NO_THROW(DistArray3 a3(*world, s, data.begin(), data.end()));
  DistArray3 a3(*world, s, data.begin(), data.end()); // check construction of
  BOOST_CHECK_CLOSE(sum_first(a3), 300.0, 0.0001);
  BOOST_CHECK_EQUAL(tile_count(a3), 24u);
}

BOOST_AUTO_TEST_CASE( clone )
{
  DistArray3 a1(*world);
  a1.clone(a);
  BOOST_CHECK_CLOSE(sum_first(a1), 300.0, 0.0001);
  BOOST_CHECK_EQUAL(tile_count(a1), 24ul);
}

BOOST_AUTO_TEST_CASE( insert_erase )
{
  DistArray3 a1(*world, s);
  std::size_t n = 0ul;
  double s = 0.0;

  for(data_array::iterator it = data.begin(); it != data.end(); ++it) { // check insert of pairs
    BOOST_CHECK_CLOSE(sum_first(a1), s, 0.0001);
    BOOST_CHECK_EQUAL(tile_count(a1), n);
    if(a1.is_local(it->first))
      a1.insert(*it);
    ++n;
    s += it->second.front();
  }

  for(data_array::iterator it = data.begin(); it != data.end(); ++it) { // check erasure of element
    BOOST_CHECK_CLOSE(sum_first(a1), s, 0.0001);
    BOOST_CHECK_EQUAL(tile_count(a1), n);
    if(a1.is_local(it->first))
      a1.erase(it->first);
    --n;
    s -= it->second.front();
  }

  for(data_array::iterator it = data.begin(); it != data.end(); ++it) { // check insert of element at index
    BOOST_CHECK_CLOSE(sum_first(a1), s, 0.0001);
    BOOST_CHECK_EQUAL(tile_count(a1), n);
    if(a1.is_local(it->first))
      a1.insert(it->first, it->second);
    ++n;
    s += it->second.front();
  }

  BOOST_CHECK_CLOSE(sum_first(a1), s, 0.0001);
  BOOST_CHECK_EQUAL(tile_count(a1), n);

  a1.erase(a1.begin(), a1.end()); // check erasing everything with iterators
  world->gop.fence();

  s = 0.0;
  n = 0ul;

  BOOST_CHECK_CLOSE(sum_first(a1), s, 0.0001);
  BOOST_CHECK_EQUAL(tile_count(a1), n);


  for(data_array::iterator it = data.begin(); it != data.end(); ++it) { // check communicating insert
    BOOST_CHECK_CLOSE(sum_first(a1), s, 0.0001);
    BOOST_CHECK_EQUAL(tile_count(a1), n);

    if(world->mpi.comm().rank() == 0)
      a1.insert(*it);
    world->gop.fence();

    ++n;
    s += it->second.front();
  }

  a1.erase(a1.begin(), a1.end());
  world->gop.fence();
  if(world->mpi.comm().rank() == 0)  // check communicating iterator insert.
    a1.insert(data.begin(), data.end());
  world->gop.fence();
  BOOST_CHECK_CLOSE(sum_first(a1), s, 0.0001);
  BOOST_CHECK_EQUAL(tile_count(a1), n);

}

BOOST_AUTO_TEST_CASE( clear )
{
  BOOST_CHECK_CLOSE(sum_first(a), 300.0, 0.0001);
  BOOST_CHECK_EQUAL(tile_count(a), 24u);
  a.clear();
  BOOST_CHECK_CLOSE(sum_first(a), 0.0, 0.0001);
  BOOST_CHECK_EQUAL(tile_count(a), 0u);
}

BOOST_AUTO_TEST_CASE( find )
{
  typedef madness::Future<DistArray3::iterator> future_iter;

  if(world->mpi.comm().rank() == 0) {
    DistArray3::ordinal_type i = 0;
    for(range_type::const_iterator it = r.begin(); it != r.end(); ++it, ++i) {
      future_iter v = a.find(*it);  // check find function with coordinate index
      BOOST_CHECK_CLOSE(v.get()->second.front(), data[i].second.front(), 0.0001);
    }
  }
  world->gop.fence();
}

BOOST_AUTO_TEST_CASE( swap_array )
{
  DistArray3 a1(*world);
  a1.clone(a);
  DistArray3 a2(*world, s);

  BOOST_CHECK_CLOSE(sum_first(a1), 300.0, 0.0001); // verify initial conditions
  BOOST_CHECK_EQUAL(tile_count(a1), 24ul);
  BOOST_CHECK(a2.begin() == a2.end());
  swap(a1, a2);

  BOOST_CHECK_CLOSE(sum_first(a2), 300.0, 0.0001); // check that the arrays were
  BOOST_CHECK_EQUAL(tile_count(a2), 24ul);           // swapped correctly.
  BOOST_CHECK(a1.begin() == a1.end());
}

BOOST_AUTO_TEST_CASE( resize )
{
  size_array s_big = {{ 4, 4, 4 }};
  size_array w_big = {{ 16, 4, 1}};
  DistArray3::ordinal_type v_big = 64;
  size_array s_small = {{ 2, 2, 2 }};
  size_array w_small = {{ 4, 2, 1 }};
  DistArray3::ordinal_type v_small = 8;

  range_type r_big(s_big);
  range_type r_small(s_small);

  DistArray3 a1(*world);
  a1.clone(a);
  DistArray3 a2(*world);
  a2.clone(a);

  a1.resize(s_big);   // Check resize for bigger resulting array
  BOOST_CHECK_EQUAL(a1.size(), s_big);
  BOOST_CHECK_EQUAL(a1.weight(), w_big);
  BOOST_CHECK_EQUAL(a1.volume(), v_big);
  BOOST_CHECK_CLOSE(sum_first(a1), 300.0, 0.0001);
  BOOST_CHECK_EQUAL(tile_count(a1), 24ul);
  BOOST_CHECK(a1.includes(index_type(3,3,3)));  // check that an element can be added
  a1.insert(index_type(3,3,3), data[23].second);// to a new location after resize
  BOOST_CHECK_CLOSE(sum_first(a1), 324.0, 0.0001);
  BOOST_CHECK_EQUAL(tile_count(a1), 25ul);

  a2.resize(s_small); // Check resize for a smaller resulting array
  BOOST_CHECK_EQUAL(a2.size(), s_small);
  BOOST_CHECK_EQUAL(a2.weight(), w_small);
  BOOST_CHECK_EQUAL(a2.volume(), v_small);
  BOOST_CHECK_CLOSE(sum_first(a2), 76.0, 0.0001);
  BOOST_CHECK_EQUAL(tile_count(a2), 8ul);
}

BOOST_AUTO_TEST_CASE( permute )
{
  Permutation<3> p(2, 0, 1);
  a ^= p;
  DistArray3::const_accessor acc;
  for(data_array::const_iterator it = data.begin(); it != data.end(); ++it) {
    if(a.find(acc, p ^ it->first)) {
      BOOST_CHECK_EQUAL(acc->second.front(), it->second.front());
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
