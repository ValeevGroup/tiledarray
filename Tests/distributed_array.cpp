#include "TiledArray/distributed_array.h"
#include <boost/functional.hpp>
#include "unit_test_config.h"


DistributedArrayFixture::DistributedArrayFixture() : world(GlobalFixture::world), r(s),
    a(*world, s), ca(a), d(s) {
  double val = 0.0;
  range_type r(index_type(0,0,0), index_type(2,3,4));
  DistArray3::ordinal_type o = 0ul;
  range_type::const_iterator r_it = r.begin();
  for(data_array::iterator it = data.begin(); it != data.end(); ++it) {
    it->first = DistArray3::key_type(o++, *r_it++);
    it->second.resize(24, ++val);
    if(a.is_local(it->first))
      a.insert(*it);
  }

  world->gop.fence();
}

double DistributedArrayFixture::sum_first(const DistArray3& a) {
  double sum = 0.0;
  for(DistArray3::const_iterator it = a.begin(); it != a.end(); ++it)
    sum += it->second.front();

  world->mpi.comm().Allreduce(MPI_IN_PLACE, &sum, 1, MPI::DOUBLE, MPI::SUM);

  return sum;
}

std::size_t DistributedArrayFixture::tile_count(const DistArray3& a) {
  int n = static_cast<int>(a.volume(true));
  world->mpi.comm().Allreduce(MPI_IN_PLACE, &n, 1, MPI::INT, MPI::SUM);

  return n;
}

template<typename D>
struct dim_ord : public std::unary_function<typename D::index_type, typename D::ordinal_type> {
  dim_ord(const D& d) : d_(d) { }

  typename D::ordinal_type operator()(const typename D::index_type& i) { return d_.ord(i); }
private:
  const D d_;
};

BOOST_FIXTURE_TEST_SUITE( distributed_storage_suite, DistributedArrayFixture )

BOOST_AUTO_TEST_CASE( array_dims )
{
  BOOST_CHECK_EQUAL(a.size(), s);
  BOOST_CHECK_EQUAL(a.weight(), w);
  BOOST_CHECK_EQUAL(a.volume(), v);
}

BOOST_AUTO_TEST_CASE( includes )
{
  BOOST_CHECK(a.includes(data[0].first)); // check index includes check
  BOOST_CHECK(a.includes(data[0].first.key1()));
  BOOST_CHECK(a.includes(data[0].first.key2()));
  BOOST_CHECK(a.includes(DistArray3::key_type(data[0].first.key1())));
  BOOST_CHECK(a.includes(DistArray3::key_type(data[0].first.key1())));
  BOOST_CHECK(a.includes(index_type(1,2,3)));
  BOOST_CHECK(a.includes(23ul));
  BOOST_CHECK(! a.includes(r.finish()));
  BOOST_CHECK(! a.includes(24ul));
}

BOOST_AUTO_TEST_CASE( is_local )
{
  bool local = a.is_local(data[0].first);
  // check to see if all key types work for is local
  BOOST_CHECK_EQUAL(a.is_local(data[0].first.key1()), local);
  BOOST_CHECK_EQUAL(a.is_local(data[0].first.key2()), local);
  BOOST_CHECK_EQUAL(a.is_local(DistArray3::key_type(data[0].first.key1())), local);
  BOOST_CHECK_EQUAL(a.is_local(DistArray3::key_type(data[0].first.key1())), local);
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
  DistArray3::accessor acc;
  DistArray3::const_accessor const_acc;
  for(data_array::const_iterator it = data.begin(); it != data.end(); ++it) {
    if(a.is_local(it->first)) {
      BOOST_CHECK(a.find(acc,it->first));
      BOOST_CHECK_CLOSE(acc->second.front(), it->second.front(), 0.000001);
      acc.release();

      BOOST_CHECK(a.find(acc,it->first.key1()));
      BOOST_CHECK_CLOSE(acc->second.front(), it->second.front(), 0.000001);
      acc.release();

      BOOST_CHECK(a.find(acc,it->first.key2()));
      BOOST_CHECK_CLOSE(acc->second.front(), it->second.front(), 0.000001);
      acc.release();

      BOOST_CHECK(a.find(acc,DistArray3::key_type(it->first.key1())));
      BOOST_CHECK_CLOSE(acc->second.front(), it->second.front(), 0.000001);
      acc.release();

      BOOST_CHECK(a.find(acc,DistArray3::key_type(it->first.key1())));
      BOOST_CHECK_CLOSE(acc->second.front(), it->second.front(), 0.000001);
      acc.release();

      BOOST_CHECK(a.find(const_acc,it->first));
      BOOST_CHECK_CLOSE(const_acc->second.front(), it->second.front(), 0.000001);
      const_acc.release();

      BOOST_CHECK(a.find(const_acc,it->first.key1()));
      BOOST_CHECK_CLOSE(const_acc->second.front(), it->second.front(), 0.000001);
      const_acc.release();

      BOOST_CHECK(a.find(const_acc,it->first.key2()));
      BOOST_CHECK_CLOSE(const_acc->second.front(), it->second.front(), 0.000001);
      const_acc.release();

      BOOST_CHECK(a.find(const_acc,DistArray3::key_type(it->first.key1())));
      BOOST_CHECK_CLOSE(const_acc->second.front(), it->second.front(), 0.000001);
      const_acc.release();

      BOOST_CHECK(a.find(const_acc,DistArray3::key_type(it->first.key1())));
      BOOST_CHECK_CLOSE(const_acc->second.front(), it->second.front(), 0.000001);
      const_acc.release();
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
  DistArray3 a3(*world);
  a3.clone(a);

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

  a3.resize(s_small, false); // Check resize for a smaller resulting array
  BOOST_CHECK_EQUAL(a3.size(), s_small);
  BOOST_CHECK_EQUAL(a3.weight(), w_small);
  BOOST_CHECK_EQUAL(a3.volume(), v_small);
  BOOST_CHECK_CLOSE(sum_first(a3), 0.0, 0.0001);
  BOOST_CHECK_EQUAL(tile_count(a3), 0ul);
}

BOOST_AUTO_TEST_CASE( permute )
{
  Permutation<3> p(2, 0, 1);
  a ^= p;
  DistArray3::const_accessor acc;
  for(data_array::const_iterator it = data.begin(); it != data.end(); ++it) {
    if(a.find(acc, p ^ it->first.key2())) {
      BOOST_CHECK_EQUAL(acc->second.front(), it->second.front());
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
