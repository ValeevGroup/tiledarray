#include "TiledArray/range_iterator.h"
#include "unit_test_config.h"
#include <world/type_traits.h>

using TiledArray::detail::RangeIterator;

// Fake container for iterator tests
struct FakeRange {
  typedef RangeIterator<int, FakeRange> iterator;
  iterator begin() const { return iterator(0, this); }
  iterator end() const { return iterator(10, this); }
  void increment(iterator::value_type& i) const { ++i; }
  void advance(iterator::value_type& i, std::ptrdiff_t n) const { i += n; }
  std::ptrdiff_t distance_to(const iterator::value_type& first, const iterator::value_type& last) const {
    return last - first;
  }
};

// Another fake container for iterator tests
struct FakePairRange {
  typedef RangeIterator<std::pair<int,int>, FakePairRange> iterator;
  iterator begin() { return iterator(std::make_pair(0,0), this); }
  iterator end() { return iterator(std::make_pair(10,10), this); }
  void increment(iterator::value_type& i) const { ++(i.first); ++(i.second); }
  void advance(iterator::value_type& i, std::ptrdiff_t n) const { i.first += n; i.second += n; }
  std::ptrdiff_t distance_to(const iterator::value_type& first, const iterator::value_type& last) const {
    return last.first - first.first + last.second - last.second;
  }
};

// Range iterator fixture class
struct RangeIteratorFixture {
  typedef RangeIterator<int, FakeRange> iterator;

  // Fixture setup
  RangeIteratorFixture() : r(), it(r.begin()) { }

  // Fixture data
  FakeRange r;
  RangeIterator<int, FakeRange> it;
};

BOOST_FIXTURE_TEST_SUITE( range_iterator_suite , RangeIteratorFixture )

// Check iterator typedefs
TA_STATIC_ASSERT( (std::is_same<RangeIterator<int, FakeRange>::value_type, int>::value) );
TA_STATIC_ASSERT( (std::is_same<RangeIterator<int, FakeRange>::reference, const int&>::value) );
TA_STATIC_ASSERT( (std::is_same<RangeIterator<int, FakeRange>::pointer, const int*>::value) );
TA_STATIC_ASSERT( (std::is_same<RangeIterator<int, FakeRange>::iterator_category, std::input_iterator_tag>::value) );
TA_STATIC_ASSERT( (std::is_same<RangeIterator<int, FakeRange>::difference_type, std::ptrdiff_t>::value) );

BOOST_AUTO_TEST_CASE( rvalue_derefence )
{
  // Check reference dereference
  BOOST_CHECK_EQUAL(*it, 0);

  FakePairRange pair_range;
  FakePairRange::iterator pair_it = pair_range.begin();

  // Check pointer dereference
  BOOST_CHECK_EQUAL(pair_it->first, 0);
  BOOST_CHECK_EQUAL(pair_it->second, 0);
}

BOOST_AUTO_TEST_CASE( equality_comparison )
{
  iterator first = r.begin();
  iterator last = r.end();

  // Check equal comparison
  BOOST_CHECK( it == first );
  BOOST_CHECK(! (it == last));

  // Check comparison for iterators to difference containers.
  FakeRange r1;
  iterator it1 = r1.begin();
  BOOST_CHECK( it1 != first);
  BOOST_CHECK( it1 != last );

  // Check not-equal comparison
  BOOST_CHECK( it != last);
  BOOST_CHECK(! (it != first));
}

BOOST_AUTO_TEST_CASE( increment )
{
  BOOST_CHECK_EQUAL(*it, 0);
  ++it;
  BOOST_CHECK_EQUAL(*it, 1);
  BOOST_CHECK_EQUAL(*it++, 1);
  BOOST_CHECK_EQUAL(*it, 2);
  it++;
  BOOST_CHECK_EQUAL(*it, 3);

}

BOOST_AUTO_TEST_CASE( assignement_copy )
{
  // Check preassignment conditions
  BOOST_CHECK(it == r.begin());
  BOOST_CHECK_EQUAL(*it, 0);

  // Check assignment
  BOOST_CHECK_EQUAL(*(it = r.end()), 10);
  BOOST_CHECK(it == r.end());
  BOOST_CHECK(it != r.begin());
  BOOST_CHECK_EQUAL(*it, 10);
}

BOOST_AUTO_TEST_CASE( advance )
{
  // Check preconditions
  BOOST_CHECK_EQUAL(*it, 0);

  std::advance(it, 5);
  BOOST_CHECK_EQUAL(*it, 5);
}

BOOST_AUTO_TEST_CASE( distance )
{
  BOOST_CHECK_EQUAL(std::distance(r.begin(), r.end()), 10);
}

BOOST_AUTO_TEST_CASE( constructor )
{
  // Check iterator construction
  BOOST_REQUIRE_NO_THROW(iterator it1(1, &r));
  iterator it1(1, &r);
  BOOST_CHECK_EQUAL(*it1, 1);

  // Check copy constructor
  BOOST_REQUIRE_NO_THROW(iterator it2(it));
  iterator it2(it);
  BOOST_CHECK_EQUAL(*it2, 0);
  BOOST_CHECK( it2 == it);
}

BOOST_AUTO_TEST_SUITE_END()
