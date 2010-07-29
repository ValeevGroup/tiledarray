#include "TiledArray/indexed_iterator.h"
#include "unit_test_config.h"
#include <vector>
#include <map>

struct IndexedIteratorFixture {
  typedef const int key_type;
  typedef std::pair<int, int> data_type;
  typedef std::pair<key_type, data_type> value_type;
  typedef std::vector<value_type> container_type;
  typedef TiledArray::detail::IndexedIterator<container_type::iterator > iterator;
  typedef TiledArray::detail::IndexedIterator<container_type::const_iterator > const_iterator;

  IndexedIteratorFixture() :
      d(/*1, data_type(2,3))*/), cd(d), it(d.begin()), cit(cd.begin())
  { }

  container_type d;
  const container_type& cd;
  iterator it;
  const_iterator cit;
};


BOOST_FIXTURE_TEST_SUITE( indexed_iterator_suite , IndexedIteratorFixture )

// Check that iterator typedef's are correct
BOOST_STATIC_ASSERT( (boost::is_same<IndexedIteratorFixture::iterator::base_type,
    IndexedIteratorFixture::container_type::iterator>::value) );
BOOST_STATIC_ASSERT( (boost::is_same<IndexedIteratorFixture::iterator::value_type,
    IndexedIteratorFixture::data_type >::value) );
BOOST_STATIC_ASSERT( (boost::is_same<IndexedIteratorFixture::iterator::pointer,
    IndexedIteratorFixture::data_type*>::value) );
BOOST_STATIC_ASSERT( (boost::is_same<IndexedIteratorFixture::iterator::reference,
    IndexedIteratorFixture::data_type&>::value) );
BOOST_STATIC_ASSERT( (boost::is_same<IndexedIteratorFixture::iterator::iterator_category,
    std::random_access_iterator_tag>::value) );
BOOST_STATIC_ASSERT( (boost::is_same<IndexedIteratorFixture::iterator::index_type,
    IndexedIteratorFixture::key_type>::value) );

// Check that const iterator typedef's are correct
BOOST_STATIC_ASSERT( (boost::is_same<IndexedIteratorFixture::const_iterator::base_type,
    IndexedIteratorFixture::container_type::const_iterator>::value) );
BOOST_STATIC_ASSERT( (boost::is_same<IndexedIteratorFixture::const_iterator::value_type,
    IndexedIteratorFixture::data_type >::value) );
BOOST_STATIC_ASSERT( (boost::is_same<IndexedIteratorFixture::const_iterator::pointer,
    const IndexedIteratorFixture::data_type*>::value) );
BOOST_STATIC_ASSERT( (boost::is_same<IndexedIteratorFixture::const_iterator::reference,
    const IndexedIteratorFixture::data_type&>::value) );
BOOST_STATIC_ASSERT( (boost::is_same<IndexedIteratorFixture::const_iterator::iterator_category,
    std::random_access_iterator_tag>::value) );
BOOST_STATIC_ASSERT( (boost::is_same<IndexedIteratorFixture::const_iterator::index_type,
    IndexedIteratorFixture::key_type>::value) );

BOOST_AUTO_TEST_CASE( base_accessor )
{
  // Check non-const iterator base accessor
  BOOST_CHECK(it.base() == d.begin());
  BOOST_CHECK(it.base() == cd.begin());

  // Check const iterator base accessor
  BOOST_CHECK(cit.base() == d.begin());
  BOOST_CHECK(cit.base() == cd.begin());
}


BOOST_AUTO_TEST_CASE( constructors )
{
  // Check default iterator construction and semantics
  {
    BOOST_REQUIRE_NO_THROW(iterator it1);
    BOOST_REQUIRE_NO_THROW(iterator it2());
    BOOST_REQUIRE_NO_THROW(iterator it3 = iterator());

    BOOST_REQUIRE_NO_THROW(const_iterator it4);
    BOOST_REQUIRE_NO_THROW(const_iterator it5());
    BOOST_REQUIRE_NO_THROW(const_iterator it6 = const_iterator());
  }

  // Check iterator constructor
  {
    BOOST_REQUIRE_NO_THROW(iterator it7(d.begin()));
    iterator it8(d.begin());
    BOOST_CHECK(it8.base() == d.begin());

    BOOST_REQUIRE_NO_THROW(const_iterator it9(d.begin()));
    const_iterator it10(cd.begin());
    BOOST_CHECK(it10.base() == cd.begin());

    BOOST_REQUIRE_NO_THROW(const_iterator it11(cd.begin()));
    const_iterator it12(d.begin());
    BOOST_CHECK(it12.base() == d.begin());
    BOOST_CHECK(it12.base() == cd.begin());
  }

  // Check copy constructor
  {
    BOOST_REQUIRE_NO_THROW(iterator it13(it));
    iterator it14(it);
    BOOST_CHECK(it14.base() == it.base());

    BOOST_REQUIRE_NO_THROW(const_iterator it15(cit));
    const_iterator it18(cit);
    BOOST_CHECK(it18.base() == cit.base());
  }

  // Check copy constructor for different base iterators
  {
    BOOST_REQUIRE_NO_THROW(const_iterator it19(it));
    const_iterator it20(it);
    BOOST_CHECK(it20.base() == it.base());
    BOOST_CHECK(it20.base() == cit.base());
  }
}

BOOST_AUTO_TEST_CASE( assignment )
{
  // Check copy assignment operator
  iterator it1;
  BOOST_CHECK(it1.base() != it.base());
  BOOST_CHECK((it1 = it).base() == it.base());
  BOOST_CHECK(it1.base() == it.base());

  // Check assignment operator for different base iterator types
  const_iterator it2;
  BOOST_CHECK(it2.base() != it.base());
  BOOST_CHECK((it2 = it).base() == it.base());
  BOOST_CHECK(it2.base() == it.base());

  // Check assignment operator for base iterator type
  iterator it3;
  BOOST_CHECK(it3.base() != it.base());
  BOOST_CHECK((it3 = d.begin()).base() == it.base());
  BOOST_CHECK(it3.base() == it.base());
}

BOOST_AUTO_TEST_CASE( index_accessor )
{
  // Check the index returns the correct value
  BOOST_CHECK_EQUAL(it.index(), 1);
}

BOOST_AUTO_TEST_CASE( dereference )
{
  // Check r-value operator*
  BOOST_CHECK_EQUAL( (*it).first, 2 );
  BOOST_CHECK_EQUAL( (*it).second, 3 );

  // Check r-value operator->
  BOOST_CHECK_EQUAL( it->first, 2 );
  BOOST_CHECK_EQUAL( it->second, 3 );

  // Check l-value operator*
  BOOST_CHECK_EQUAL( (*it = data_type(4,5)).first, 4);
  BOOST_CHECK_EQUAL( it->first, 4 );
  BOOST_CHECK_EQUAL( it->second, 5 );
}

BOOST_AUTO_TEST_CASE( comparison )
{
  iterator first(d.begin());
  iterator last(d.end());
  const_iterator cfirst(d.begin());
  const_iterator clast(d.end());

  // Check equal operator
  BOOST_CHECK( first == it );
  BOOST_CHECK( first == cfirst );
  BOOST_CHECK(! (last == it));

  // Check not equal operator
  BOOST_CHECK( last != it );
  BOOST_CHECK( last != cfirst );
  BOOST_CHECK(! (first != it));

  // Check less operator
  BOOST_CHECK( first < last );
  BOOST_CHECK( first < clast );
  BOOST_CHECK(! (last < first) );

  // Check greater operator
  BOOST_CHECK( last > first );
  BOOST_CHECK( last > cfirst );
  BOOST_CHECK(! (first > last) );

  // Check less-equal operator
  BOOST_CHECK( first <= last );
  BOOST_CHECK( first <= it );
  BOOST_CHECK( first <= clast );
  BOOST_CHECK(! (last <= first) );

  // Check greater-equal operator
  BOOST_CHECK( last >= first );
  BOOST_CHECK( it >= first );
  BOOST_CHECK( last >= cfirst );
  BOOST_CHECK(! (first >= last) );
}

BOOST_AUTO_TEST_CASE( increment )
{
  container_type::iterator it = d.begin();

  iterator it1(it);
  // Check initial conditions
  BOOST_CHECK(it1.base() == it);

  // Check prefix iteration operator
  BOOST_CHECK((++it1).base() == ++it);
  BOOST_CHECK(it1.base() == it);

  // check postfix iteration operator
  BOOST_CHECK((it1++).base() == it++);
  BOOST_CHECK(it1.base() == it);
}

BOOST_AUTO_TEST_CASE( decrement )
{
  container_type::iterator it = d.end();

  iterator it1(it);
  // Check initial conditions
  BOOST_CHECK(it1.base() == it);

  // Check prefix iteration operator
  BOOST_CHECK((--it1).base() == --it);
  BOOST_CHECK(it1.base() == it);

  // check postfix iteration operator
  BOOST_CHECK((it1--).base() == it--);
  BOOST_CHECK(it1.base() == it);
}

BOOST_AUTO_TEST_CASE( distance )
{
  iterator first(d.begin());
  iterator last(d.end());

  BOOST_CHECK((first + 10) == last);
  BOOST_CHECK((last - 10) == first);
  BOOST_CHECK_EQUAL(last - first, 10);
  BOOST_CHECK_EQUAL(first - last, -10);
  BOOST_CHECK_EQUAL(std::distance(first, last), 10);
}

BOOST_AUTO_TEST_CASE( advance )
{
  BOOST_CHECK(it.base() == d.begin());
  it += 10;
  BOOST_CHECK(it.base() == d.end());
  it -= 10;
  BOOST_CHECK(it.base() == d.begin());
}

BOOST_AUTO_TEST_CASE( offset_dereference )
{
  for(int i = 0; i < 10; ++i) {
    BOOST_CHECK_EQUAL(it[i].first, 0);
    BOOST_CHECK_EQUAL(it[i].second, 3);
  }
}

BOOST_AUTO_TEST_SUITE_END()
