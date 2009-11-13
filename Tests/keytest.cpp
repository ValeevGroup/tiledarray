#include "key.h"
#include "coordinates.h"
#include <cstddef>
#include <boost/test/unit_test.hpp>

using namespace TiledArray;
using namespace TiledArray::detail;

struct KeyFixture {
  typedef std::size_t ordinal_type;
  typedef ArrayCoordinate<std::size_t, 3, LevelTag<0> > index_type;
  typedef Key<ordinal_type, index_type> key_type;

  KeyFixture() : o(10ul), i(1,2,3), k(o, i), ko(o), ki(i),
      khi(11ul, index_type(1,2,4)), klow(9ul, index_type(1,2,2))
  { }

  ~KeyFixture() { }

  ordinal_type ordinal(ordinal_type o) {
    return o;
  }

  index_type index(index_type i) {
    return i;
  }

  ordinal_type o;
  index_type i;
  key_type k;
  key_type k0;
  key_type ko;
  key_type ki;
  key_type khi;
  key_type klow;
}; // struct KeyFixture

BOOST_FIXTURE_TEST_SUITE( key_suite , KeyFixture )

BOOST_AUTO_TEST_CASE( accessors )
{
  BOOST_CHECK_EQUAL(k.key1(), o); // check key1 accessor
  BOOST_CHECK_EQUAL(k.key2(), i); // check key2 accessor
  BOOST_CHECK_EQUAL(k.keys(), 3u);// check keys defined accessor.

#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(k0.key1(), std::runtime_error); // check that accessors throw correctly.
  BOOST_CHECK_THROW(k0.key2(), std::runtime_error);
  BOOST_CHECK_THROW(ki.key1(), std::runtime_error);
  BOOST_CHECK_THROW(ko.key2(), std::runtime_error);
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( constructor )
{
  BOOST_REQUIRE_NO_THROW(key_type k0); // check default constructor.
  key_type k0;
  BOOST_CHECK_EQUAL(k0.keys(), 0u);

  BOOST_REQUIRE_NO_THROW(key_type k1(o, i)); // check constructor w/ both keys.
  key_type k1(o, i);
  BOOST_CHECK_EQUAL(k1.key1(), o);
  BOOST_CHECK_EQUAL(k1.key2(), i);
  BOOST_CHECK_EQUAL(k1.keys(), 3u);

  BOOST_REQUIRE_NO_THROW(key_type k2(o)); // Check constructor w/ 1st key.
  key_type k2(o);
  BOOST_CHECK_EQUAL(k2.key1(), o);
  BOOST_CHECK_EQUAL(k2.keys(), 1u);

  BOOST_REQUIRE_NO_THROW(key_type k3(i)); // Check constructor w/ 2nd key.
  key_type k3(i);
  BOOST_CHECK_EQUAL(k3.key2(), i);
  BOOST_CHECK_EQUAL(k3.keys(), 2u);

  BOOST_REQUIRE_NO_THROW(key_type k4(k)); // Check copy constructor
  key_type k4(k);
  BOOST_CHECK_EQUAL(k4.key1(), o);
  BOOST_CHECK_EQUAL(k4.key2(), i);
  BOOST_CHECK_EQUAL(k4.keys(), 3u);

#ifdef __GXX_EXPERIMENTAL_CXX0X__
  BOOST_REQUIRE_NO_THROW(key_type k5(o, index_type(1,2,3))); // check key2 move constructor.
  key_type k5(o, index_type(1,2,3));
  BOOST_CHECK_EQUAL(k5.key1(), o);
  BOOST_CHECK_EQUAL(k5.key2(), i);
  BOOST_CHECK_EQUAL(k5.keys(), 3u);

  BOOST_REQUIRE_NO_THROW(key_type k6(ordinal_type(10), i)); // check key1 move constructor.
  key_type k6(ordinal_type(10), i);
  BOOST_CHECK_EQUAL(k6.key1(), o);
  BOOST_CHECK_EQUAL(k6.key2(), i);
  BOOST_CHECK_EQUAL(k6.keys(), 3u);

  BOOST_REQUIRE_NO_THROW(key_type k7(ordinal_type(10), index_type(1,2,3))); // check both keys move constructor.
  key_type k7(ordinal_type(10), index_type(1,2,3));
  BOOST_CHECK_EQUAL(k7.key1(), o);
  BOOST_CHECK_EQUAL(k7.key2(), i);
  BOOST_CHECK_EQUAL(k7.keys(), 3u);

  BOOST_REQUIRE_NO_THROW(key_type k8(ordinal_type(10))); // check move constructor for just key1.
  key_type k8(ordinal_type(10));
  BOOST_CHECK_EQUAL(k8.key1(), o);
  BOOST_CHECK_EQUAL(k8.keys(), 1u);

  BOOST_REQUIRE_NO_THROW(key_type k9(index_type(1,2,3))); // check move constructor for just key2.
  key_type k9(index_type(1,2,3));
  BOOST_CHECK_EQUAL(k9.key2(), i);
  BOOST_CHECK_EQUAL(k9.keys(), 2u);

  BOOST_REQUIRE_NO_THROW(key_type k10(key_type(o, i))); // check key move constructor.
  key_type k10(key_type(o, i));
  BOOST_CHECK_EQUAL(k10.key1(), o);
  BOOST_CHECK_EQUAL(k10.key2(), i);
  BOOST_CHECK_EQUAL(k10.keys(), 3u);
#endif // __GXX_EXPERIMENTAL_CXX0X__

}

BOOST_AUTO_TEST_CASE( set )
{
  k.set(o, i);
  BOOST_CHECK_EQUAL(k.key1(), o);
  BOOST_CHECK_EQUAL(k.key2(), i);
  BOOST_CHECK_EQUAL(k.keys(), 3u);

  k.set(o);
  BOOST_CHECK_EQUAL(k.key1(), o);
  BOOST_CHECK_EQUAL(k.keys(), 1u);

  k.set(i);
  BOOST_CHECK_EQUAL(k.key2(), i);
  BOOST_CHECK_EQUAL(k.keys(), 2u);

#ifdef __GXX_EXPERIMENTAL_CXX0X__
  k.set(o, index_type(1,2,3));
  BOOST_CHECK_EQUAL(k.key1(), o);
  BOOST_CHECK_EQUAL(k.key2(), i);
  BOOST_CHECK_EQUAL(k.keys(), 3u);

  k.set(ordinal_type(10), i);
  BOOST_CHECK_EQUAL(k.key1(), o);
  BOOST_CHECK_EQUAL(k.key2(), i);
  BOOST_CHECK_EQUAL(k.keys(), 3u);

  k.set(ordinal_type(10), index_type(1,2,3));
  BOOST_CHECK_EQUAL(k.key1(), o);
  BOOST_CHECK_EQUAL(k.key2(), i);
  BOOST_CHECK_EQUAL(k.keys(), 3u);

  k.set(ordinal_type(10));
  BOOST_CHECK_EQUAL(k.key1(), o);
  BOOST_CHECK_EQUAL(k.keys(), 1u);

  k.set(index_type(1,2,3));
  BOOST_CHECK_EQUAL(k.key2(), i);
  BOOST_CHECK_EQUAL(k.keys(), 2u);

#endif // __GXX_EXPERIMENTAL_CXX0X__
}

BOOST_AUTO_TEST_CASE( conversion )
{
  ordinal_type o1;
  o1 = k;
  BOOST_CHECK_EQUAL(o1, o);
  ordinal_type o2(k);
  BOOST_CHECK_EQUAL(o2, o);
  BOOST_CHECK_EQUAL(ordinal(k), o);

  index_type i1;
  i1 = k;
  BOOST_CHECK_EQUAL(i1, i);
//  index_type i2(k);
//  BOOST_CHECK_EQUAL(i2, i);
//  BOOST_CHECK_EQUAL(index(k), i);
}

BOOST_AUTO_TEST_SUITE_END()
