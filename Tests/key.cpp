#include "TiledArray/key.h"
#include "TiledArray/coordinates.h"
#include "unit_test_config.h"

using namespace TiledArray;
using namespace TiledArray::detail;

struct IndexTag { };

struct KeyFixture {
  typedef std::size_t ordinal_type;
  typedef ArrayCoordinate<std::size_t, 3, IndexTag > index_type;
  typedef Key<ordinal_type, index_type> key_type;

  KeyFixture() : o(10ul), ohi(11ul), olow(9ul), i(1,2,3), ihi(1,2,4), ilow(1,2,2),
      k(o, i), ko(o), kohi(ohi), kolow(olow), ki(i), kihi(ihi), kilow(ilow),
      khi(ohi, ihi), klow(olow, ilow)
  { }

  ~KeyFixture() { }

  ordinal_type ordinal(ordinal_type o) {
    return o;
  }

  index_type index(index_type i) {
    return i;
  }

  const ordinal_type o;
  const ordinal_type ohi;
  const ordinal_type olow;
  const index_type i;
  const index_type ihi;
  const index_type ilow;
  key_type k;
  const key_type k0;
  const key_type ko;
  const key_type kohi;
  const key_type kolow;
  const key_type ki;
  const key_type kihi;
  const key_type kilow;
  const key_type khi;
  const key_type klow;
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
}

BOOST_AUTO_TEST_CASE( assign )
{
  k = o;
  BOOST_CHECK_EQUAL(k.key1(), o);
  BOOST_CHECK_EQUAL(k.keys(), 1u);

  k = i;
  BOOST_CHECK_EQUAL(k.key2(), i);
  BOOST_CHECK_EQUAL(k.keys(), 2u);

  k = khi;
  BOOST_CHECK_EQUAL(k.key1(), ohi);
  BOOST_CHECK_EQUAL(k.key2(), ihi);
  BOOST_CHECK_EQUAL(k.keys(), 3u);

  k = ordinal_type(10ul);
  BOOST_CHECK_EQUAL(k.key1(), o);
  BOOST_CHECK_EQUAL(k.keys(), 1u);

  k = index_type(1,2,3);
  BOOST_CHECK_EQUAL(k.key2(), i);
  BOOST_CHECK_EQUAL(k.keys(), 2u);

  k = key_type(o, i);
  BOOST_CHECK_EQUAL(k.key1(), o);
  BOOST_CHECK_EQUAL(k.key2(), i);
  BOOST_CHECK_EQUAL(k.keys(), 3u);
}

BOOST_AUTO_TEST_CASE( set )
{
  k.set(o, i); // Check setting both keys.
  BOOST_CHECK_EQUAL(k.key1(), o);
  BOOST_CHECK_EQUAL(k.key2(), i);
  BOOST_CHECK_EQUAL(k.keys(), 3u);

  k.set(o); // Check setting key1.
  BOOST_CHECK_EQUAL(k.key1(), o);
  BOOST_CHECK_EQUAL(k.keys(), 1u);

  k.set(i); // Check setting key2.
  BOOST_CHECK_EQUAL(k.key2(), i);
  BOOST_CHECK_EQUAL(k.keys(), 2u);
}

BOOST_AUTO_TEST_CASE( conversion )
{
  ordinal_type o1; // convert to key 1.
  o1 = k;
  BOOST_CHECK_EQUAL(o1, o);
  ordinal_type o2(k);
  BOOST_CHECK_EQUAL(o2, o);
  BOOST_CHECK_EQUAL(ordinal(k), o);
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(ordinal(ki), std::runtime_error); // check that type conversions throw correctly.
#endif // TA_EXCEPTION_ERROR

  index_type i1; // convert to key 2.
  i1 = k;
  BOOST_CHECK_EQUAL(i1, i);
  // this does not work because key2_type's constructors can also accept key1_type.
  index_type i2 = k;
  BOOST_CHECK_EQUAL(i2, i);
  BOOST_CHECK_EQUAL(index(k), i);
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(index(ko), std::runtime_error); // check that type conversions throw correctly.
#endif // TA_EXCEPTION_ERROR
}

BOOST_AUTO_TEST_CASE( equal_comp )
{
  key_type k1(k);
  BOOST_CHECK(k == k1); // check bool operator ==(const Key<Key1, Key2>&, const Key<Key1, Key2>&);
  BOOST_CHECK(! (k == khi));
  BOOST_CHECK(k == ko);
  BOOST_CHECK(! (k == kohi));
  BOOST_CHECK(k == ki);
  BOOST_CHECK(! (k == kihi));
  BOOST_CHECK(! (ko == ki));

  BOOST_CHECK(k == o); // check bool operator ==(const Key<Key1, Key2>&, const Key1&);
  BOOST_CHECK(! (k == ohi));

  BOOST_CHECK(o == k); // check bool operator ==(const Key1&, const Key<Key1, Key2>&);
  BOOST_CHECK(! (ohi == k));

  BOOST_CHECK(k == i); // check bool operator ==(const Key<Key1, Key2>&, const Key2&);
  BOOST_CHECK(! (k == ihi));

  BOOST_CHECK(i == k); // check bool operator ==(const Key2&, const Key<Key1, Key2>&);
  BOOST_CHECK(! (ihi == k));
}

BOOST_AUTO_TEST_CASE( ne_comp )
{
  key_type k1(k);
  BOOST_CHECK(! (k != k1)); // check bool operator !=(const Key<Key1, Key2>&, const Key<Key1, Key2>&);
  BOOST_CHECK(k != khi);
  BOOST_CHECK(! (k != ko));
  BOOST_CHECK(k != kohi);
  BOOST_CHECK(! (k != ki));
  BOOST_CHECK(k != kihi);
  BOOST_CHECK(! (ko != ki));

  BOOST_CHECK(! (k != o)); // check bool operator !=(const Key<Key1, Key2>&, const Key1&);
  BOOST_CHECK(k != ohi);

  BOOST_CHECK(! (o != k)); // check bool operator !=(const Key1&, const Key<Key1, Key2>&);
  BOOST_CHECK(ohi != k);

  BOOST_CHECK(! (k != i)); // check bool operator !=(const Key<Key1, Key2>&, const Key2&);
  BOOST_CHECK(k != ihi);

  BOOST_CHECK(! (i != k)); // check bool operator !=(const Key2&, const Key<Key1, Key2>&);
  BOOST_CHECK(ihi != k);
}

BOOST_AUTO_TEST_CASE( lt_comp )
{
  BOOST_CHECK(klow < k); // check bool operator <(const Key<Key1, Key2>&, const Key<Key1, Key2>&);
  BOOST_CHECK(! (khi < k));
  BOOST_CHECK(kolow < k);
  BOOST_CHECK(! (kohi < k));
  BOOST_CHECK(kilow < k);
  BOOST_CHECK(! (kihi < k));
  BOOST_CHECK(! (ko < ki));

  BOOST_CHECK(klow < o); // check bool operator <(const Key<Key1, Key2>&, const Key1&);
  BOOST_CHECK(! (khi < o));

  BOOST_CHECK(olow < k); // check bool operator <(const Key1&, const Key<Key1, Key2>&);
  BOOST_CHECK(! (ohi < k));

  BOOST_CHECK(klow < i); // check bool operator <(const Key<Key1, Key2>&, const Key2&);
  BOOST_CHECK(! (khi < i));

  BOOST_CHECK(ilow < k); // check bool operator <(const Key2&, const Key<Key1, Key2>&);
  BOOST_CHECK(! (ihi < k));
}

BOOST_AUTO_TEST_CASE( le_comp )
{
  key_type k1(k);
  BOOST_CHECK(klow <= k); // check bool operator <=(const Key<Key1, Key2>&, const Key<Key1, Key2>&);
  BOOST_CHECK(k1 <= k);
  BOOST_CHECK(! (khi <= k));
  BOOST_CHECK(kolow <= k);
  BOOST_CHECK(ko <= k);
  BOOST_CHECK(! (kohi <= k));
  BOOST_CHECK(kilow <= k);
  BOOST_CHECK(ki <= k);
  BOOST_CHECK(! (kihi <= k));
  BOOST_CHECK(! (ko <= ki));

  BOOST_CHECK(klow <= o); // check bool operator <=(const Key<Key1, Key2>&, const Key1&);
  BOOST_CHECK(k <= o);
  BOOST_CHECK(! (khi <= o));

  BOOST_CHECK(olow <= k); // check bool operator <=(const Key1&, const Key<Key1, Key2>&);
  BOOST_CHECK(o <= k);
  BOOST_CHECK(! (ohi <= k));

  BOOST_CHECK(klow <= i); // check bool operator <=(const Key<Key1, Key2>&, const Key2&);
  BOOST_CHECK(k <= i);
  BOOST_CHECK(! (khi <= i));

  BOOST_CHECK(ilow <= k); // check bool operator <=(const Key2&, const Key<Key1, Key2>&);
  BOOST_CHECK(i <= k);
  BOOST_CHECK(! (ihi <= k));
}

BOOST_AUTO_TEST_CASE( gt_comp )
{
  BOOST_CHECK(khi > k); // check bool operator >(const Key<Key1, Key2>&, const Key<Key1, Key2>&);
  BOOST_CHECK(! (klow > k));
  BOOST_CHECK(kohi > k);
  BOOST_CHECK(! (kolow > k));
  BOOST_CHECK(kihi > k);
  BOOST_CHECK(! (kilow > k));
  BOOST_CHECK(! (ko > ki));

  BOOST_CHECK(khi > o); // check bool operator >(const Key<Key1, Key2>&, const Key1&);
  BOOST_CHECK(! (klow > o));

  BOOST_CHECK(ohi > k); // check bool operator >(const Key1&, const Key<Key1, Key2>&);
  BOOST_CHECK(! (olow > k));

  BOOST_CHECK(khi > i); // check bool operator >(const Key<Key1, Key2>&, const Key2&);
  BOOST_CHECK(! (klow > i));

  BOOST_CHECK(ihi > k); // check bool operator >(const Key2&, const Key<Key1, Key2>&);
  BOOST_CHECK(! (ilow > k));
}

BOOST_AUTO_TEST_CASE( ge_comp )
{
  key_type k1(k);
  BOOST_CHECK(khi >= k); // check bool operator >=(const Key<Key1, Key2>&, const Key<Key1, Key2>&);
  BOOST_CHECK(k1 >= k);
  BOOST_CHECK(! (klow >= k));
  BOOST_CHECK(kohi >= k);
  BOOST_CHECK(ko >= k);
  BOOST_CHECK(! (kolow >= k));
  BOOST_CHECK(kihi >= k);
  BOOST_CHECK(ki >= k);
  BOOST_CHECK(! (kilow >= k));
  BOOST_CHECK(! (ko >= ki));

  BOOST_CHECK(khi >= o); // check bool operator >=(const Key<Key1, Key2>&, const Key1&);
  BOOST_CHECK(k >= o);
  BOOST_CHECK(! (klow >= o));

  BOOST_CHECK(ohi >= k); // check bool operator >=(const Key1&, const Key<Key1, Key2>&);
  BOOST_CHECK(o >= k);
  BOOST_CHECK(! (olow >= k));

  BOOST_CHECK(khi >= i); // check bool operator >=(const Key<Key1, Key2>&, const Key2&);
  BOOST_CHECK(k >= i);
  BOOST_CHECK(! (klow >= i));

  BOOST_CHECK(ihi >= k); // check bool operator >=(const Key2&, const Key<Key1, Key2>&);
  BOOST_CHECK(i >= k);
  BOOST_CHECK(! (ilow >= k));
}

BOOST_AUTO_TEST_SUITE_END()
