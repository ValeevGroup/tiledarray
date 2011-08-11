#include "TiledArray/dense_storage.h"
#include "unit_test_config.h"
#include <world/bufar.h>

using namespace TiledArray;

struct DenseStorageFixture {
  typedef TiledArray::DenseStorage<int> Storage;
  typedef Storage::size_type size_type;

  DenseStorageFixture() : t(10) {
    for(size_type i = 0; i < 10; ++i)
      t[i] = i;
  }

  Storage t;
};

BOOST_FIXTURE_TEST_SUITE( dense_storage_suite , DenseStorageFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  // check default constructor
  {
    BOOST_REQUIRE_NO_THROW(Storage t0);
    Storage t0;
    BOOST_CHECK_EQUAL(t0.size(), 0u);
    BOOST_CHECK_EQUAL(t0.volume(), 0u);
    BOOST_CHECK_THROW(t0.at(0), Exception);
    BOOST_CHECK_EQUAL(t0.begin(), t0.end());
  }

  // check copy constructor
  {
    BOOST_REQUIRE_NO_THROW(Storage tc(t));
    Storage tc(t);
    BOOST_CHECK_EQUAL(tc.size(), t.size());
    BOOST_CHECK_EQUAL(tc.volume(), t.volume());
    BOOST_CHECK_EQUAL_COLLECTIONS(tc.begin(), tc.end(), t.begin(), t.end());
  }

  // check constructing with a range
  {
    BOOST_REQUIRE_NO_THROW(Storage t1(10));
    Storage t1(10);
    BOOST_CHECK_EQUAL(t1.size(), 10u);
    BOOST_CHECK_EQUAL(t1.volume(), 10u);
    for(Storage::const_iterator it = t1.begin(); it != t1.end(); ++it)
      BOOST_CHECK_EQUAL(*it, int());
  }

  // check constructing with a range and initial value.
  {
    BOOST_REQUIRE_NO_THROW(Storage t1(10, 1));
    Storage t1(10, 1);
    BOOST_CHECK_EQUAL(t1.size(), 10u);
    BOOST_CHECK_EQUAL(t1.volume(), 10);
    for(Storage::const_iterator it = t1.begin(); it != t1.end(); ++it)
      BOOST_CHECK_EQUAL(*it, 1);
  }

  // check constructing with range and iterators.
  {
    BOOST_REQUIRE_NO_THROW(Storage t3(10, t.begin()));
    Storage t3(10, t.begin());
    BOOST_CHECK_EQUAL(t3.size(), 10);
    BOOST_CHECK_EQUAL(t3.volume(), 10);
    BOOST_CHECK_EQUAL_COLLECTIONS(t3.begin(), t3.end(), t.begin(), t.end());
  }
}

BOOST_AUTO_TEST_CASE( size_accessor )
{
  BOOST_CHECK_EQUAL(t.size(), 10);    // check size accessor
  BOOST_CHECK_EQUAL(t.volume(), 10);  // check volume accessor
}

BOOST_AUTO_TEST_CASE( element_access )
{
  for(int i = 0; i < 10; ++i) {
    // check at()
    BOOST_CHECK_EQUAL(t.at(i), i);
    // check operator[]
    BOOST_CHECK_EQUAL(t[i], i);

    BOOST_CHECK((& t.at(i)) == (& t[i]) );
  }


  // check out of range error
  BOOST_CHECK_THROW(t.at(10), Exception);
#ifndef NDEBUG
  BOOST_CHECK_THROW(t[10], Exception);
#endif
}

BOOST_AUTO_TEST_CASE( iteration )
{
  int i = 0;
  for(Storage::const_iterator it = t.begin(); it != t.end(); ++it, ++i)
    BOOST_CHECK_EQUAL(*it, i);

  Storage t1(t);
  Storage::iterator it1 = t1.begin();
  *it1 = 1;

  // check iterator assignment
  BOOST_CHECK_EQUAL(*it1, 1);
  BOOST_CHECK_EQUAL(t1.at(0), 1);
  Storage t2;
  BOOST_CHECK_EQUAL(t2.begin(), t2.end());
}

BOOST_AUTO_TEST_CASE( element_assignment )
{
  // verify preassignment conditions
  BOOST_CHECK_NE(t.at(0), 2);
  // check that assignment returns itself.
  BOOST_CHECK_EQUAL(t.at(0) = 2, 2);
  // check for correct assignment.
  BOOST_CHECK_EQUAL(t.at(0), 2);

  // verify preassignment conditions
  BOOST_CHECK_NE(t[1], 2);
  // check that assignment returns itself.
  BOOST_CHECK_EQUAL(t[1] = 2, 2) ;
  // check for correct assignment.
  BOOST_CHECK_EQUAL(t[1], 2);
}

BOOST_AUTO_TEST_CASE( assignment_operator )
{
  Storage t1;
  BOOST_CHECK_EQUAL(t1.size(), 0);
  BOOST_CHECK_EQUAL(t.size(), 10);
  BOOST_CHECK_EQUAL(t1.begin(), t1.end());

  t1 = t;
  BOOST_CHECK_EQUAL(t1.size(), t.size());
  BOOST_CHECK_EQUAL_COLLECTIONS(t1.begin(), t1.end(), t.begin(), t.end());
  BOOST_CHECK_NE(t1.begin(), t.begin());
}

BOOST_AUTO_TEST_CASE( plus_assignment_operator )
{
  Storage t1(10, 1);
  t += t1;
  for(size_type i = 0; i < 10; ++ i)
    BOOST_CHECK_EQUAL(t[i], i + 1);
}

BOOST_AUTO_TEST_CASE( plus_assignment_value_operator )
{
  t += 1;
  for(size_type i = 0; i < 10; ++ i)
    BOOST_CHECK_EQUAL(t[i], i + 1);
}

BOOST_AUTO_TEST_CASE( minus_assignment_operator )
{
  t += 2;
  Storage t1(10, 1);
  t -= t1;
  for(size_type i = 0; i < 10; ++ i)
    BOOST_CHECK_EQUAL(t[i], i + 1);
}

BOOST_AUTO_TEST_CASE( minus_assignment_value_operator )
{
  t += 2;

  t -= 1;
  for(size_type i = 0; i < 10; ++ i)
    BOOST_CHECK_EQUAL(t[i], i + 1);
}

BOOST_AUTO_TEST_CASE( scale_assignment_operator )
{
  t *= 2;
  for(size_type i = 0; i < 10; ++ i)
    BOOST_CHECK_EQUAL(t[i], i * 2);
}

BOOST_AUTO_TEST_CASE( serialize )
{

  unsigned char buf[4*(sizeof(Storage::size_type)+(sizeof(Storage::value_type)*10))];
  madness::archive::BufferOutputArchive oar(buf,sizeof(buf));
  t.store(oar);
  std::size_t nbyte = oar.size();
  BOOST_CHECK_GT(oar.size(), 0u);

  // Deserialize 2 pointers from a buffer
  madness::archive::BufferInputArchive iar(buf,nbyte);
  Storage t1;
  t1.load(iar);
  iar.close();

  BOOST_CHECK_EQUAL(t1.size(), t.size());
  BOOST_CHECK_EQUAL_COLLECTIONS(t1.begin(), t1.end(), t.begin(), t.end());
}

BOOST_AUTO_TEST_CASE( swap )
{
  Storage t1;
  Storage::const_iterator t1_begin = t1.begin();
  Storage::const_iterator t1_end = t1.end();

  Storage::const_iterator t_begin = t.begin();
  Storage::const_iterator t_end = t.end();

  t.swap(t1);

  BOOST_CHECK_EQUAL(t1_begin, t.begin());
  BOOST_CHECK_EQUAL(t1_end, t.end());
  BOOST_CHECK_EQUAL(t_begin, t1.begin());
  BOOST_CHECK_EQUAL(t_end, t1.end());
}

BOOST_AUTO_TEST_SUITE_END()

