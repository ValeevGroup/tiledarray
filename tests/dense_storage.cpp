/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include "TiledArray/dense_storage.h"
#include "unit_test_config.h"
#include "TiledArray/madness.h"

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
#ifdef TA_EXCEPTION_ERROR
    BOOST_CHECK_THROW(t0.at(0), Exception);
#endif
    BOOST_CHECK_EQUAL(t0.begin(), t0.end());
  }

  // check copy constructor
  {
    BOOST_REQUIRE_NO_THROW(Storage tc(t));
    Storage tc(t);
    BOOST_CHECK_EQUAL(tc.size(), t.size());
    BOOST_CHECK_EQUAL_COLLECTIONS(tc.begin(), tc.end(), t.begin(), t.end());
  }

  // check constructing with a range
  {
    BOOST_REQUIRE_NO_THROW(Storage t1(10));
    Storage t1(10);
    BOOST_CHECK_EQUAL(t1.size(), 10u);
//    for(Storage::const_iterator it = t1.begin(); it != t1.end(); ++it)
//      BOOST_CHECK_EQUAL(*it, int());
  }

  // check constructing with a range and initial value.
  {
    BOOST_REQUIRE_NO_THROW(Storage t1(10, 1));
    Storage t1(10, 1);
    BOOST_CHECK_EQUAL(t1.size(), 10ul);
    for(Storage::const_iterator it = t1.begin(); it != t1.end(); ++it)
      BOOST_CHECK_EQUAL(*it, 1);
  }

  // check constructing with range and iterators.
  {
    BOOST_REQUIRE_NO_THROW(Storage t3(10, t.begin()));
    Storage t3(10, t.begin());
    BOOST_CHECK_EQUAL(t3.size(), 10ul);
    BOOST_CHECK_EQUAL_COLLECTIONS(t3.begin(), t3.end(), t.begin(), t.end());
  }
}

BOOST_AUTO_TEST_CASE( transform_constructor )
{
  const std::size_t n = 100;
  std::vector<int> vl;
  std::vector<int> vr;
  vl.reserve(n);
  vr.reserve(n);

  GlobalFixture::world->srand(27);
  for(std::size_t i = 0ul; i < n; ++i) {
    vl.push_back(GlobalFixture::world->rand());
    vr.push_back(GlobalFixture::world->rand());
  }

  // check pair transform constructor
  {
    BOOST_REQUIRE_NO_THROW(Storage t(n, vl.begin(), std::negate<int>()));
    Storage t(n, vl.begin(), std::negate<int>());

    for(std::size_t i = 0ul; i < n; ++i)
      BOOST_CHECK_EQUAL(t[i], -(vl[i]));
  }

  // check pair transform constructor
  {
    BOOST_REQUIRE_NO_THROW(Storage t(n, vl.begin(), vr.begin(), std::plus<int>()));
    Storage t(n, vl.begin(), vr.begin(), std::plus<int>());

    for(std::size_t i = 0ul; i < n; ++i)
      BOOST_CHECK_EQUAL(t[i], vl[i] + vr[i]);
  }
}

BOOST_AUTO_TEST_CASE( size_accessor )
{
  BOOST_CHECK_EQUAL(t.size(), 10ul);    // check size accessor
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
#ifdef TA_EXCEPTION_ERROR
  BOOST_CHECK_THROW(t[10], Exception);
#endif // TA_EXCEPTION_ERROR
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
  BOOST_CHECK_EQUAL(t1.size(), 0ul);
  BOOST_CHECK_EQUAL(t.size(), 10ul);
  BOOST_CHECK_EQUAL(t1.begin(), t1.end());

  t1 = t;
  BOOST_CHECK_EQUAL(t1.size(), t.size());
  BOOST_CHECK_EQUAL_COLLECTIONS(t1.begin(), t1.end(), t.begin(), t.end());
  BOOST_CHECK_NE(t1.begin(), t.begin());
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

