/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2025  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Ajay Melekamburath
 *  Department of Chemistry, Virginia Tech
 *  Aug 02, 2025
 */

#include <TiledArray/device/um_tensor.h>
#include "global_fixture.h"
#include "unit_test_config.h"

using namespace TiledArray;

struct TensorUMFixture {
  typedef UMTensor<int> TensorN;
  typedef TensorN::value_type value_type;
  typedef TensorN::range_type::index index;
  typedef TensorN::size_type size_type;
  typedef TensorN::range_type::index_view_type* index_view_type;
  typedef TensorN::range_type range_type;

  const range_type r;

  TensorUMFixture() : r(make_range(81)), t(r, TensorN::nbatches{1}) {
    rand_fill(18, t.size(), t.data());
  }

  ~TensorUMFixture() {}

  static range_type make_range(const int seed) {
    GlobalFixture::world->srand(seed);
    std::array<std::size_t, GlobalFixture::dim> start, finish;

    for (unsigned int i = 0ul; i < GlobalFixture::dim; ++i) {
      start[i] = GlobalFixture::world->rand() % 10;
      finish[i] = GlobalFixture::world->rand() % 8 + start[i] + 2;
    }

    return range_type(start, finish);
  }

  static void rand_fill(const int seed, const size_type n, int* const data) {
    GlobalFixture::world->srand(seed);
    for (size_type i = 0ul; i < n; ++i)
      data[i] = GlobalFixture::world->rand() % 42;
  }

  template <typename T>
  static void rand_fill(const int seed, const size_type n,
                        std::complex<T>* const data) {
    GlobalFixture::world->srand(seed);
    for (size_type i = 0ul; i < n; ++i)
      data[i] = std::complex<T>(GlobalFixture::world->rand() % 42,
                                GlobalFixture::world->rand() % 42);
  }

  static TensorN make_tensor(const int range_seed, const int data_seed) {
    TensorN tensor(make_range(range_seed));
    rand_fill(data_seed, tensor.size(), tensor.data());
    return tensor;
  }

  //  // make permutation definition object
  //  static Permutation make_perm() {
  //    std::array<unsigned int, GlobalFixture::dim> temp;
  //    for(std::size_t i = 0; i < temp.size(); ++i)
  //      temp[i] = i + 1;
  //
  //    temp.back() = 0;
  //
  //    return Permutation(temp.begin(), temp.end());
  //  }

  TensorN t;
};

BOOST_FIXTURE_TEST_SUITE(ta_tensor_um_suite, TensorUMFixture,
                         TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(default_constructor) {
  // check constructor
  BOOST_REQUIRE_NO_THROW(TensorN x);
  TensorN x;

  BOOST_CHECK(x.empty());

  // Check that range data is correct
  BOOST_CHECK_EQUAL(x.size(), 0ul);
  BOOST_CHECK_EQUAL(x.range().volume(), 0ul);

  // Check the element data
  BOOST_CHECK_EQUAL(x.begin(), x.end());
  BOOST_CHECK_EQUAL(const_cast<const TensorN&>(x).begin(),
                    const_cast<const TensorN&>(x).end());
}

BOOST_AUTO_TEST_CASE(range_constructor) {
  BOOST_REQUIRE_NO_THROW(TensorN x(r));
  TensorN x(r);

  BOOST_CHECK(!x.empty());

  // Check that range data is correct
  BOOST_CHECK_NE(x.data(), static_cast<int*>(NULL));
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(),
                                  const_cast<const TensorN&>(x).end()),
                    r.volume());
}

BOOST_AUTO_TEST_CASE(value_constructor) {
  BOOST_REQUIRE_NO_THROW(TensorN x(r, 8));
  TensorN x(r, 8);

  BOOST_CHECK(!x.empty());

  // Check that range data is correct
  BOOST_CHECK_NE(x.data(), static_cast<int*>(NULL));
  BOOST_CHECK_EQUAL(x.size(), r.volume());
  BOOST_CHECK_EQUAL(x.range(), r);
  BOOST_CHECK_EQUAL(std::distance(x.begin(), x.end()), r.volume());
  BOOST_CHECK_EQUAL(std::distance(const_cast<const TensorN&>(x).begin(),
                                  const_cast<const TensorN&>(x).end()),
                    r.volume());

  for (TensorN::const_iterator it = x.begin(); it != x.end(); ++it)
    BOOST_CHECK_EQUAL(*it, 8);
}

// BOOST_AUTO_TEST_CASE( copy_constructor ) {
//  // check constructor
//  BOOST_REQUIRE_NO_THROW(TensorN tc(t));
//  TensorN tc(t);
//
//  BOOST_CHECK_EQUAL(tc.empty(), t.empty());
//
//  // Check that range data is correct
//  BOOST_CHECK_EQUAL(tc.data(), t.data());
//  BOOST_CHECK_EQUAL(tc.size(), t.size());
//  BOOST_CHECK_EQUAL(tc.range(), t.range());
//  BOOST_CHECK_EQUAL(tc.begin(), t.begin());
//  BOOST_CHECK_EQUAL(tc.end(), t.end());
//  BOOST_CHECK_EQUAL(const_cast<const TensorN&>(tc).begin(), const_cast<const
//  TensorN&>(t).begin()); BOOST_CHECK_EQUAL(const_cast<const
//  TensorN&>(tc).end(), const_cast<const TensorN&>(t).end());
//  BOOST_CHECK_EQUAL_COLLECTIONS(tc.begin(), tc.end(), t.begin(), t.end());
//}

BOOST_AUTO_TEST_CASE(range_accessor) {
  BOOST_CHECK_EQUAL_COLLECTIONS(
      t.range().lobound_data(), t.range().lobound_data() + t.range().rank(),
      r.lobound_data(), r.lobound_data() + r.rank());  // check start accessor
  BOOST_CHECK_EQUAL_COLLECTIONS(
      t.range().upbound_data(), t.range().upbound_data() + t.range().rank(),
      r.upbound_data(), r.upbound_data() + r.rank());  // check finish accessor
  BOOST_CHECK_EQUAL_COLLECTIONS(
      t.range().extent_data(), t.range().extent_data() + t.range().rank(),
      r.extent_data(), r.extent_data() + r.rank());  // check size accessor
  BOOST_CHECK_EQUAL_COLLECTIONS(
      t.range().stride_data(), t.range().stride_data() + t.range().rank(),
      r.stride_data(), r.stride_data() + r.rank());   // check weight accessor
  BOOST_CHECK_EQUAL(t.range().volume(), r.volume());  // check volume accessor
  BOOST_CHECK_EQUAL(t.range(), r);                    // check range accessof
}

BOOST_AUTO_TEST_CASE(element_access) {
  // check operator[] with array coordinate index and ordinal index
  for (std::size_t i = 0ul; i < t.size(); ++i) {
    BOOST_CHECK_LT(t[i], 42);
    BOOST_CHECK_EQUAL(t[r.idx(i)], t[i]);
  }

  // check access via call operator, if implemented
#if defined(TILEDARRAY_HAS_VARIADIC_TEMPLATES)
#if TEST_DIM == 3u
  BOOST_CHECK_EQUAL(t(0, 0, 0), t[0]);
#endif
#endif
}

BOOST_AUTO_TEST_CASE(iteration) {
  BOOST_CHECK_EQUAL(t.begin(), const_cast<const TensorN&>(t).begin());
  BOOST_CHECK_EQUAL(t.end(), const_cast<const TensorN&>(t).end());

  for (TensorN::iterator it = t.begin(); it != t.end(); ++it) {
    BOOST_CHECK_LT(*it, 42);
    BOOST_CHECK_EQUAL(*it, t[std::distance(t.begin(), it)]);
  }

  // check iterator assignment
  TensorN::iterator it = t.begin();
  BOOST_CHECK_NE(t[0], 88);
  *it = 88;
  BOOST_CHECK_EQUAL(t[0], 88);

  // Check that the iterators of an empty tensor are equal
  TensorN t2;
  BOOST_CHECK_EQUAL(t2.begin(), t2.end());
}

BOOST_AUTO_TEST_CASE(element_assignment) {
  // verify preassignment conditions
  BOOST_CHECK_NE(t[1], 2);
  // check that assignment returns itself.
  BOOST_CHECK_EQUAL(t[1] = 2, 2);
  // check for correct assignment.
  BOOST_CHECK_EQUAL(t[1], 2);
}

BOOST_AUTO_TEST_CASE(serialization) {
  std::size_t buf_size = (t.range().volume() * sizeof(int) +
                          sizeof(size_type) * (r.rank() * 4 + 2)) *
                         2;
  unsigned char* buf = new unsigned char[buf_size];
  madness::archive::BufferOutputArchive oar(buf, buf_size);
  BOOST_REQUIRE_NO_THROW(oar & t);
  std::size_t nbyte = oar.size();
  oar.close();

  TensorN ts;
  madness::archive::BufferInputArchive iar(buf, nbyte);
  BOOST_REQUIRE_NO_THROW(iar & ts);
  iar.close();

  delete[] buf;

  BOOST_CHECK_EQUAL(t.range(), ts.range());
  BOOST_CHECK_EQUAL_COLLECTIONS(t.begin(), t.end(), ts.begin(), ts.end());
}

BOOST_AUTO_TEST_SUITE_END()
