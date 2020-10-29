/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  tile_op_mult.cpp
 *  May 7, 2013
 *
 */

#include "TiledArray/tile_op/mult.h"
#include "range_fixture.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;
using TiledArray::detail::Mult;

struct MultFixture : public RangeFixture {
  MultFixture() : a(RangeFixture::r), b(RangeFixture::r), c(), perm({2, 0, 1}) {
    GlobalFixture::world->srand(27);
    for (std::size_t i = 0ul; i < r.volume(); ++i) {
      a[i] = GlobalFixture::world->rand() / 101;
      b[i] = GlobalFixture::world->rand() / 101;
    }
  }

  ~MultFixture() {}

  Tensor<int> a;
  Tensor<int> b;
  Tensor<int> c;
  Permutation perm;

};  // MultFixture

BOOST_FIXTURE_TEST_SUITE(tile_op_mult_suite, MultFixture,
                         TA_UT_SKIP_IF_DISTRIBUTED)

BOOST_AUTO_TEST_CASE(constructor) {
  // Check that the constructors can be called without throwing exceptions
  BOOST_CHECK_NO_THROW(
      (Mult<Tensor<int>, Tensor<int>, Tensor<int>, false, false>()));
  BOOST_CHECK_NO_THROW(
      (Mult<Tensor<int>, Tensor<int>, Tensor<int>, true, false>()));
  BOOST_CHECK_NO_THROW(
      (Mult<Tensor<int>, Tensor<int>, Tensor<int>, false, true>()));
  BOOST_CHECK_NO_THROW(
      (Mult<Tensor<int>, Tensor<int>, Tensor<int>, true, true>()));
}

BOOST_AUTO_TEST_CASE(binary_mult) {
  Mult<Tensor<int>, Tensor<int>, Tensor<int>, false, false> mult_op;

  // Store the multiplication of a and b in c
  BOOST_CHECK_NO_THROW(c = mult_op(a, b));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[i], a[i] * b[i]);
  }
}

BOOST_AUTO_TEST_CASE(binary_mult_perm) {
  Mult<Tensor<int>, Tensor<int>, Tensor<int>, false, false> mult_op;

  // Store the multiplication of a and b in c
  BOOST_CHECK_NO_THROW(c = mult_op(a, b, perm));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm * a.range().idx(i)], a[i] * b[i]);
  }
}

BOOST_AUTO_TEST_CASE(binary_mult_consume_left) {
  Mult<Tensor<int>, Tensor<int>, Tensor<int>, true, false> mult_op;
  const Tensor<int> ax(a.range(), a.begin());

  // Store the multiplication of a and b in c
  BOOST_CHECK_NO_THROW(c = mult_op(a, b));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_EQUAL(c.data(), a.data());
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[i], ax[i] * b[i]);
  }
}

BOOST_AUTO_TEST_CASE(binary_mult_perm_consume_left) {
  Mult<Tensor<int>, Tensor<int>, Tensor<int>, true, false> mult_op;

  // Store the multiplication of a and b in c
  BOOST_CHECK_NO_THROW(c = mult_op(a, b, perm));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm * a.range().idx(i)], a[i] * b[i]);
  }
}

BOOST_AUTO_TEST_CASE(binary_mult_consume_right) {
  Mult<Tensor<int>, Tensor<int>, Tensor<int>, false, true> mult_op;
  const Tensor<int> bx(b.range(), b.begin());

  // Store the multiplication of a and b in c
  BOOST_CHECK_NO_THROW(c = mult_op(a, b));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());
  BOOST_CHECK_EQUAL(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[i], a[i] * bx[i]);
  }
}

BOOST_AUTO_TEST_CASE(binary_mult_perm_consume_right) {
  Mult<Tensor<int>, Tensor<int>, Tensor<int>, false, true> mult_op;

  // Store the multiplication of a and b in c
  BOOST_CHECK_NO_THROW(c = mult_op(a, b, perm));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm * a.range().idx(i)], a[i] * b[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END()
