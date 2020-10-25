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
 *  tile_op_scal_mult.cpp
 *  May 7, 2013
 *
 */

#include "TiledArray/tile_op/mult.h"
#include "range_fixture.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;
using TiledArray::detail::ScalMult;

struct ScalMultFixture : public RangeFixture {
  ScalMultFixture()
      : a(RangeFixture::r), b(RangeFixture::r), c(), perm({2, 0, 1}) {
    GlobalFixture::world->srand(27);
    for (std::size_t i = 0ul; i < r.volume(); ++i) {
      a[i] = GlobalFixture::world->rand() / 101;
      b[i] = GlobalFixture::world->rand() / 101;
    }
  }

  ~ScalMultFixture() {}

  Tensor<int> a;
  Tensor<int> b;
  Tensor<int> c;
  Permutation perm;

};  // ScalMultFixture

BOOST_FIXTURE_TEST_SUITE(tile_op_scal_mult_suite, ScalMultFixture,
                         TA_UT_SKIP_IF_DISTRIBUTED)

BOOST_AUTO_TEST_CASE(constructor) {
  // Check that the constructors can be called without throwing exceptions
  BOOST_CHECK_NO_THROW(
      (ScalMult<Tensor<int>, Tensor<int>, Tensor<int>, int, false, false>(7)));
  BOOST_CHECK_NO_THROW(
      (ScalMult<Tensor<int>, Tensor<int>, Tensor<int>, int, true, false>(7)));
  BOOST_CHECK_NO_THROW(
      (ScalMult<Tensor<int>, Tensor<int>, Tensor<int>, int, false, true>(7)));
  BOOST_CHECK_NO_THROW(
      (ScalMult<Tensor<int>, Tensor<int>, Tensor<int>, int, true, true>(7)));
}

BOOST_AUTO_TEST_CASE(binary_scale_mult) {
  ScalMult<Tensor<int>, Tensor<int>, Tensor<int>, int, false, false> mult_op(7);

  // Store the multiplication of a and b in c
  BOOST_CHECK_NO_THROW(c = mult_op(a, b));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[i], 7 * (a[i] * b[i]));
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_mult_perm) {
  ScalMult<Tensor<int>, Tensor<int>, Tensor<int>, int, false, false> mult_op(7);

  // Store the multiplication of a and b in c
  BOOST_CHECK_NO_THROW(c = mult_op(a, b, perm));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm * a.range().idx(i)], 7 * (a[i] * b[i]));
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_mult_consume_left) {
  ScalMult<Tensor<int>, Tensor<int>, Tensor<int>, int, true, false> mult_op(7);
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
    BOOST_CHECK_EQUAL(c[i], 7 * (ax[i] * b[i]));
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_mult_perm_consume_left) {
  ScalMult<Tensor<int>, Tensor<int>, Tensor<int>, int, true, false> mult_op(7);

  // Store the multiplication of a and b in c
  BOOST_CHECK_NO_THROW(c = mult_op(a, b, perm));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm * a.range().idx(i)], 7 * (a[i] * b[i]));
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_mult_consume_right) {
  ScalMult<Tensor<int>, Tensor<int>, Tensor<int>, int, false, true> mult_op(7);
  const Tensor<int> bx = b.clone();

  // Store the multiplication of a and b in c
  BOOST_CHECK_NO_THROW(c = mult_op(a, b));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());
  BOOST_CHECK_EQUAL(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[i], 7 * (a[i] * bx[i]));
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_mult_perm_consume_right) {
  ScalMult<Tensor<int>, Tensor<int>, Tensor<int>, int, false, true> mult_op(7);

  // Store the multiplication of a and b in c
  BOOST_CHECK_NO_THROW(c = mult_op(a, b, perm));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm * a.range().idx(i)], 7 * (a[i] * b[i]));
  }
}

BOOST_AUTO_TEST_SUITE_END()
