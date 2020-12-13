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
 *  tile_op_scal.cpp
 *  June 20, 2013
 *
 */

#include "TiledArray/tile_op/scal.h"
#include "range_fixture.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;
using TiledArray::detail::Scal;

struct ScalFixture : public RangeFixture {
  ScalFixture() : a(RangeFixture::r), b(), perm({2, 0, 1}) {
    GlobalFixture::world->srand(27);
    for (std::size_t i = 0ul; i < r.volume(); ++i) {
      a[i] = GlobalFixture::world->rand() / 101;
    }
  }

  ~ScalFixture() {}

  TensorI a;
  TensorI b;
  Permutation perm;

};  // ScalFixture

BOOST_FIXTURE_TEST_SUITE(tile_op_scal_neg_suite, ScalFixture,
                         TA_UT_SKIP_IF_DISTRIBUTED)

BOOST_AUTO_TEST_CASE(constructor) {
  // Check that the constructors can be called without throwing exceptions
  BOOST_CHECK_NO_THROW((Scal<TensorI, TensorI, int, false>(7)));
  BOOST_CHECK_NO_THROW((Scal<TensorI, TensorI, int, true>(7)));
}

BOOST_AUTO_TEST_CASE(unary_scale) {
  Scal<TensorI, TensorI, int, false> scal_op(7);

  // Store the sum of a and b in c
  BOOST_CHECK_NO_THROW(b = scal_op(a));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(b.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(b.data(), a.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(b[i], 7 * a[i]);
  }
}

BOOST_AUTO_TEST_CASE(unary_scale_perm) {
  Scal<TensorI, TensorI, int, false> scal_op(7);

  // Store the sum of a and b in c
  BOOST_CHECK_NO_THROW(b = scal_op(a, perm));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(b.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(b.data(), a.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(b[perm * a.range().idx(i)], 7 * a[i]);
  }
}

BOOST_AUTO_TEST_CASE(unary_scale_consume) {
  Scal<TensorI, TensorI, int, true> scal_op(7);
  const TensorI ax(a.range(), a.begin());

  // Store the sum of a and b in c
  BOOST_CHECK_NO_THROW(b = scal_op(a));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(b.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_EQUAL(b.data(), a.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(b[i], 7 * ax[i]);
  }
}

BOOST_AUTO_TEST_CASE(unary_scale_runtime_consume) {
  Scal<TensorI, TensorI, int, false> scal_op(7);
  const TensorI ax(a.range(), a.begin());

  // Store the sum of a and b in c
  BOOST_CHECK_NO_THROW(b = scal_op.consume(a));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(b.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_EQUAL(b.data(), a.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(b[i], 7 * ax[i]);
  }
}

BOOST_AUTO_TEST_CASE(unary_scale_perm_consume) {
  Scal<TensorI, TensorI, int, true> scal_op(7);

  // Store the sum of a and b in c
  BOOST_CHECK_NO_THROW(b = scal_op(a, perm));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(b.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(b.data(), a.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(b[perm * a.range().idx(i)], 7 * a[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END()
