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
 *  tile_op_scal_subt.cpp
 *  May 8, 2013
 *
 */

#include "TiledArray/tile_op/subt.h"
#include "range_fixture.h"
#include "tiledarray.h"
#include "unit_test_config.h"

using namespace TiledArray;
using TiledArray::detail::ScalSubt;

struct ScalSubtFixture : public RangeFixture {
  ScalSubtFixture()
      : a(RangeFixture::r), b(RangeFixture::r), c(), perm({2, 0, 1}) {
    GlobalFixture::world->srand(27);
    for (std::size_t i = 0ul; i < r.volume(); ++i) {
      a[i] = GlobalFixture::world->rand() / 101;
      b[i] = GlobalFixture::world->rand() / 101;
    }
  }

  ~ScalSubtFixture() {}

  Tensor<int> a;
  Tensor<int> b;
  Tensor<int> c;
  Permutation perm;

};  // SubtFixture

BOOST_FIXTURE_TEST_SUITE(tile_op_scal_subt_suite, ScalSubtFixture,
                         TA_UT_SKIP_IF_DISTRIBUTED)

BOOST_AUTO_TEST_CASE(constructor) {
  // Check that the constructors can be called without throwing exceptions
  BOOST_CHECK_NO_THROW(
      (ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, false, false>(7)));
  BOOST_CHECK_NO_THROW(
      (ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, true, false>(7)));
  BOOST_CHECK_NO_THROW(
      (ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, false, true>(7)));
  BOOST_CHECK_NO_THROW(
      (ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, true, true>(7)));
}

BOOST_AUTO_TEST_CASE(binary_scale_subt) {
  ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, false, false> subt_op(7);

  // Store the difference of a and b in c
  BOOST_CHECK_NO_THROW(c = subt_op(a, b));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[i], 7 * (a[i] - b[i]));
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_subt_left_zero) {
  ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, false, false> subt_op(7);

  // Store the difference of 0 and b in c
  BOOST_CHECK_NO_THROW(c = subt_op(ZeroTensor(), b));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[i], -7 * b[i]);
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_subt_right_zero) {
  ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, false, false> subt_op(7);

  // Store the difference of a and 0 in c
  BOOST_CHECK_NO_THROW(c = subt_op(a, ZeroTensor()));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[i], 7 * a[i]);
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_subt_perm) {
  ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, false, false> subt_op(7);

  // Store the difference of a and b in c
  BOOST_CHECK_NO_THROW(c = subt_op(a, b, perm));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm * a.range().idx(i)], 7 * (a[i] - b[i]));
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_subt_left_zero_perm) {
  ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, false, false> subt_op(7);

  // Store the difference of 0 and b in c
  BOOST_CHECK_NO_THROW(c = subt_op(ZeroTensor(), b, perm));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), b.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm * b.range().idx(i)], -7 * b[i]);
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_subt_right_zero_perm) {
  ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, false, false> subt_op(7);

  // Store the difference of a and 0 in c
  BOOST_CHECK_NO_THROW(c = subt_op(a, ZeroTensor(), perm));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm * a.range().idx(i)], 7 * a[i]);
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_subt_consume_left) {
  ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, true, false> subt_op(7);
  const Tensor<int> ax(a.range(), a.begin());

  // Store the difference of a and b in c
  BOOST_CHECK_NO_THROW(c = subt_op(a, b));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_EQUAL(c.data(), a.data());
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[i], 7 * (ax[i] - b[i]));
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_subt_left_zero_consume_left) {
  ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, true, false> subt_op(7);

  // Store the difference of 0 and b in c
  BOOST_CHECK_NO_THROW(c = subt_op(ZeroTensor(), b));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[i], -7 * b[i]);
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_subt_right_zero_consume_left) {
  ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, true, false> subt_op(7);
  const Tensor<int> ax(a.range(), a.begin());

  // Store the difference of a and 0 in c
  BOOST_CHECK_NO_THROW(c = subt_op(a, ZeroTensor()));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_EQUAL(c.data(), a.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[i], 7 * ax[i]);
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_subt_perm_consume_left) {
  ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, true, false> subt_op(7);

  // Store the difference of a and b in c
  BOOST_CHECK_NO_THROW(c = subt_op(a, b, perm));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm * a.range().idx(i)], 7 * (a[i] - b[i]));
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_subt_left_zero_perm_consume_left) {
  ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, true, false> subt_op(7);

  // Store the difference of 0 and b in c
  BOOST_CHECK_NO_THROW(c = subt_op(ZeroTensor(), b, perm));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), b.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm * b.range().idx(i)], -7 * b[i]);
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_subt_right_zero_perm_consume_left) {
  ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, true, false> subt_op(7);

  // Store the difference of a and 0 in c
  BOOST_CHECK_NO_THROW(c = subt_op(a, ZeroTensor(), perm));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm * a.range().idx(i)], 7 * a[i]);
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_subt_consume_right) {
  ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, false, true> subt_op(7);
  const Tensor<int> bx(b.range(), b.begin());

  // Store the difference of a and b in c
  BOOST_CHECK_NO_THROW(c = subt_op(a, b));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());
  BOOST_CHECK_EQUAL(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[i], 7 * (a[i] - bx[i]));
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_subt_left_zero_consume_right) {
  ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, false, true> subt_op(7);
  const Tensor<int> bx(b.range(), b.begin());

  // Store the difference of 0 and b in c
  BOOST_CHECK_NO_THROW(c = subt_op(ZeroTensor(), b));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_EQUAL(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[i], -7 * bx[i]);
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_subt_right_zero_consume_right) {
  ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, false, true> subt_op(7);

  // Store the difference of a and 0 in c
  BOOST_CHECK_NO_THROW(c = subt_op(a, ZeroTensor()));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[i], 7 * a[i]);
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_subt_perm_consume_right) {
  ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, false, true> subt_op(7);

  // Store the difference of a and b in c
  BOOST_CHECK_NO_THROW(c = subt_op(a, b, perm));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm * a.range().idx(i)], 7 * (a[i] - b[i]));
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_subt_left_zero_perm_consume_right) {
  ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, false, true> subt_op(7);

  // Store the difference of 0 and b in c
  BOOST_CHECK_NO_THROW(c = subt_op(ZeroTensor(), b, perm));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), b.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), b.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm * b.range().idx(i)], -7 * b[i]);
  }
}

BOOST_AUTO_TEST_CASE(binary_scale_subt_right_zero_perm_consume_right) {
  ScalSubt<Tensor<int>, Tensor<int>, Tensor<int>, int, false, true> subt_op(7);

  // Store the difference of a and 0 in c
  BOOST_CHECK_NO_THROW(c = subt_op(a, ZeroTensor(), perm));

  // Check that the result range is correct
  BOOST_CHECK_EQUAL(c.range(), a.range());

  // Check that a nor b were consumed
  BOOST_CHECK_NE(c.data(), a.data());

  // Check that the data in the new tile is correct
  for (std::size_t i = 0ul; i < r.volume(); ++i) {
    BOOST_CHECK_EQUAL(c[perm * a.range().idx(i)], 7 * a[i]);
  }
}

BOOST_AUTO_TEST_SUITE_END()
