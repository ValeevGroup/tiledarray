/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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
 *  math_transpose.cpp
 *  Jun 10, 2014
 *
 */

#include "TiledArray/math/transpose.h"
#include "tiledarray.h"
#include "unit_test_config.h"

struct TransposeFixture {
  TransposeFixture() {}

  ~TransposeFixture() {}

};  // TransposeFixture

BOOST_FIXTURE_TEST_SUITE(transpose_suite, TransposeFixture,
                         TA_UT_SKIP_IF_DISTRIBUTED)

BOOST_AUTO_TEST_CASE(copy) {
  const std::size_t m = 20;
  const std::size_t n = 25;
  const std::size_t mn = m * n;

  int* a = new int[mn];
  int* b = new int[mn];

  GlobalFixture::world->srand(1764);
  for (std::size_t i = 0ul; i < mn; ++i)
    a[i] = GlobalFixture::world->rand() % 42;

  const auto no_op = [](const int& a) -> const int& { return a; };
  const auto copy_op = [](int* b, const int a) { *b = a; };

  for (std::size_t x = 1ul; x < m; ++x) {
    for (std::size_t y = 1ul; y < n; ++y) {
      std::fill_n(b, mn, 0);

      TiledArray::math::transpose(no_op, copy_op, x, y, m, b, n, a);

      for (std::size_t i = 0ul; i < m; ++i) {
        for (std::size_t j = 0ul; j < n; ++j) {
          if ((i < x) && (j < y)) {
            BOOST_CHECK_EQUAL(b[j * m + i], a[i * n + j]);
          } else {
            BOOST_CHECK_EQUAL(b[j * m + i], 0);
          }
        }
      }
    }
  }

  delete[] a;
  delete[] b;
}

BOOST_AUTO_TEST_CASE(unary) {
  const std::size_t m = 20;
  const std::size_t n = 25;
  const std::size_t mn = m * n;

  int* a = new int[mn];
  int* b = new int[mn];

  GlobalFixture::world->srand(1764);
  for (std::size_t i = 0ul; i < mn; ++i)
    a[i] = GlobalFixture::world->rand() % 42;

  const auto op = [](const int arg) { return arg * 3; };
  const auto copy_op = [](int* b, const int a) { *b = a; };

  for (std::size_t x = 1ul; x < m; ++x) {
    for (std::size_t y = 1ul; y < n; ++y) {
      std::fill_n(b, mn, 0);

      TiledArray::math::transpose(op, copy_op, x, y, m, b, n, a);

      for (std::size_t i = 0ul; i < m; ++i) {
        for (std::size_t j = 0ul; j < n; ++j) {
          if ((i < x) && (j < y)) {
            BOOST_CHECK_EQUAL(b[j * m + i], op(a[i * n + j]));
          } else {
            BOOST_CHECK_EQUAL(b[j * m + i], 0);
          }
        }
      }
    }
  }

  delete[] a;
  delete[] b;
}

BOOST_AUTO_TEST_CASE(binary) {
  const std::size_t m = 20;
  const std::size_t n = 25;
  const std::size_t mn = m * n;

  int* a = new int[mn];
  int* b = new int[mn];
  int* c = new int[mn];

  GlobalFixture::world->srand(1764);
  for (std::size_t i = 0ul; i < mn; ++i)
    a[i] = GlobalFixture::world->rand() % 42;
  for (std::size_t i = 0ul; i < mn; ++i)
    b[i] = GlobalFixture::world->rand() % 42;

  const auto op = [](const int l, const int r) { return l - r; };
  const auto copy_op = [](int* b, const int a) { *b = a; };

  for (std::size_t x = 1ul; x < m; ++x) {
    for (std::size_t y = 1ul; y < n; ++y) {
      std::fill_n(c, mn, 0);

      TiledArray::math::transpose(op, copy_op, x, y, m, c, n, a, b);

      for (std::size_t i = 0ul; i < m; ++i) {
        for (std::size_t j = 0ul; j < n; ++j) {
          if ((i < x) && (j < y)) {
            BOOST_CHECK_EQUAL(c[j * m + i], op(a[i * n + j], b[i * n + j]));
          } else {
            BOOST_CHECK_EQUAL(c[j * m + i], 0);
          }
        }
      }
    }
  }

  delete[] a;
  delete[] b;
  delete[] c;
}
BOOST_AUTO_TEST_SUITE_END()
