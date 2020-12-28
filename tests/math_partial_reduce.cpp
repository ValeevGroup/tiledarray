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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  math_partial_reduce.cpp
 *  Apr 13, 2014
 *
 */

#include "TiledArray/math/partial_reduce.h"
#include "unit_test_config.h"

using namespace TiledArray;

struct PartialReduceFixture {
  PartialReduceFixture() {
    rand_fill(x, 78);
    rand_fill(y, 27);
    rand_fill(a, 388);
  }

  ~PartialReduceFixture() {}

  template <std::size_t N>
  static void rand_fill(int (&vec)[N], const int seed) {
    GlobalFixture::world->srand(seed);
    for (std::size_t i = 0ul; i < N; ++i)
      vec[i] = GlobalFixture::world->rand() % 101;
  }

  struct Sum {
    typedef void result_type;

    void operator()(int& result, const int arg) const { result += arg; }
  };

  struct MultSum {
    typedef void result_type;

    void operator()(int& result, const int left, const int right) const {
      result += left * right;
    }
  };

  static const std::size_t m = TILEDARRAY_LOOP_UNWIND * 2.5;
  static const std::size_t n = TILEDARRAY_LOOP_UNWIND * 4.5;
  int x[m];
  int y[n];
  int a[m * n];

};  // PartialReduceFixture

BOOST_FIXTURE_TEST_SUITE(math_partial_reduce_suite, PartialReduceFixture,
                         TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(unary_row_reduce) {
  // Create a copy of the result for later use.
  int x_ref[m];
  std::copy(x, x + m, x_ref);

  // Do the partial reduction
  math::row_reduce(m, n, a, x, Sum());

  // Check the result of the partial reduction.
  for (std::size_t i = 0ul; i < m; ++i) {
    int expected = x_ref[i];
    for (std::size_t j = 0ul; j < n; ++j) {
      expected += a[i * n + j];
    }

    BOOST_CHECK_EQUAL(x[i], expected);
  }
}

BOOST_AUTO_TEST_CASE(binary_row_reduce) {
  // Create a copy of the result for later use.
  int x_ref[m];
  std::copy(x, x + m, x_ref);

  // Do the partial reduction
  math::row_reduce(m, n, a, y, x, MultSum());

  // Check the result of the partial reduction.
  for (std::size_t i = 0ul; i < m; ++i) {
    int expected = x_ref[i];
    for (std::size_t j = 0ul; j < n; ++j) {
      expected += a[i * n + j] * y[j];
    }

    BOOST_CHECK_EQUAL(x[i], expected);
  }
}

BOOST_AUTO_TEST_CASE(unary_col_reduce) {
  // Create a copy of the result for later use.
  int y_ref[n];
  std::copy(y, y + n, y_ref);

  // Do the partial reduction
  math::col_reduce(m, n, a, y, Sum());

  // Check the result of the partial reduction.
  for (std::size_t j = 0ul; j < n; ++j) {
    int expected = y_ref[j];
    for (std::size_t i = 0ul; i < m; ++i) {
      expected += a[i * n + j];
    }

    BOOST_CHECK_EQUAL(y[j], expected);
  }
}

BOOST_AUTO_TEST_CASE(binary_col_reduce) {
  // Create a copy of the result for later use.
  int y_ref[n];
  std::copy(y, y + n, y_ref);

  // Do the partial reduction
  math::col_reduce(m, n, a, x, y, MultSum());

  // Check the result of the partial reduction.
  for (std::size_t j = 0ul; j < n; ++j) {
    int expected = y_ref[j];
    for (std::size_t i = 0ul; i < m; ++i) {
      expected += a[i * n + j] * x[i];
    }

    BOOST_CHECK_EQUAL(y[j], expected);
  }
}

BOOST_AUTO_TEST_SUITE_END()
