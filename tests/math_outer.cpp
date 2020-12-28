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
 *  math_outer.cpp
 *  Apr 9, 2014
 *
 */

#include "TiledArray/math/outer.h"
#include "unit_test_config.h"

struct OuterFixture {
  OuterFixture()
      : left(TILEDARRAY_LOOP_UNWIND * 4.5, 0),
        right(TILEDARRAY_LOOP_UNWIND * 2.5, 0),
        result(left.size() * right.size(), 1) {
    rand_fill(left, 23);
    rand_fill(right, 42);
    rand_fill(result, 79);
  }

  ~OuterFixture() {}

  static void rand_fill(std::vector<int>& vec, const int seed) {
    GlobalFixture::world->srand(seed);
    for (std::size_t i = 0ul; i < vec.size(); ++i)
      vec[i] = GlobalFixture::world->rand() % 101;
  }

  template <std::size_t N>
  static void rand_fill(int (&vec)[N], const int seed) {
    GlobalFixture::world->srand(seed);
    for (std::size_t i = 0ul; i < N; ++i)
      vec[i] = GlobalFixture::world->rand() % 101;
  }

  static void add_subt(int& result, const int left, const int right) {
    result += left - right;
  }

  static int subt(const int left, const int right) { return left - right; }

  std::vector<int> left;
  std::vector<int> right;
  std::vector<int> result;
};  // OuterFixture

BOOST_FIXTURE_TEST_SUITE(math_outer_suite, OuterFixture,
                         TA_UT_LABEL_SERIAL)

BOOST_AUTO_TEST_CASE(outer_kernel) {
  const std::vector<int> reference = result;

  // Perform outer with the block kernel.
  TiledArray::math::OuterVectorOpUnwindN::outer(&left.front(), &right.front(),
                                                &result.front(), right.size(),
                                                &OuterFixture::add_subt);

  // Check the result
  for (std::size_t i = 0ul; i < TILEDARRAY_LOOP_UNWIND; ++i) {
    for (std::size_t j = 0ul; j < TILEDARRAY_LOOP_UNWIND; ++j) {
      BOOST_CHECK_EQUAL(result[i * right.size() + j],
                        reference[i * right.size() + j] + left[i] - right[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(outer) {
  std::vector<int> reference = result;

  // Perform outer
  TiledArray::math::outer(left.size(), right.size(), &left.front(),
                          &right.front(), &result.front(),
                          &OuterFixture::add_subt);

  // Check the result
  for (std::size_t i = 0ul; i < left.size(); ++i) {
    for (std::size_t j = 0ul; j < right.size(); ++j) {
      BOOST_CHECK_EQUAL(result[i * right.size() + j],
                        reference[i * right.size() + j] + left[i] - right[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(outer_small_left) {
  std::vector<int> reference = result;
  const std::size_t m = TILEDARRAY_LOOP_UNWIND * 0.5;

  // Perform outer with sizes that are smaller than TILEDARRAY_LOOP_UNWIND for
  // right.
  TiledArray::math::outer(m, right.size(), &left.front(), &right.front(),
                          &result.front(), &OuterFixture::add_subt);

  // Check the result
  for (std::size_t i = 0ul; i < m; ++i) {
    for (std::size_t j = 0ul; j < right.size(); ++j) {
      BOOST_CHECK_EQUAL(result[i * right.size() + j],
                        reference[i * right.size() + j] + left[i] - right[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(outer_small_right) {
  std::vector<int> reference = result;
  const std::size_t n = TILEDARRAY_LOOP_UNWIND * 0.75;

  // Perform outer with sizes that are smaller than TILEDARRAY_LOOP_UNWIND for
  // right.
  TiledArray::math::outer(left.size(), n, &left.front(), &right.front(),
                          &result.front(), &OuterFixture::add_subt);

  // Check the result
  for (std::size_t i = 0ul; i < left.size(); ++i) {
    for (std::size_t j = 0ul; j < n; ++j) {
      BOOST_CHECK_EQUAL(result[i * n + j],
                        reference[i * n + j] + left[i] - right[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(outer_small_left_right) {
  std::vector<int> reference = result;
  const std::size_t m = TILEDARRAY_LOOP_UNWIND * 0.75;
  const std::size_t n = TILEDARRAY_LOOP_UNWIND * 0.5;

  // Perform outer with sizes that are smaller than TILEDARRAY_LOOP_UNWIND.
  TiledArray::math::outer(m, n, &left.front(), &right.front(), &result.front(),
                          &OuterFixture::add_subt);

  // Check the result
  for (std::size_t i = 0ul; i < m; ++i) {
    for (std::size_t j = 0ul; j < n; ++j) {
      BOOST_CHECK_EQUAL(result[i * n + j],
                        reference[i * n + j] + left[i] - right[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(outer_fill_kernel) {
  // Perform outer with the block kernel.
  TiledArray::math::OuterVectorOpUnwindN::fill(&left.front(), &right.front(),
                                               &result.front(), right.size(),
                                               &OuterFixture::subt);

  // Check the result
  for (std::size_t i = 0ul; i < TILEDARRAY_LOOP_UNWIND; ++i) {
    for (std::size_t j = 0ul; j < TILEDARRAY_LOOP_UNWIND; ++j) {
      BOOST_CHECK_EQUAL(result[i * right.size() + j], left[i] - right[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(outer_fill) {
  // Perform outer
  TiledArray::math::outer_fill(left.size(), right.size(), &left.front(),
                               &right.front(), &result.front(),
                               &OuterFixture::subt);

  // Check the result
  for (std::size_t i = 0ul; i < left.size(); ++i) {
    for (std::size_t j = 0ul; j < right.size(); ++j) {
      BOOST_CHECK_EQUAL(result[i * right.size() + j], left[i] - right[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(outer_fill_small_left) {
  const std::size_t m = TILEDARRAY_LOOP_UNWIND * 0.5;

  // Perform outer with sizes that are smaller than TILEDARRAY_LOOP_UNWIND for
  // right.
  TiledArray::math::outer_fill(m, right.size(), &left.front(), &right.front(),
                               &result.front(), &OuterFixture::subt);

  // Check the result
  for (std::size_t i = 0ul; i < m; ++i) {
    for (std::size_t j = 0ul; j < right.size(); ++j) {
      BOOST_CHECK_EQUAL(result[i * right.size() + j], left[i] - right[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(outer_fill_small_right) {
  const std::size_t n = TILEDARRAY_LOOP_UNWIND * 0.75;

  // Perform outer with sizes that are smaller than TILEDARRAY_LOOP_UNWIND for
  // right.
  TiledArray::math::outer_fill(left.size(), n, &left.front(), &right.front(),
                               &result.front(), &OuterFixture::subt);

  // Check the result
  for (std::size_t i = 0ul; i < left.size(); ++i) {
    for (std::size_t j = 0ul; j < n; ++j) {
      BOOST_CHECK_EQUAL(result[i * n + j], left[i] - right[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(outer_fill_small_left_right) {
  const std::size_t m = TILEDARRAY_LOOP_UNWIND * 0.75;
  const std::size_t n = TILEDARRAY_LOOP_UNWIND * 0.5;

  // Perform outer with sizes that are smaller than TILEDARRAY_LOOP_UNWIND.
  TiledArray::math::outer_fill(m, n, &left.front(), &right.front(),
                               &result.front(), &OuterFixture::subt);

  // Check the result
  for (std::size_t i = 0ul; i < m; ++i) {
    for (std::size_t j = 0ul; j < n; ++j) {
      BOOST_CHECK_EQUAL(result[i * n + j], left[i] - right[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(outer_and_fill_kernel) {
  const std::vector<int> reference = result;
  std::fill(result.begin(), result.end(), 0);

  // Perform outer with the block kernel.
  TiledArray::math::OuterVectorOpUnwindN::fill(
      &left.front(), &right.front(), &reference.front(), &result.front(),
      right.size(), &OuterFixture::add_subt);

  // Check the result
  for (std::size_t i = 0ul; i < TILEDARRAY_LOOP_UNWIND; ++i) {
    for (std::size_t j = 0ul; j < TILEDARRAY_LOOP_UNWIND; ++j) {
      BOOST_CHECK_EQUAL(result[i * right.size() + j],
                        reference[i * right.size() + j] + left[i] - right[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(outer_and_fill) {
  std::vector<int> reference = result;
  std::fill(result.begin(), result.end(), 0);

  // Perform outer
  TiledArray::math::outer_fill(left.size(), right.size(), &left.front(),
                               &right.front(), &reference.front(),
                               &result.front(), &OuterFixture::add_subt);

  // Check the result
  for (std::size_t i = 0ul; i < left.size(); ++i) {
    for (std::size_t j = 0ul; j < right.size(); ++j) {
      BOOST_CHECK_EQUAL(result[i * right.size() + j],
                        reference[i * right.size() + j] + left[i] - right[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(outer_and_fill_small_left) {
  std::vector<int> reference = result;
  std::fill(result.begin(), result.end(), 0);
  const std::size_t m = TILEDARRAY_LOOP_UNWIND * 0.5;

  // Perform outer with sizes that are smaller than TILEDARRAY_LOOP_UNWIND for
  // right.
  TiledArray::math::outer_fill(m, right.size(), &left.front(), &right.front(),
                               &reference.front(), &result.front(),
                               &OuterFixture::add_subt);

  // Check the result
  for (std::size_t i = 0ul; i < m; ++i) {
    for (std::size_t j = 0ul; j < right.size(); ++j) {
      BOOST_CHECK_EQUAL(result[i * right.size() + j],
                        reference[i * right.size() + j] + left[i] - right[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(outer_and_fill_small_right) {
  std::vector<int> reference = result;
  std::fill(result.begin(), result.end(), 0);
  const std::size_t n = TILEDARRAY_LOOP_UNWIND * 0.75;

  // Perform outer with sizes that are smaller than TILEDARRAY_LOOP_UNWIND for
  // right.
  TiledArray::math::outer_fill(left.size(), n, &left.front(), &right.front(),
                               &reference.front(), &result.front(),
                               &OuterFixture::add_subt);

  // Check the result
  for (std::size_t i = 0ul; i < left.size(); ++i) {
    for (std::size_t j = 0ul; j < n; ++j) {
      BOOST_CHECK_EQUAL(result[i * n + j],
                        reference[i * n + j] + left[i] - right[j]);
    }
  }
}

BOOST_AUTO_TEST_CASE(outer_and_fill_small_left_right) {
  std::vector<int> reference = result;
  std::fill(result.begin(), result.end(), 0);
  const std::size_t m = TILEDARRAY_LOOP_UNWIND * 0.75;
  const std::size_t n = TILEDARRAY_LOOP_UNWIND * 0.5;

  // Perform outer with sizes that are smaller than TILEDARRAY_LOOP_UNWIND.
  TiledArray::math::outer_fill(m, n, &left.front(), &right.front(),
                               &reference.front(), &result.front(),
                               &OuterFixture::add_subt);

  // Check the result
  for (std::size_t i = 0ul; i < m; ++i) {
    for (std::size_t j = 0ul; j < n; ++j) {
      BOOST_CHECK_EQUAL(result[i * n + j],
                        reference[i * n + j] + left[i] - right[j]);
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
