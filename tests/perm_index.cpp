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
 *  perm_index.cpp
 *  Oct 10, 2014
 *
 */

#include "TiledArray/perm_index.h"
#include "unit_test_config.h"

using namespace TiledArray;
using namespace TiledArray::detail;

struct PermIndexFixture {
  PermIndexFixture()
      : perm({1, 2, 0, 3}), range(start, finish), perm_range(perm * range) {}

  ~PermIndexFixture() {}

  static const std::array<std::size_t, 4> start;
  static const std::array<std::size_t, 4> finish;

  Permutation perm;
  Range range;
  Range perm_range;

};  // PermIndexFixture

const std::array<std::size_t, 4> PermIndexFixture::start = {
    {0ul, 0ul, 0ul, 0ul}};
const std::array<std::size_t, 4> PermIndexFixture::finish = {
    {3ul, 5ul, 7ul, 11ul}};

BOOST_FIXTURE_TEST_SUITE(perm_index_suite, PermIndexFixture,
                         TA_UT_SKIP_IF_DISTRIBUTED)

BOOST_AUTO_TEST_CASE(default_constructor) {
  BOOST_CHECK_NO_THROW(PermIndex x;);
  PermIndex x;

  // Check that the data has not bee initialized
  BOOST_CHECK(!bool(x));
  BOOST_CHECK(!x.data());

  // Check that an exception is thrown when using a default constructed object
  BOOST_CHECK_THROW(x(0), TiledArray::Exception);
}

BOOST_AUTO_TEST_CASE(constructor) {
  BOOST_CHECK_NO_THROW(PermIndex x(range, perm););
  PermIndex x(range, perm);

  // Check for valid permutation object
  BOOST_CHECK(bool(x));
  BOOST_CHECK(x.data());

  // Check for valid permutation data

  BOOST_CHECK_EQUAL(x.dim(), 4);

  const std::size_t* const input_weight_begin = x.data();
  const std::size_t* const input_weight_end = x.data() + x.dim();
  BOOST_CHECK_EQUAL_COLLECTIONS(input_weight_begin, input_weight_end,
                                range.stride_data(),
                                range.stride_data() + range.rank());

  const std::size_t* const output_weight_begin = input_weight_end;
  const std::size_t* const output_weight_end = input_weight_end + x.dim();
  const std::vector<Range::index1_type> inv_result_weight =
      -perm * perm_range.stride_data();
  BOOST_CHECK_EQUAL_COLLECTIONS(output_weight_begin, output_weight_end,
                                inv_result_weight.begin(),
                                inv_result_weight.end());
}

BOOST_AUTO_TEST_CASE(assignment_operator) {
  PermIndex x;

  // Verify initial state of x
  BOOST_CHECK(!bool(x));
  BOOST_CHECK(!x.data());

  // Assign x
  BOOST_CHECK_NO_THROW(x = PermIndex(range, perm););

  // Check for valid permutation object
  BOOST_CHECK(bool(x));
  BOOST_CHECK(x.data());

  // Check for valid permutation data

  BOOST_CHECK_EQUAL(x.dim(), 4);

  const std::size_t* const input_weight_begin = x.data();
  const std::size_t* const input_weight_end = x.data() + x.dim();
  BOOST_CHECK_EQUAL_COLLECTIONS(input_weight_begin, input_weight_end,
                                range.stride_data(),
                                range.stride_data() + range.rank());

  const std::size_t* const output_weight_begin = input_weight_end;
  const std::size_t* const output_weight_end = input_weight_end + x.dim();
  const std::vector<Range::index1_type> inv_result_weight =
      -perm * perm_range.stride_data();
  BOOST_CHECK_EQUAL_COLLECTIONS(output_weight_begin, output_weight_end,
                                inv_result_weight.begin(),
                                inv_result_weight.end());
}

BOOST_AUTO_TEST_CASE(permute_constructor_tensor) {
  std::array<unsigned int, 4> p = {{0, 1, 2, 3}};

  while (std::next_permutation(p.begin(), p.end())) {
    // set new permutation permutation
    perm = Permutation(p.begin(), p.end());
    perm_range = perm * range;

    // Construct the permute index object
    BOOST_CHECK_NO_THROW(PermIndex perm_index(range, perm););
    PermIndex perm_index(range, perm);

    // Check for valid permutation object
    BOOST_CHECK(bool(perm_index));
    BOOST_CHECK(perm_index.data());

    // Check for valid permutation data

    BOOST_CHECK_EQUAL(perm_index.dim(), 4);

    const std::size_t* const input_weight_begin = perm_index.data();
    const std::size_t* const input_weight_end =
        perm_index.data() + perm_index.dim();
    BOOST_CHECK_EQUAL_COLLECTIONS(input_weight_begin, input_weight_end,
                                  range.stride_data(),
                                  range.stride_data() + range.rank());

    const std::size_t* const output_weight_begin = input_weight_end;
    const std::size_t* const output_weight_end =
        input_weight_end + perm_index.dim();
    const std::vector<Range::index1_type> inv_result_weight =
        -perm * perm_range.stride_data();
    BOOST_CHECK_EQUAL_COLLECTIONS(output_weight_begin, output_weight_end,
                                  inv_result_weight.begin(),
                                  inv_result_weight.end());

    for (std::size_t i = 0ul; i < range.volume(); ++i) {
      std::size_t pi = 0;
      BOOST_CHECK_NO_THROW(pi = perm_index(i));

      BOOST_CHECK_EQUAL(pi, perm_range.ordinal(perm * range.idx(i)));
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
