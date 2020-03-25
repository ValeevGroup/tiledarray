/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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
 *  block_range.cpp
 *  May 29, 2015
 *
 */

#include "TiledArray/block_range.h"
#include "range_fixture.h"
#include "unit_test_config.h"

struct BlockRangeFixture {
  BlockRangeFixture() {}

  ~BlockRangeFixture() {}

  static const Range r0;
  static const Range r;
};  // BlockRangeFixture

const Range BlockRangeFixture::r0{std::array<int, 3>{{5, 11, 8}}};
const Range BlockRangeFixture::r{std::array<int, 3>{{0, 1, 2}},
                                 std::array<int, 3>{{5, 11, 8}}};

BOOST_FIXTURE_TEST_SUITE(block_range_suite, BlockRangeFixture)

BOOST_AUTO_TEST_CASE(block_zero_lower_bound) {
  BlockRange block_range;

  for (auto lower_it = r0.begin(); lower_it != r0.end(); ++lower_it) {
    const auto lower = *lower_it;
    for (auto upper_it = r0.begin(); upper_it != r0.end(); ++upper_it) {
      auto upper = *upper_it;
      for (unsigned int i = 0u; i < upper.size(); ++i) ++(upper[i]);

      if (std::equal(lower.begin(), lower.end(), upper.begin(),
                     [](std::size_t l, std::size_t r0) { return l < r0; })) {
        // Check that the sub-block is constructed without exceptions
        BOOST_CHECK_NO_THROW(block_range = BlockRange(r0, lower, upper));

        // Check that the data of the block range is correct
        std::size_t volume = 1ul;
        for (unsigned int i = 0u; i < r0.rank(); ++i) {
          // Check that the range data is correct
          BOOST_CHECK_EQUAL(block_range.lobound(i), lower[i]);
          BOOST_CHECK_EQUAL(block_range.upbound(i), upper[i]);
          BOOST_CHECK_EQUAL(block_range.extent(i), upper[i] - lower[i]);
          BOOST_CHECK_EQUAL(block_range.stride(i), r0.stride(i));
          volume *= upper[i] - lower[i];
        }
        // Check for the correct volume
        BOOST_CHECK_EQUAL(block_range.volume(), volume);

        // Check that the subrange ordinal calculation returns the same offset
        // as the original range.
        Range::size_type index = 0ul;
        for (auto it = block_range.begin(); it != block_range.end();
             ++it, ++index) {
          // Check that the ordinal offset returned for an ordianl offset and a
          // coordinate index agree.
          BOOST_CHECK_EQUAL(block_range.ordinal(*it), r0.ordinal(*it));

          // Check that the ordinal function returns the correct offset in the
          // parent index space.
          BOOST_CHECK_EQUAL(block_range.ordinal(index), r0.ordinal(*it));

          // Check that the index returned by idx is correct
          BOOST_CHECK_EQUAL(block_range.idx(index), *it);
        }
      }
#ifdef TA_EXCEPTION_ERROR
      else {
        // Check for excpetion with invalid input
        BOOST_CHECK_THROW(BlockRange(r0, lower, upper), TiledArray::Exception);
      }
#endif  // TA_EXCEPTION_ERROR
    }
  }
}

BOOST_AUTO_TEST_CASE(block) {
  BlockRange block_range;

  for (auto lower_it = r.begin(); lower_it != r.end(); ++lower_it) {
    const auto lower = *lower_it;
    for (auto upper_it = r.begin(); upper_it != r.end(); ++upper_it) {
      auto upper = *upper_it;
      for (unsigned int i = 0u; i < r.rank(); ++i) ++(upper[i]);

      if (std::equal(lower.begin(), lower.end(), upper.begin(),
                     [](std::size_t l, std::size_t r) { return l < r; })) {
        // Check that the sub-block is constructed without exceptions
        BOOST_CHECK_NO_THROW(block_range = BlockRange(r, lower, upper));

        // Check that the data of the block range is correct
        std::size_t volume = 1ul;
        for (unsigned int i = 0u; i < r.rank(); ++i) {
          // Check that the range data is correct
          BOOST_CHECK_EQUAL(block_range.lobound(i), lower[i]);
          BOOST_CHECK_EQUAL(block_range.upbound(i), upper[i]);
          BOOST_CHECK_EQUAL(block_range.extent(i), upper[i] - lower[i]);
          BOOST_CHECK_EQUAL(block_range.stride(i), r.stride(i));
          volume *= upper[i] - lower[i];
        }
        // Check for the correct volume
        BOOST_CHECK_EQUAL(block_range.volume(), volume);

        // Check that the subrange ordinal calculation returns the same
        // offset as the original range.
        Range::size_type index = 0ul;
        for (auto it = block_range.begin(); it != block_range.end();
             ++it, ++index) {
          // Check that the ordinal offset returned for an ordianl offset and a
          // coordinate index agree.
          BOOST_CHECK_EQUAL(block_range.ordinal(*it), r.ordinal(*it));

          // Check that the ordinal function returns the correct offset in the
          // parent index space.
          BOOST_CHECK_EQUAL(block_range.ordinal(index), r.ordinal(*it));
          // Check that the index returned by idx is correct
          BOOST_CHECK_EQUAL(block_range.idx(index), *it);
        }
      }
#ifdef TA_EXCEPTION_ERROR
      else {
        // Check for excpetion with invalid input
        BOOST_CHECK_THROW(BlockRange(r, lower, upper), TiledArray::Exception);
      }
#endif  // TA_EXCEPTION_ERROR
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
