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

#include <TiledArray/util/eigen.h>
#include <boost/range/combine.hpp>
#ifdef TILEDARRAY_HAS_RANGEV3
#include <range/v3/view/zip.hpp>
#endif

#include "TiledArray/block_range.h"
#include "range_fixture.h"
#include "unit_test_config.h"

struct BlockRangeFixture {
  BlockRangeFixture() {}

  ~BlockRangeFixture() {}

  static const Range r0;
  static const Range r;
};  // BlockRangeFixture

const Range BlockRangeFixture::r0(std::array<int, 3>{{5, 11, 8}});
const Range BlockRangeFixture::r(std::array<int, 3>{{0, 1, 2}},
                                 std::array<int, 3>{{5, 11, 8}});

BOOST_FIXTURE_TEST_SUITE(block_range_suite, BlockRangeFixture,
                         TA_UT_LABEL_SERIAL)

const auto target_count = 20;

BOOST_AUTO_TEST_CASE(block_zero_lower_bound) {
  BlockRange block_range;

  auto count_valid = 0;
  auto count_invalid = 0;
  auto skip = []() { return GlobalFixture::world->rand() % 20 > 9; };

  // loop over all possible subblocks, skipping randomly, until target_count
  // valid+invalid block ranges have been considered
  for (auto lower_it = r0.begin(); lower_it != r0.end(); ++lower_it) {
    const auto lower = *lower_it;
    for (auto upper_it = r0.begin(); upper_it != r0.end(); ++upper_it) {
      if (skip()) continue;
      if (count_valid == target_count && count_invalid == target_count) {
        goto end;
      }

      auto upper = *upper_it;
      for (unsigned int i = 0u; i < upper.size(); ++i) ++(upper[i]);

      if (std::equal(lower.begin(), lower.end(), upper.begin(),
                     [](std::size_t l, std::size_t r0) { return l < r0; })) {
        if (count_valid == target_count) continue;
        ++count_valid;

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
      } else {
        if (count_invalid == target_count) continue;
        ++count_invalid;

        // Check for exception with invalid input
        BOOST_CHECK_THROW(BlockRange(r0, lower, upper), TiledArray::Exception);
      }
    }
  }
end:;
}

BOOST_AUTO_TEST_CASE(block) {
  BlockRange block_range;

  auto count_valid = 0;
  auto count_invalid = 0;
  auto skip = []() { return GlobalFixture::world->rand() % 20 > 9; };

  // loop over all possible subblocks, skipping randomly, until target_count
  // valid+invalid block ranges have been considered
  for (auto lower_it = r.begin(); lower_it != r.end(); ++lower_it) {
    const auto lower = *lower_it;
    for (auto upper_it = r.begin(); upper_it != r.end(); ++upper_it) {
      if (skip()) continue;
      if (count_valid == target_count && count_invalid == target_count) {
        goto end;
      }

      auto upper = *upper_it;
      for (unsigned int i = 0u; i < r.rank(); ++i) ++(upper[i]);

      if (std::equal(lower.begin(), lower.end(), upper.begin(),
                     [](std::size_t l, std::size_t r) { return l < r; })) {
        if (count_valid == target_count) continue;
        ++count_valid;

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

        // check that can also construct using sequence of bound pairs
        std::vector<std::pair<std::size_t, std::size_t>> bounds;
        bounds.reserve(r.rank());
        for (unsigned int i = 0u; i < r.rank(); ++i) {
          bounds.emplace_back(lower[i], upper[i]);
        }
        BlockRange br2;
        BOOST_CHECK_NO_THROW(br2 = BlockRange(r, bounds));
        BOOST_CHECK_EQUAL(br2, block_range);

        // test the rest of ctors
        {
          Range r(10, 10, 10);
          std::vector<size_t> lobounds = {0, 1, 2};
          std::vector<size_t> upbounds = {4, 6, 8};
          BlockRange bref(r, lobounds, upbounds);

          // using vector of pairs
          std::vector<std::pair<size_t, size_t>> vpbounds{
              {0, 4}, {1, 6}, {2, 8}};
          BOOST_CHECK_NO_THROW(BlockRange br0(r, vpbounds));
          BlockRange br0(r, vpbounds);
          BOOST_CHECK_EQUAL(br0, bref);

          // using initializer_list of pairs
          BOOST_CHECK_NO_THROW(BlockRange br0a(
              r, {std::make_pair(0, 4), std::pair{1, 6}, std::pair(2, 8)}));
          BlockRange br0a(
              r, {std::make_pair(0, 4), std::pair{1, 6}, std::pair(2, 8)});
          BOOST_CHECK_EQUAL(br0a, bref);

          // using vector of tuples
          std::vector<std::tuple<size_t, size_t>> vtbounds{
              {0, 4}, {1, 6}, {2, 8}};
          BOOST_CHECK_NO_THROW(BlockRange br1(r, vtbounds));
          BlockRange br1(r, vpbounds);
          BOOST_CHECK_EQUAL(br1, bref);

          // using initializer_list of tuples
          BOOST_CHECK_NO_THROW(BlockRange br1a(
              r, {std::make_tuple(0, 4), std::tuple{1, 6}, std::tuple(2, 8)}));
          BlockRange br1a(
              r, {std::make_tuple(0, 4), std::tuple{1, 6}, std::tuple(2, 8)});
          BOOST_CHECK_EQUAL(br1a, bref);

          // using zipped ranges of bounds (using Boost.Range)
          // need to #include <boost/range/combine.hpp>
          BOOST_CHECK_NO_THROW(
              BlockRange br2(r, boost::combine(lobounds, upbounds)));
          BlockRange br2(r, boost::combine(lobounds, upbounds));
          BOOST_CHECK_EQUAL(br2, bref);

#ifdef TILEDARRAY_HAS_RANGEV3
          // using zipped ranges of bounds (using Ranges-V3)
          // need to #include <range/v3/view/zip.hpp>
          BOOST_CHECK_NO_THROW(
              BlockRange br3(r, ranges::views::zip(lobounds, upbounds)));
          BlockRange br3(r, ranges::views::zip(lobounds, upbounds));
          BOOST_CHECK_EQUAL(br3, bref);
#endif

          // using nested initializer_list
          BOOST_CHECK_NO_THROW(BlockRange br4(r, {{0, 4}, {1, 6}, {2, 8}}));
          BlockRange br4(r, {{0, 4}, {1, 6}, {2, 8}});
          BOOST_CHECK_EQUAL(br4, bref);

          // using Eigen
          {
            using TiledArray::eigen::iv;

            BOOST_CHECK_NO_THROW(BlockRange br5(r, iv(0, 1, 2), iv(4, 6, 8)));
            BlockRange br5(r, iv(0, 1, 2), iv(4, 6, 8));
            BOOST_CHECK_EQUAL(br5, bref);

            BOOST_CHECK_NO_THROW(
                BlockRange br6(r, boost::combine(iv(0, 1, 2), iv(4, 6, 8))));
            BlockRange br6(r, boost::combine(iv(0, 1, 2), iv(4, 6, 8)));
            BOOST_CHECK_EQUAL(br6, bref);
          }
        }
      } else {
        if (count_invalid == target_count) continue;
        ++count_invalid;

        // Check for exception with invalid input
        BOOST_CHECK_THROW(BlockRange(r, lower, upper), TiledArray::Exception);
      }
    }
  }
end:;
}

BOOST_AUTO_TEST_SUITE_END()
