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
#include "unit_test_config.h"
#include "range_fixture.h"

struct BlockRangeFixture : public RangeFixture {

  BlockRangeFixture() { }

  ~BlockRangeFixture() { }

}; // BlockRangeFixture

BOOST_FIXTURE_TEST_SUITE( block_range_suite, BlockRangeFixture )


BOOST_AUTO_TEST_CASE( block )
{
  BlockRange block_range;

  for(auto lower_it = r.begin(); lower_it != r.end(); ++lower_it) {
    const auto lower = *lower_it;
    for(auto upper_it = r.begin(); upper_it != r.end(); ++upper_it) {
      std::vector<std::size_t> upper = *upper_it;
      for(unsigned int i = 0u; i < upper.size(); ++i)
        ++(upper[i]);

      if(std::equal(lower.begin(), lower.end(), upper.begin(),
          [] (std::size_t l, std::size_t r) { return l < r; })) {

        // Check that the sub-block is constructed without exceptions
        BOOST_CHECK_NO_THROW(block_range = BlockRange(r, lower,upper));

        // Check that the data of the block range is correct
        std::size_t volume = 1ul;
        for(unsigned int i = 0u; i < r.dim(); ++i) {
          BOOST_CHECK_EQUAL(block_range.start()[i], lower[i]);
          BOOST_CHECK_EQUAL(block_range.finish()[i], upper[i]);
          BOOST_CHECK_EQUAL(block_range.size()[i], upper[i] - lower[i]);
          BOOST_CHECK_EQUAL(block_range.weight()[i], r.weight()[i]);
          volume *= upper[i] - lower[i];
        }
        BOOST_CHECK_EQUAL(block_range.volume(), volume);

        // Check that the subrange ordinal calculation returns the same offset
        // as the original range.
        Range::size_type i = 0ul;
        for(auto it = block_range.begin(); it != block_range.end(); ++it, ++i) {
          BOOST_CHECK_EQUAL(block_range.ord(*it), r.ord(*it));
          BOOST_CHECK_EQUAL(block_range.ord(i), r.ord(*it));
        }
      }
#ifdef TA_EXCEPTION_ERROR
      else {
        // Check for excpetion with invalid input
        BOOST_CHECK_THROW(BlockRange(r, lower,upper), TiledArray::Exception);
      }
#endif // TA_EXCEPTION_ERROR

    }
  }


  Range x(p2, p5);
  for(Range::const_iterator lower_it = x.begin(); lower_it != x.end(); ++lower_it) {
    const auto lower = *lower_it;
    for(Range::const_iterator upper_it = x.begin(); upper_it != x.end(); ++upper_it) {
      std::vector<std::size_t> upper = *upper_it;
      for(unsigned int i = 0u; i < x.dim(); ++i)
        ++(upper[i]);

      if(std::equal(lower.begin(), lower.end(), upper.begin(),
          [] (std::size_t l, std::size_t r) { return l < r; })) {

        // Check that the sub-block is constructed without exceptions
        BOOST_CHECK_NO_THROW(block_range = BlockRange(x, lower,upper));

        // Check that the data of the block range is correct
        std::size_t volume = 1ul;
        for(unsigned int i = 0u; i < x.dim(); ++i) {
          BOOST_CHECK_EQUAL(block_range.start()[i], lower[i]);
          BOOST_CHECK_EQUAL(block_range.finish()[i], upper[i]);
          BOOST_CHECK_EQUAL(block_range.size()[i], upper[i] - lower[i]);
          BOOST_CHECK_EQUAL(block_range.weight()[i], x.weight()[i]);
          volume *= upper[i] - lower[i];
        }
        BOOST_CHECK_EQUAL(block_range.volume(), volume);

        // Check that the subrange ordinal calculation returns the same
        // offset as the original range.
        Range::size_type i = 0ul;
        for(Range::const_iterator it = block_range.begin(); it != block_range.end(); ++it, ++i) {
          BOOST_CHECK_EQUAL(block_range.ord(*it), x.ord(*it));
          BOOST_CHECK_EQUAL(block_range.ord(i), x.ord(*it));
        }

      }
#ifdef TA_EXCEPTION_ERROR
      else {
        // Check for excpetion with invalid input
        BOOST_CHECK_THROW(BlockRange(x, lower,upper), TiledArray::Exception);
      }
#endif // TA_EXCEPTION_ERROR

    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
