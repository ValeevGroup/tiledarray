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
 *  block_range.h
 *  May 29, 2015
 *
 */

#ifndef TILEDARRAY_BLOCK_RANGE_H__INCLUDED
#define TILEDARRAY_BLOCK_RANGE_H__INCLUDED

#include <TiledArray/range.h>

namespace TiledArray {

  /// Range that references a subblock of another range
  class BlockRange : public Range {
  private:
    using Range::data_;
    using Range::offset_;
    using Range::volume_;
    using Range::rank_;

    Range::size_type block_offset_ = 0ul;

  public:

    BlockRange() : Range() { }

    BlockRange(const BlockRange& other) :
      Range(static_cast<const Range&>(other))
    { }

    BlockRange(BlockRange&& other) :
      Range(static_cast<Range&&>(other))
    { }

    template <typename Index>
    BlockRange(const Range& range, const Index& lower_bound,
        const Index& upper_bound) :
        Range()
    {
      TA_ASSERT(range.rank());
      // Check for valid lower and upper bounds
      TA_ASSERT(std::equal(lower_bound.begin(), lower_bound.end(), range.start(),
          [](const size_type l, const size_type r) { return l >= r; }));
      TA_ASSERT(std::equal(upper_bound.begin(), upper_bound.end(), lower_bound.begin(),
          [](const size_type l, const size_type r) { return l > r; }));
      TA_ASSERT(std::equal(upper_bound.begin(), upper_bound.end(), range.finish(),
          [](const size_type l, const size_type r) { return l <= r; }));

      // Initialize the block range data members
      data_ = new size_type[range.rank() << 2];
      offset_ = range.offset();
      volume_ = 1ul;
      rank_ = range.rank();
      block_offset_ = 0ul;

      // Construct temp pointers
      const size_type* restrict const range_stride = range.weight();
      const auto* restrict const lower_bound_ptr = detail::data(lower_bound);
      const auto* restrict const upper_bound_ptr = detail::data(upper_bound);
      size_type* restrict const lower  = data_;
      size_type* restrict const upper  = lower + rank_;
      size_type* restrict const extent = upper + rank_;
      size_type* restrict const stride = extent + rank_;

      // Compute range data
      for(int i = int(rank_) - 1; i >= 0; --i) {
        // Check input dimensions
        TA_ASSERT(lower_bound[i] >= 0ul);
        TA_ASSERT(lower_bound[i] < upper_bound[i]);

        // Compute data for element i of lower, upper, and extent
        const auto lower_bound_i = lower_bound_ptr[i];
        const auto upper_bound_i = upper_bound_ptr[i];
        const size_type range_stride_i = range_stride[i];
        const auto extent_i = upper_bound_i - lower_bound_i;

        // Set the block range data
        lower[i]       = lower_bound_i;
        upper[i]       = upper_bound_i;
        extent[i]      = extent_i;
        stride[i]      = range_stride_i;
        block_offset_ += lower_bound_i * range_stride_i;
        volume_       *= extent_i;
      }
    }

    ~BlockRange() = default;

    BlockRange& operator=(const BlockRange& other) {
      Range::operator=(other);
      block_offset_ = other.block_offset_;
      return *this;
    }

    BlockRange& operator=(BlockRange&& other) {
      Range::operator=(std::move(other));
      block_offset_ = other.block_offset_;
      other.block_offset_ = 0ul;
      return *this;
    }

    /// calculate the ordinal index of \c i

    /// Convert a coordinate index to an ordinal index.
    /// \tparam Index A coordinate index type (array type)
    /// \param index The index to be converted to an ordinal index
    /// \return The ordinal index of \c index
    /// \throw When \c index is not included in this range.
    template <typename Index,
        enable_if_t<! std::is_integral<Index>::value>* = nullptr>
    size_type ord(const Index& index) const {
      return Range::ord(index);
    }

    template <typename... Index,
        enable_if_t<(sizeof...(Index) > 1ul)>* = nullptr>
    size_type ord(const Index&... index) const {
      return Range::ord(index...);
    }

    /// calculate the coordinate index of the ordinal index, \c index.

    /// Convert an ordinal index to a coordinate index.
    /// \param index Ordinal index
    /// \return The index of the ordinal index
    /// \throw TiledArray::Exception When \c index is not included in this range
    /// \throw std::bad_alloc When memory allocation fails
    size_type ord(size_type index) const {
      // Check that index is contained by range.
      TA_ASSERT(includes(index));

      // Construct result coordinate index object and allocate its memory.
      size_type result = 0ul;

      // Get pointers to the data
      const size_type * restrict const lower = data_;
      const size_type * restrict const size = data_ + rank_ + rank_;
      const size_type * restrict const stride = size + rank_;

      // Compute the coordinate index of o in range.
      for(int i = int(rank_) - 1; i >= 0; --i) {
        const size_type lower_i = lower[i];
        const size_type size_i = size[i];
        const size_type stride_i = stride[i];

        // Compute result index element i
        result += (index % size_i) * stride_i;
        index /= size_i;
      }

      return result + block_offset_ - offset_;
    }

    /// Resize of block range is not supported
    template <typename Index>
    BlockRange& resize(const Index&, const Index&) {
      // This function is here to shadow the base class resize function
      TA_EXCEPTION("BlockRange::resize() is not supported");
      return *this;
    }

    void swap(BlockRange& other) {
      Range::swap(other);
      std::swap(block_offset_, other.block_offset_);
    }
  }; // BlockRange


} // namespace TiledArray

#endif // TILEDARRAY_BLOCK_RANGE_H__INCLUDED
