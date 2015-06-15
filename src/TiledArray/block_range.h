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


    template <typename Index>
    void init(const Range& range, const Index& lower_bound, const Index& upper_bound) {
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
      const auto* restrict const range_stride = range.weight();
      const auto* restrict const lower_bound_ptr = detail::data(lower_bound);
      const auto* restrict const upper_bound_ptr = detail::data(upper_bound);
      auto* restrict const lower  = data_;
      auto* restrict const upper  = lower + rank_;
      auto* restrict const extent = upper + rank_;
      auto* restrict const stride = extent + rank_;

      // Compute range data
      for(int i = int(rank_) - 1; i >= 0; --i) {
        // Compute data for element i of lower, upper, and extent
        const auto lower_bound_i = lower_bound_ptr[i];
        const auto upper_bound_i = upper_bound_ptr[i];
        const auto range_stride_i = range_stride[i];
        const auto extent_i = upper_bound_i - lower_bound_i;

        // Check input dimensions
        TA_ASSERT(lower_bound_i >= range.start()[i]);
        TA_ASSERT(lower_bound_i < upper_bound_i);
        TA_ASSERT(upper_bound_i <= range.finish()[i]);

        // Set the block range data
        lower[i]       = lower_bound_i;
        upper[i]       = upper_bound_i;
        extent[i]      = extent_i;
        stride[i]      = range_stride_i;
        block_offset_ += lower_bound_i * range_stride_i;
        volume_       *= extent_i;
      }
    }

  public:

    // Compiler generated functions
    BlockRange() = default;
    BlockRange(const BlockRange&) = default;
    BlockRange(BlockRange&&) = default;
    ~BlockRange() = default;
    BlockRange& operator=(const BlockRange&) = default;
    BlockRange& operator=(BlockRange&&) = default;

    template <typename Index>
    BlockRange(const Range& range, const Index& lower_bound,
        const Index& upper_bound) :
        Range()
    {
      init(range, lower_bound, upper_bound);
    }


    BlockRange(const Range& range, const std::initializer_list<size_type>& lower_bound,
        const std::initializer_list<size_type>& upper_bound) :
      Range()
    {
      init(range, lower_bound, upper_bound);
    }


    /// calculate the ordinal index of \c i

    /// Convert a coordinate index to an ordinal index.
    /// \tparam Index A coordinate index type (array type)
    /// \param index The index to be converted to an ordinal index
    /// \return The ordinal index of \c index
    /// \throw When \c index is not included in this range.
    template <typename Index,
        typename std::enable_if<! std::is_integral<Index>::value>::type* = nullptr>
    size_type ordinal(const Index& index) const {
      return Range::ordinal(index);
    }

    template <typename... Index,
        typename std::enable_if<(sizeof...(Index) > 1ul)>::type* = nullptr>
    size_type ordinal(const Index&... index) const {
      return Range::ordinal(index...);
    }

    /// calculate the coordinate index of the ordinal index, \c index.

    /// Convert an ordinal index to a coordinate index.
    /// \param index Ordinal index
    /// \return The index of the ordinal index
    /// \throw TiledArray::Exception When \c index is not included in this range
    /// \throw std::bad_alloc When memory allocation fails
    size_type ordinal(size_type index) const {
      // Check that index is contained by range.
      TA_ASSERT(includes(index));

      // Construct result coordinate index object and allocate its memory.
      size_type result = 0ul;

      // Get pointers to the data
      const size_type * restrict const size = data_ + rank_ + rank_;
      const size_type * restrict const stride = size + rank_;

      // Compute the coordinate index of o in range.
      for(int i = int(rank_) - 1; i >= 0; --i) {
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


    /// Shift the lower and upper bound of this range

    /// \warning This function is here to shadow the base class inplace_shift
    /// function, and disable it.
    template <typename Index>
    Range_& inplace_shift(const Index&) {
      TA_EXCEPTION("BlockRange::inplace_shift() is not supported");
      return *this;
    }

    /// Shift the lower and upper bound of this range

    /// \warning This function is here to shadow the base class shift function,
    /// and disable it.
    template <typename Index>
    Range_ shift(const Index&) {
      TA_EXCEPTION("BlockRange::shift() is not supported");
      return *this;
    }

    void swap(BlockRange& other) {
      Range::swap(other);
      std::swap(block_offset_, other.block_offset_);
    }

    /// Serialization Block range
    template <typename Archive>
    void serialize(const Archive& ar) const {
      Range::serialize(ar);
      ar & block_offset_;
    }
  }; // BlockRange


} // namespace TiledArray

#endif // TILEDARRAY_BLOCK_RANGE_H__INCLUDED
