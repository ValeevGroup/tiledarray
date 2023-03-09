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

/// a Range that references a subblock of another Range
class BlockRange : public Range {
 private:
  /// datavec_ contains same data as if this were an ordinary Range, except:
  /// - stride describes the host range
  /// - offset_ corrects ordinals so that the first element in this (sub)Range
  /// has ordinal of zero ...
  /// - block_offset_ is the distance from the first element of this Range to
  /// the first element of the host Range
  using Range::datavec_;
  using Range::offset_;
  using Range::rank_;
  using Range::volume_;

  Range::ordinal_type block_offset_ = 0ul;

  friend inline bool operator==(const BlockRange& r1, const BlockRange& r2);

  template <typename Index1, typename Index2,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2>>>
  void init(const Range& range, const Index1& lower_bound,
            const Index2& upper_bound) {
    TA_ASSERT(range.rank());

    // Initialize the block range data members
    init_datavec(range.rank());
    offset_ = range.offset();
    volume_ = 1ul;
    rank_ = range.rank();
    block_offset_ = 0ul;

    // Construct temp pointers
    const auto* MADNESS_RESTRICT const range_stride = range.stride_data();
    auto* MADNESS_RESTRICT const lower = lobound_data_nc();
    auto* MADNESS_RESTRICT const upper = upbound_data_nc();
    auto* MADNESS_RESTRICT const extent = extent_data_nc();
    auto* MADNESS_RESTRICT const stride = stride_data_nc();

    // initialize bounds and extents
    auto lower_it = std::begin(lower_bound);
    auto upper_it = std::begin(upper_bound);
    auto lower_end = std::end(lower_bound);
    auto upper_end = std::end(upper_bound);
    for (int d = 0; lower_it != lower_end && upper_it != upper_end;
         ++lower_it, ++upper_it, ++d) {
      // Compute data for element i of lower, upper, and extent
      const auto lower_bound_d = *lower_it;
      const auto upper_bound_d = *upper_it;

      // Set the block range data
      lower[d] = lower_bound_d;
      upper[d] = upper_bound_d;
      // Check input dimensions
      TA_ASSERT(lower[d] >= range.lobound(d));
      TA_ASSERT(lower[d] < upper[d]);
      TA_ASSERT(upper[d] <= range.upbound(d));
      extent[d] = upper[d] - lower[d];
      TA_ASSERT(extent[d] ==
                static_cast<index1_type>(upper_bound_d - lower_bound_d));
    }

    // Compute strides, volume, and offset, starting with last (least
    // significant) dimension
    for (int d = int(rank_) - 1; d >= 0; --d) {
      const auto range_stride_d = range_stride[d];
      stride[d] = range_stride_d;
      block_offset_ += lower[d] * range_stride_d;
      volume_ *= extent[d];
    }
  }

  template <typename PairRange,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange>>>
  void init(const Range& range, const PairRange& bounds) {
    TA_ASSERT(range.rank());

    // Initialize the block range data members
    init_datavec(range.rank());
    offset_ = range.offset();
    volume_ = 1ul;
    rank_ = range.rank();
    block_offset_ = 0ul;

    // Construct temp pointers
    const auto* MADNESS_RESTRICT const range_stride = range.stride_data();
    auto* MADNESS_RESTRICT const lower = lobound_data_nc();
    auto* MADNESS_RESTRICT const upper = upbound_data_nc();
    auto* MADNESS_RESTRICT const extent = extent_data_nc();
    auto* MADNESS_RESTRICT const stride = stride_data_nc();

    // Compute range data
    int d = 0;
    for (auto&& bound_d : bounds) {
      // Compute data for element i of lower, upper, and extent
      const auto lower_bound_d = detail::at(bound_d, 0);
      const auto upper_bound_d = detail::at(bound_d, 1);

      lower[d] = lower_bound_d;
      upper[d] = upper_bound_d;
      // Check input dimensions
      TA_ASSERT(lower[d] >= range.lobound(d));
      TA_ASSERT(lower[d] < upper[d]);
      TA_ASSERT(upper[d] <= range.upbound(d));
      extent[d] = upper[d] - lower[d];
      TA_ASSERT(extent[d] ==
                static_cast<index1_type>(upper_bound_d - lower_bound_d));
      ++d;
    }

    // Compute strides, volume, and offset, starting with last (least
    // significant) dimension
    for (int d = int(rank_) - 1; d >= 0; --d) {
      const auto range_stride_d = range_stride[d];
      stride[d] = range_stride_d;
      block_offset_ += lower[d] * range_stride_d;
      volume_ *= extent[d];
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

  // clang-format off
  /// Construct a BlockRange defined by lower and upper bounds

  /// Construct a BlockRange within host Range defined by \p lower_bound and \p upper_bound.
  /// Examples of using this constructor:
  /// \code
  ///   Range r(10, 10, 10);
  ///   std::vector<size_t> lobounds = {0, 1, 2};
  ///   std::vector<size_t> upbounds = {4, 6, 8};
  ///   BlockRange br(r, lobounds, upbounds);
  ///   // or using in-place ctors
  ///   Range br2(r, std::vector<size_t>{0, 1, 2}, std::vector<size_t>{4, 6, 8});
  ///   assert(r == r2);
  /// \endcode
  /// \tparam Index An integral range type
  /// \param range the host Range
  /// \param lower_bound A sequence of lower bounds for each dimension
  /// \param upper_bound A sequence of upper bounds for each dimension
  /// \throw TiledArray::Exception When the size of \p lower_bound is not
  /// equal to that of \p upper_bound.
  /// \throw TiledArray::Exception When `lower_bound[i] >= upper_bound[i]`
  // clang-format on
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<detail::is_integral_range_v<Index1> &&
                                        detail::is_integral_range_v<Index2>>>
  BlockRange(const Range& range, const Index1& lower_bound,
             const Index2& upper_bound)
      : Range() {
    init(range, lower_bound, upper_bound);
  }

  // clang-format off
  /// Construct a BlockRange defined by lower and upper bounds

  /// Construct a BlockRange within host Range defined by \p lower_bound and \p upper_bound.
  /// Examples of using this constructor:
  /// \code
  ///   Range r(10, 10, 10);
  ///   BlockRange br(r, {0, 1, 2}, {4, 6, 8});
  /// \endcode
  /// \tparam Index1 An integral type
  /// \tparam Index2 An integral type
  /// \param range the host Range
  /// \param lower_bound An initializer list of lower bounds for each dimension
  /// \param upper_bound An initializer list of upper bounds for each dimension
  /// \throw TiledArray::Exception When the size of \p lower_bound is not
  /// equal to that of \p upper_bound.
  /// \throw TiledArray::Exception When `lower_bound[i] >= upper_bound[i]`
  // clang-format on
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2>>>
  BlockRange(const Range& range,
             const std::initializer_list<Index1>& lower_bound,
             const std::initializer_list<Index2>& upper_bound)
      : Range() {
    init(range, lower_bound, upper_bound);
  }

  // clang-format off
  /// Construct Range defined by a range of {lower,upper} bound pairs

  /// Examples of using this constructor:
  /// \code
  ///   Range r(10, 10, 10);
  ///   std::vector<size_t> lobounds = {0, 1, 2};
  ///   std::vector<size_t> upbounds = {4, 6, 8};
  ///
  ///   // using vector of pairs
  ///   std::vector<std::pair<size_t,size_t>> vpbounds{{0,4}, {1,6}, {2,8}};
  ///   BlockRange br0(r, vpbounds);
  ///   // using vector of tuples
  ///   std::vector<std::tuple<size_t,size_t>> vtbounds{{0,4}, {1,6}, {2,8}};
  ///   BlockRange br1(r, vpbounds);
  ///   assert(br0 == br1);
  ///
  ///   // using zipped ranges of bounds (using Boost.Range)
  ///   // need to #include <boost/range/combine.hpp>
  ///   BlockRange br2(r, boost::combine(lobounds, upbounds));
  ///   assert(br0 == br2);
  ///
  ///   // using zipped ranges of bounds (using Ranges-V3)
  ///   // need to #include <range/v3/view/zip.hpp>
  ///   BlockRange br3(r, ranges::views::zip(lobounds, upbounds));
  ///   assert(br0 == br3);
  /// \endcode
  /// \tparam PairRange Type representing a range of generalized pairs (see TiledArray::detail::is_gpair_v )
  /// \param bounds A range of {lower,upper} bounds for each dimension
  /// \throw TiledArray::Exception When `bounds[i].lower>=bounds[i].upper` for any \c i .
  // clang-format on
  template <typename PairRange,
            typename = std::enable_if_t<detail::is_gpair_range_v<PairRange>>>
  BlockRange(const Range& range, const PairRange& bounds) : Range() {
    init(range, bounds);
  }

  // clang-format off
  /// Construct range defined by an initializer_list of std::initializer_list{lower,upper} bounds

  /// Examples of using this constructor:
  /// \code
  ///   Range r(10, 10, 10);
  ///   BlockRange br0(r, {std::make_pair(0,4), std::pair{1,6}, std::pair(2,8)});
  /// \endcode
  /// \tparam GPair a generalized pair of integral types
  /// \param bound A range of {lower,upper} bounds for each dimension
  /// \throw TiledArray::Exception When `bound[i].lower>=bound[i].upper` for any \c i .
  // clang-format on
  template <typename GPair>
  BlockRange(const Range& range, const std::initializer_list<GPair>& bounds,
             std::enable_if_t<detail::is_gpair_v<GPair>>* = nullptr)
      : Range() {
#ifndef NDEBUG
    if constexpr (detail::is_contiguous_range_v<GPair>) {
      for (auto&& bound_d : bounds) {
        TA_ASSERT(size(bound_d) == 2);
      }
    }
#endif
    init<std::initializer_list<GPair>>(range, bounds);
  }

  // clang-format off
  /// Construct range defined by an initializer_list of std::initializer_list{lower,upper} bounds

  /// Examples of using this constructor:
  /// \code
  ///   Range r(10, 10, 10);
  ///   BlockRange br0(r, {{0,4}, {1,6}, {2,8}});
  /// \endcode
  /// \tparam Index An integral type
  /// \param bound A range of {lower,upper} bounds for each dimension
  /// \throw TiledArray::Exception When `bound[i].lower>=bound[i].upper` for any \c i .
  // clang-format on
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  BlockRange(const Range& range,
             const std::initializer_list<std::initializer_list<Index>>& bounds)
      : Range() {
#ifndef NDEBUG
    for (auto&& bound_d : bounds) {
      TA_ASSERT(size(bound_d) == 2);
    }
#endif
    init<std::initializer_list<std::initializer_list<Index>>>(range, bounds);
  }

  /// Converts coordinate index \c i to its ordinal index in the host Range

  /// \tparam Index An integral range type
  /// \param index The index to be converted to the ordinal index
  /// \return The ordinal index of \c index in the host Range
  /// \throw When \c index is not included in this range.
  template <typename Index,
            typename std::enable_if_t<
                detail::is_integral_sized_range_v<Index>>* = nullptr>
  ordinal_type ordinal(const Index& index) const {
    return Range::ordinal(index);
  }

  /// Converts coordinate index \c i to its ordinal index in the host Range

  /// \tparam Index An integral range type
  /// \param index The index to be converted to the ordinal index
  /// \return The ordinal index of \c index in the host Range
  /// \throw When \c index is not included in this range.
  template <typename Integer,
            typename std::enable_if_t<std::is_integral_v<Integer>>* = nullptr>
  ordinal_type ordinal(const std::initializer_list<Integer>& index) const {
    return Range::ordinal(index);
  }

  /// Converts coordinate index \c i to its ordinal index in the host Range

  /// \tparam Index An integral range type
  /// \param index The index to be converted to the ordinal index
  /// \return The ordinal index of \c index in the host Range
  /// \throw When \c index is not included in this range.
  template <
      typename... Index,
      typename std::enable_if_t<(sizeof...(Index) > 1ul) &&
                                (std::is_integral_v<Index> && ...)>* = nullptr>
  ordinal_type ordinal(const Index&... index) const {
    return Range::ordinal(index...);
  }

  /// Converts the ordinal index within this block range to the ordinal index of
  /// the host range

  /// \warning this overloads base method that did not perform any conversion
  ///          of \p ord
  /// \param index the ordinal index for the host Range
  /// \return The ordinal index in the
  /// \throw TiledArray::Exception When \c index is not included in this range
  ordinal_type ordinal(ordinal_type ord) const {
    // Check that ord is contained by this range.
    TA_ASSERT(Range::includes_ordinal(ord));

    // Construct result coordinate index object and allocate its memory.
    ordinal_type host_ord = 0ul;

    // Get pointers to the data
    const auto* MADNESS_RESTRICT const size = extent_data();
    const auto* MADNESS_RESTRICT const stride = stride_data();

    // Compute the coordinate index of o in range.
    for (int i = int(rank_) - 1; i >= 0; --i) {
      const auto size_i = size[i];
      const auto stride_i = stride[i];

      // Compute result index element i
      host_ord += (ord % size_i) * stride_i;
      ord /= size_i;
    }

    return host_ord + block_offset_ - offset_;
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
  void serialize(Archive& ar) const {
    Range::serialize(ar);
    ar& block_offset_;
  }
};  // BlockRange

// clang-format off
/// Test that two BlockRange objects are congruent

/// This function tests that the rank, extent of \p r1 is equal to that of \p r2.
/// \param r1 The first BlockRange to compare
/// \param r2 The second BlockRange to compare
// clang-format on
inline bool is_congruent(const BlockRange& r1, const BlockRange& r2) {
  return is_congruent(static_cast<const Range&>(r1),
                      static_cast<const Range&>(r2));
}

// clang-format off
/// Test that BlockRange and Range are congruent

/// This function tests that the rank, extent of \p r1 is equal to that of \p r2.
/// \param r1 The BlockRange to compare
/// \param r2 The Range to compare
// clang-format on
inline bool is_congruent(const BlockRange& r1, const Range& r2) {
  return is_congruent(static_cast<const Range&>(r1), r2);
}

// clang-format off
/// Test that Range and BlockRange are congruent

/// This function tests that the rank, extent of \c r1 is equal to that of \c r2.
/// \param r1 The Range to compare
/// \param r2 The BlockRange to compare
// clang-format on
inline bool is_congruent(const Range& r1, const BlockRange& r2) {
  return is_congruent(r2, r1);
}

/// BlockRange equality comparison

/// \param r1 The first range to be compared
/// \param r2 The second range to be compared
/// \return \c true when \p r1 represents the same range as \p r2, otherwise
/// \c false.
inline bool operator==(const BlockRange& r1, const BlockRange& r2) {
  return r1.block_offset_ == r2.block_offset_ &&
         static_cast<const Range&>(r1) == static_cast<const Range&>(r2);
}

/// Tests whether a range is contiguous, i.e. whether its ordinal values form a
/// contiguous range

/// \param range a BlockRange
/// \return true since TiledArray::BlockRange is contiguous by definition
inline bool is_contiguous(const BlockRange& range) { return true; }

}  // namespace TiledArray

#endif  // TILEDARRAY_BLOCK_RANGE_H__INCLUDED
