/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 */

#ifndef TILEDARRAY_TILED_RANGE1_H__INCLUDED
#define TILEDARRAY_TILED_RANGE1_H__INCLUDED

#include <TiledArray/config.h>

#include <TiledArray/error.h>
#include <TiledArray/range1.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/utility.h>
#include <madness/world/archive.h>

#include <cassert>
#include <initializer_list>
#include <memory>
#include <mutex>
#include <vector>

namespace TiledArray {

/// TiledRange1 class defines a non-uniformly-tiled, contiguous, one-dimensional
/// range. The tiling data is constructed with and stored in an array with
/// the format {a0, a1, a2, ...}, where a0 <= a1 <= a2 <= ... Each tile is
/// defined as [a0,a1), [a1,a2), ... The number of tiles in the range will be
/// equal to one less than the number of elements in the array.
/// \note if TiledArray was configured with `TA_SIGNED_1INDEX_TYPE=OFF` then the
/// tile boundaries must be non-negative.
class TiledRange1 {
 private:
  struct Enabler {};

 public:
  using range_type = Range1;
  using index1_type = range_type::index1_type;
  using signed_index1_type = range_type::signed_index1_type;
  using const_iterator = std::vector<range_type>::const_iterator;

  /// Default constructor creates an empty range (tile and element ranges are
  /// both [0,0)) .
  /// \post \code
  ///   TiledRange1 tr;
  ///   assert(tr.tiles_range() == (TiledRange1::range_type{0,0}));
  ///   assert(tr.elements_range() == (TiledRange1::range_type{0,0}));
  ///   assert(tr.begin() == tr.end());
  /// \endcode
  TiledRange1() : range_(0, 0), elements_range_(0, 0) {}

  /// Constructs a range with the tile boundaries ("hashmarks") provided by
  /// the range [ \p first , \p last ).
  /// \note validity of the [ \p first , \p last ) range is checked using
  /// #TA_ASSERT() only if preprocessor macro \c NDEBUG is not defined
  template <typename RandIter,
            typename std::enable_if<
                detail::is_random_iterator<RandIter>::value>::type* = nullptr>
  explicit TiledRange1(RandIter first, RandIter last) {
    init_tiles_(first, last, 0);
  }

  /// Copy constructor
  TiledRange1(const TiledRange1& rng) = default;

  /// Move constructor
  TiledRange1(TiledRange1&& rng) = default;

  /// Construct a 1D tiled range.

  /// This will construct a 1D tiled range with tile boundaries ("hashmarks")
  /// {\p t0 , \p t_rest... }
  /// The number of tile boundaries is n + 1, where n is the number of tiles.
  /// Tiles are defined as [\p t0, t1), [t1, t2), [t2, t3), ...
  /// \param t0 The starting index of the first tile
  /// \param t_rest The rest of tile boundaries
  /// \note validity of the {\p t0 , \p t_rest... } range is checked using
  /// #TA_ASSERT() only if preprocessor macro \c NDEBUG is not defined
  template <typename... _sizes>
  explicit TiledRange1(const index1_type& t0, const _sizes&... t_rest) {
    const auto n = sizeof...(_sizes) + 1;
    index1_type tile_boundaries[n] = {t0, static_cast<index1_type>(t_rest)...};
    init_tiles_(tile_boundaries, tile_boundaries + n, 0);
  }

  /// Construct a 1D tiled range.

  /// This will construct a 1D tiled range from range {t0, t1, t2, ... tn}
  /// specifying the tile boundaries (hashmarks).
  /// The number of tile boundaries is n + 1, where n is the number of tiles.
  /// Tiles are defined as [\p t0 , t1), [t1, t2), [t2, t3), ...
  /// Tiles are indexed starting with 0.
  /// \tparam Integer An integral type
  /// \param tile_boundaries The list of tile boundaries in order from smallest
  /// to largest
  /// \note validity of the {\p t0 , \p t_rest... } range is checked using
  ///   #TA_ASSERT() only if preprocessor macro \c NDEBUG is not defined
  template <typename Range,
            typename = std::enable_if_t<detail::is_integral_range_v<Range>>>
  explicit TiledRange1(Range&& tile_boundaries) {
    init_tiles_(tile_boundaries.begin(), tile_boundaries.end(), 0);
  }

  /// Construct a 1D tiled range.

  /// This will construct a 1D tiled range from range {t0, t1, t2, ... tn}
  /// specifying the tile boundaries (hashmarks).
  /// The number of tile boundaries is n + 1, where n is the number of tiles.
  /// Tiles are defined as [\p t0 , t1), [t1, t2), [t2, t3), ...
  /// Tiles are indexed starting with 0.
  /// \tparam Integer An integral type
  /// \param tile_boundaries The list of tile boundaries in order from smallest
  /// to largest
  /// \note validity of the {\p t0 , \p t_rest... } range is checked using
  /// #TA_ASSERT() only if preprocessor macro \c NDEBUG is not defined
  template <typename Integer,
            typename = std::enable_if_t<std::is_integral_v<Integer>>>
  explicit TiledRange1(const std::initializer_list<Integer>& tile_boundaries) {
    init_tiles_(tile_boundaries.begin(), tile_boundaries.end(), 0);
  }

  /// Copy assignment operator
  TiledRange1& operator=(const TiledRange1& rng) = default;

  /// Move assignment operator
  TiledRange1& operator=(TiledRange1&& rng) = default;

  /// Returns an iterator to the first tile in the range.
  const_iterator begin() const { return tiles_ranges_.begin(); }

  /// Returns an iterator to the end of the range.
  const_iterator end() const { return tiles_ranges_.end(); }

  /// Returns true if this range is empty (i.e. has no tiles)
  bool empty() const { return tiles_ranges_.empty(); }

  /// Return tile iterator associated with ordinal_index
  const_iterator find(const index1_type& e) const {
    if (!includes(elements_range_, e)) return tiles_ranges_.end();
    const_iterator result = tiles_ranges_.begin();
    result += element_to_tile(e);
    return result;
  }

  /// \deprecated use TiledRange1::tiles_range()
  [[deprecated]] const range_type& tiles() const { return range_; }

  /// Tile range accessor

  /// \return a reference to the tile index range
  const range_type& tiles_range() const { return range_; }

  /// \deprecated use TiledRange1::elements_range()
  const range_type& elements() const { return elements_range_; }

  /// Elements range accessor

  /// \return a reference to the element index range
  const range_type& elements_range() const { return elements_range_; }

  /// Tile range extent accessor
  /// \return the number of tiles in the range
  index1_type tile_extent() const { return TiledArray::extent(range_); }

  /// Elements range extent accessor
  /// \return the number of elements in the range
  index1_type extent() const { return TiledArray::extent(elements_range_); }

  /// Computes hashmarks
  /// \return the hashmarks of the tiled range, consisting of the following
  /// values:
  ///         `{ tile(0).first, tile(0).second, tile(1).second, tile(2).second
  ///         ... }`
  std::vector<index1_type> hashmarks() const {
    std::vector<index1_type> result;
    result.reserve(tile_extent() + 1);
    result.push_back(elements_range().first);
    for (auto& t : tiles_ranges_) {
      result.push_back(t.second);
    }
    return result;
  }

  /// \return the size of the largest tile in the range
  /// \pre `this->tile_extent() > 0`
  index1_type largest_tile_extent() const {
    TA_ASSERT(tile_extent() > 0);
    using TiledArray::extent;
    auto largest_tile_it =
        std::max_element(tiles_ranges_.begin(), tiles_ranges_.end(),
                         [](const auto& tile1, const auto& tile2) {
                           return extent(tile1) < extent(tile2);
                         });
    return extent(*largest_tile_it);
  }

  /// \return the size of the smallest tile in the range
  /// \pre `this->tile_extent() > 0`
  index1_type smallest_tile_extent() const {
    TA_ASSERT(tile_extent() > 0);
    using TiledArray::extent;
    auto smallest_tile_it =
        std::min_element(tiles_ranges_.begin(), tiles_ranges_.end(),
                         [](const auto& tile1, const auto& tile2) {
                           return extent(tile1) < extent(tile2);
                         });
    return extent(*smallest_tile_it);
  }

  /// Accesses range of a particular tile

  /// \param i tile index
  /// \return A const reference to the range of tile \c i
  /// \pre
  /// \code
  /// assert(i >= tiles_range().first && i < tiles_range().second);
  /// \endcode
  const range_type& tile(const index1_type i) const {
    TA_ASSERT(includes(range_, i));
    return tiles_ranges_[i - range_.first];
  }

  /// Maps element index to tile index

  /// \param i element index
  /// \return tile index
  /// \pre
  /// \code
  /// assert(i >= elements_range().first && i < elements_range().second);
  /// \endcode
  /// \note element->tile map is memoized, thus complexity of the first call is
  /// linear in the number of elements,
  ///       complexity of subsequent calls is constant. Note that the
  ///       memoization assumes infrequent (or serial) use as it is serialized
  ///       across ALL TiledRange1 instances.
  const index1_type& element_to_tile(const index1_type& i) const {
    TA_ASSERT(includes(elements_range_, i));
    if (!elem2tile_) {
      init_elem2tile_();
    }
    // N.B. only track elements in this range
    return elem2tile_[i - elements_range_.first];
  }

  /// \deprecated use TiledRange1::element_to_tile()
  [[deprecated]] const index1_type& element2tile(const index1_type& i) const {
    return element_to_tile(i);
  }

  // clang-format off
  /// @brief makes a uniform (or, as uniform as possible) TiledRange1

  /// @param[in] range the Range to be tiled
  /// @param[in] target_tile_size the desired tile size
  /// @return TiledRange1 obtained by tiling \p range into
  /// `ntiles = (range.extent() + target_tile_size - 1)/target_tile_size`
  ///         tiles; if `x = range.extent() % ntiles` is not zero, first `x` tiles
  /// have size `target_tile_size` and last
  /// `ntiles - x` tiles have size `target_tile_size - 1`, else
  /// all tiles have size `target_tile_size` .
  // clang-format on
  static TiledRange1 make_uniform(const Range1& range,
                                  std::size_t target_tile_size) {
    const auto range_extent = range.extent();
    if (range_extent > 0) {
      TA_ASSERT(target_tile_size > 0);
      std::size_t ntiles =
          (range_extent + target_tile_size - 1) / target_tile_size;
      auto dv = std::div((long)(range_extent + ntiles - 1), (long)ntiles);
      auto avg_tile_size = dv.quot - 1, num_avg_plus_one = dv.rem + 1;
      std::vector<std::size_t> hashmarks;
      hashmarks.reserve(ntiles + 1);
      std::size_t element = range.lobound();
      for (auto i = 0; i < num_avg_plus_one;
           ++i, element += avg_tile_size + 1) {
        hashmarks.push_back(element);
      }
      for (auto i = num_avg_plus_one; i < ntiles;
           ++i, element += avg_tile_size) {
        hashmarks.push_back(element);
      }
      hashmarks.push_back(range.upbound());
      return TiledRange1(hashmarks.begin(), hashmarks.end());
    } else
      return TiledRange1{};
  }

  /// same as make_uniform(const Range1&, std::size_t) for a 0-based range
  /// specified by its extent
  static TiledRange1 make_uniform(std::size_t range_extent,
                                  std::size_t target_tile_size) {
    return make_uniform(Range1(0, range_extent), target_tile_size);
  }

  /// shifts this TiledRange1

  /// @param[in] shift the shift to apply
  /// @return reference to this
  TiledRange1& inplace_shift(signed_index1_type shift) {
    if (shift == 0) return *this;
    // ensure that it's safe to shift
    TA_ASSERT(shift <= 0 || elements_range().upbound() <= 0 ||
              (shift <= (std::numeric_limits<index1_type>::max() -
                         elements_range().upbound())));
    TA_ASSERT(shift >= 0 || elements_range().lobound() >= 0 ||
              (std::abs(shift) <= (elements_range().lobound() -
                                   std::numeric_limits<index1_type>::min())));
    elements_range_.inplace_shift(shift);
    for (auto& tile : tiles_ranges_) {
      tile.inplace_shift(shift);
    }
    elem2tile_.reset();
    return *this;
  }

  /// creates a shifted TiledRange1

  /// equivalent to (but more efficient than) `TiledRange1(*this).shift(shift)`
  /// @param[in] shift the shift value
  [[nodiscard]] TiledRange1 shift(signed_index1_type shift) const {
    if (shift == 0) return *this;
    // ensure that it's safe to shift
    TA_ASSERT(shift <= 0 || elements_range().upbound() <= 0 ||
              (shift <= (std::numeric_limits<index1_type>::max() -
                         elements_range().upbound())));
    TA_ASSERT(shift >= 0 || elements_range().lobound() >= 0 ||
              (std::abs(shift) <= (elements_range().lobound() -
                                   std::numeric_limits<index1_type>::min())));
    std::vector<index1_type> hashmarks;
    hashmarks.reserve(tile_extent() + 1);
    if (tiles_ranges_.empty())
      hashmarks.emplace_back(elements_range_.lobound() + shift);
    else {
      for (auto& t : tiles_ranges_) {
        hashmarks.push_back(t.first + shift);
      }
      hashmarks.push_back(elements_range_.upbound() + shift);
    }
    return TiledRange1(hashmarks.begin(), hashmarks.end());
  }

  /// swapper

  /// \param other the range with which the contents of this range will be
  /// swapped
  void swap(TiledRange1& other) {  // no throw
    std::swap(range_, other.range_);
    std::swap(elements_range_, other.elements_range_);
    std::swap(tiles_ranges_, other.tiles_ranges_);
    std::swap(elem2tile_, other.elem2tile_);
  }

  template <typename Archive,
            typename std::enable_if<madness::is_input_archive_v<
                std::decay_t<Archive>>>::type* = nullptr>
  void serialize(Archive& ar) {
    ar & range_ & elements_range_ & tiles_ranges_;
  }

  template <typename Archive,
            typename std::enable_if<madness::is_output_archive_v<
                std::decay_t<Archive>>>::type* = nullptr>
  void serialize(Archive& ar) const {
    ar & range_ & elements_range_ & tiles_ranges_;
  }

 private:
  static bool includes(const range_type& r, index1_type i) {
    return (i >= r.first) && (i < r.second);
  }

  /// Validates tile_boundaries
  template <typename RandIter>
  static void valid_(RandIter first, RandIter last) {
    // Need at least 1 tile hashmark to position the element range
    // (zero hashmarks is handled by the default ctor)
    TA_ASSERT((std::distance(first, last) >= 1) &&
              "TiledRange1 construction failed: You need at least 1 "
              "element in the tile boundary list.");
    // Verify the requirement that a0 <= a1 <= a2 <= ...
    for (; first != (last - 1); ++first) {
      TA_ASSERT(
          *first <= *(first + 1) &&
          "TiledRange1 construction failed: Invalid tile boundary, tile "
          "boundary i must not be greater than tile boundary i+1 for all i. ");
      TA_ASSERT(
          static_cast<index1_type>(*first) <=
              static_cast<index1_type>(*(first + 1)) &&
          "TiledRange1 construction failed: Invalid tile boundary, tile "
          "boundary i must not be greater than tile boundary i+1 for all i. ");
    }
  }

  /// Initialize tiles use a set of tile offsets
  template <typename RandIter>
  void init_tiles_(RandIter first, RandIter last,
                   index1_type start_tile_index) {
#ifndef NDEBUG
    valid_(first, last);
#endif  // NDEBUG
    range_.first = start_tile_index;
    using std::distance;
    range_.second =
        start_tile_index + static_cast<index1_type>(distance(first, last)) - 1;
    elements_range_.first = *first;
    elements_range_.second = *(last - 1);
    for (; first != (last - 1); ++first)
      tiles_ranges_.emplace_back(*first, *(first + 1));
  }

  /// Initialize elem2tile
  void init_elem2tile_() const {
    using TiledArray::extent;
    // check for 0 size range.
    const auto n = extent(elements_range_);
    if (n == 0) return;

    static std::mutex mtx;
    {
      std::lock_guard<std::mutex> lock(mtx);
      if (!elem2tile_) {
        // initialize elem2tile map
        auto e2t =
            // #if __cplusplus >= 202002L  ... still broken in Xcode 14
            //             std::make_shared<index1_type[]>(n);
            // #else
            std::shared_ptr<index1_type[]>(
                new index1_type[n], [](index1_type* ptr) { delete[] ptr; });
        // #endif
        const auto end = extent(range_);
        for (index1_type t = 0; t < end; ++t)
          for (auto e : tiles_ranges_[t]) {
            // only track elements in this range
            e2t[e - elements_range_.first] = t + range_.first;
          }
        auto e2t_const = std::const_pointer_cast<const index1_type[]>(e2t);
        // commit the changes
        std::swap(elem2tile_, e2t_const);
      }
    }
  }

  friend std::ostream& operator<<(std::ostream&, const TiledRange1&);

  // TiledRange1 data
  range_type range_;           ///< the range of tile indices
  range_type elements_range_;  ///< the range of element indices
  std::vector<range_type>
      tiles_ranges_;  ///< ranges of each tile (NO GAPS between tiles)
  mutable std::shared_ptr<const index1_type[]>
      elem2tile_;  ///< maps element index to tile index (memoized data).

};  // class TiledRange1

/// Exchange the data of the two given ranges.
inline void swap(TiledRange1& r0, TiledRange1& r1) {  // no throw
  r0.swap(r1);
}

/// Equality operator
inline bool operator==(const TiledRange1& r1, const TiledRange1& r2) {
  return (r1.tiles_range() == r2.tiles_range()) &&
         (r1.elements_range() == r2.elements_range()) &&
         std::equal(r1.begin(), r1.end(), r2.begin());
}

/// Inequality operator
inline bool operator!=(const TiledRange1& r1, const TiledRange1& r2) {
  return !operator==(r1, r2);
}

/// TiledRange1 ostream operator
inline std::ostream& operator<<(std::ostream& out, const TiledRange1& rng) {
  out << "( tiles = " << rng.tiles_range()
      << ", elements = " << rng.elements_range() << " )";
  return out;
}

// clang-format off
/// Concatenates two ranges

/// Tiles of the second range are concatenated to the tiles of the first. For
/// example:
/// \code
/// assert(concat((TiledRange1{1, 3, 7, 9}),(TiledRange1{0, 3, 4, 5})) == (TiledRange1{1, 3, 7, 9, 12, 13, 14}));
/// assert(concat((TiledRange1{0, 3, 4, 5}),(TiledRange1{1, 3, 7, 9})) == (TiledRange1{0, 3, 4, 5, 7, 11, 13}));
/// \endcode
/// \param r1 first range
/// \param r2 second range
/// \return concatenated range
// clang-format on
inline TiledRange1 concat(const TiledRange1& r1, const TiledRange1& r2) {
  std::vector<TiledRange1::index1_type> hashmarks;
  hashmarks.reserve(r1.tile_extent() + r2.tile_extent() + 1);
  if (!r1.empty()) {
    hashmarks.push_back(r1.tile(0).first);
    for (const auto& tile : r1) {
      hashmarks.push_back(tile.second);
    }
    for (const auto& tile : r2) {
      hashmarks.push_back(hashmarks.back() + TiledArray::extent(tile));
    }
    return TiledRange1(hashmarks.begin(), hashmarks.end());
  } else {
    return r2;
  }
}

/// Test that two TiledRange1 objects are congruent

/// This function tests that the tile sizes of the two ranges coincide.
/// \tparam Range The range type
/// \param r1 an TiledRange1 object
/// \param r2 an TiledRange1 object
inline bool is_congruent(const TiledRange1& r1, const TiledRange1& r2) {
  return r1.tile_extent() == r2.tile_extent() &&
         std::equal(r1.begin(), r1.end(), r2.begin(),
                    [](const auto& tile1, const auto& tile2) {
                      return TiledArray::extent(tile1) ==
                             TiledArray::extent(tile2);
                    });
}

}  // namespace TiledArray

#endif  // TILEDARRAY_TILED_RANGE1_H__INCLUDED
