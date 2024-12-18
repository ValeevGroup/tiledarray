/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2022  Virginia Tech
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

#ifndef TILEDARRAY_RANGE1_H__INCLUDED
#define TILEDARRAY_RANGE1_H__INCLUDED

#include <TiledArray/config.h>

#include <madness/world/archive.h>
#include <boost/iterator/iterator_facade.hpp>

#include <TiledArray/error.h>

namespace TiledArray {

/// an integer range `[first,second)`
/// @note previously represented by std::pair, hence the design
struct Range1 {
  using index1_type = TA_1INDEX_TYPE;
  using signed_index1_type = std::make_signed_t<index1_type>;
  index1_type first = 0;
  index1_type second = 0;  //< N.B. second >= first

  Range1() = default;
  Range1(Range1 const&) = default;
  Range1(Range1&&) = default;
  Range1& operator=(Range1 const&) = default;
  Range1& operator=(Range1&&) = default;

  /// constructs `[0,end)`
  /// \param[in] end the value immediately past the last value in the range
  /// \pre 0 <= end
  template <typename I, typename = std::enable_if_t<
                            !std::is_same_v<std::decay_t<I>, Range1> &&
                            std::is_integral_v<I>>>
  explicit Range1(I end) : second(static_cast<index1_type>(end)) {
    TA_ASSERT(second >= first);
  }

  /// constructs `[begin,end)`
  /// \param[in] begin the first value in the range
  /// \param[in] end the value immediately past the last value in the range
  /// \pre begin <= end
  template <typename I1, typename I2,
            typename = std::enable_if_t<std::is_integral_v<I1> &&
                                        std::is_integral_v<I2>>>
  explicit Range1(I1 begin, I2 end)
      : first(static_cast<index1_type>(begin)),
        second(static_cast<index1_type>(end)) {
    TA_ASSERT(second >= first);
  }

  /// @return the lower bound of this range, i.e. first
  auto lobound() const noexcept { return first; }

  /// @return the upper bound of this range, i.e. second
  auto upbound() const noexcept { return second; }

  /// @return the extent of this range, i.e. second - first
  auto extent() const noexcept { return second - first; }

  /// @return the volume of this range, i.e. second - first
  auto volume() const noexcept { return second - first; }

  /// swaps `*this` with @p other
  /// @p other a Range1 object
  void swap(Range1& other) noexcept {
    std::swap(first, other.first);
    std::swap(second, other.second);
  }

  /// converts this to a std::pair
  /// @return std::pair<index1_type,index1_type> representation of this
  explicit operator std::pair<index1_type, index1_type>() const {
    return std::make_pair(first, second);
  }

  /// Checks if a given index is within this range
  /// @return true if \p i is within this range
  template <typename I>
  typename std::enable_if<std::is_integral<I>::value, bool>::type includes(
      const I& i) const {
    return first <= i && i < second;
  }

  /// Checks if a given range overlaps with this range

  /// @return true if \p r overlaps with this range
  bool overlaps_with(const Range1& rng) const {
    return lobound() < rng.upbound() && upbound() > rng.lobound();
  }

  /// \brief Range1 iterator type
  ///
  /// Iterates over Range1
  class Iterator
      : public boost::iterator_facade<Iterator, const Range1::index1_type,
                                      boost::random_access_traversal_tag> {
   public:
    Iterator(value_type v) : v(v) {}

   private:
    value_type v;

    friend class boost::iterator_core_access;

    /// \brief increments this iterator
    void increment() { ++v; }
    /// \brief decrements this iterator
    void decrement() { --v; }

    /// \brief advances this iterator by `n` positions
    void advance(difference_type n) { v += n; }

    /// \brief advances this iterator by `n` positions
    auto distance_to(Iterator other) { return other.v - v; }

    /// \brief Iterator comparer
    /// \return true, if \c `*this==*other`
    bool equal(Iterator const& other) const { return this->v == other.v; }

    /// \brief dereferences this iterator
    /// \return const reference to the current index
    const auto& dereference() const { return v; }
  };
  friend class Iterator;

  typedef Iterator const_iterator;  ///< Iterator type

  /// Begin local element iterator

  /// \return An iterator that points to the beginning of the local element set
  const_iterator begin() const { return Iterator{first}; }

  /// End local element iterator

  /// \return An iterator that points to the beginning of the local element set
  const_iterator end() const { return Iterator{second}; }

  /// Begin local element iterator

  /// \return An iterator that points to the beginning of the local element set
  const_iterator cbegin() const { return begin(); }

  /// End local element iterator

  /// \return An iterator that points to the beginning of the local element set
  const_iterator cend() const { return end(); }

  /// shifts this Range1

  /// @param[in] shift the shift to apply
  /// @return reference to this
  Range1& inplace_shift(signed_index1_type shift) {
    if (shift == 0) return *this;
    // ensure that it's safe to shift
    TA_ASSERT(shift <= 0 || upbound() <= 0 ||
              (shift <= (std::numeric_limits<index1_type>::max() - upbound())));
    TA_ASSERT(shift >= 0 || lobound() >= 0 ||
              (std::abs(shift) <=
               (lobound() - std::numeric_limits<index1_type>::min())));
    first += shift;
    second += shift;
    return *this;
  }

  /// creates a shifted Range1

  /// @param[in] shift the shift value
  /// @return a copy of this shifted by @p shift
  [[nodiscard]] Range1 shift(signed_index1_type shift) const {
    return Range1(*this).inplace_shift(shift);
  }

  template <typename Archive,
            typename std::enable_if<madness::is_input_archive_v<
                std::decay_t<Archive>>>::type* = nullptr>
  void serialize(Archive& ar) {
    ar & first & second;
  }

  template <typename Archive,
            typename std::enable_if<madness::is_output_archive_v<
                std::decay_t<Archive>>>::type* = nullptr>
  void serialize(Archive& ar) const {
    ar & first & second;
  }
};

inline bool operator==(const Range1& x, const Range1& y) {
  return x.first == y.first && x.second == y.second;
}

inline bool operator!=(const Range1& x, const Range1& y) { return !(x == y); }

/// Exchange the data of the two given ranges.
inline void swap(Range1& r0, Range1& r1) {  // no throw
  r0.swap(r1);
}

/// Range1 ostream operator
inline std::ostream& operator<<(std::ostream& out, const Range1& rng) {
  out << "[ " << rng.first << ", " << rng.second << " )";
  return out;
}

/// Test that two Range1 objects are congruent

/// This function tests that the sizes of the two Range1 objects coincide.
/// \param r1 an Range1 object
/// \param r2 an Range1 object
inline bool is_congruent(const Range1& r1, const Range1& r2) {
  return r1.extent() == r2.extent();
}

inline auto extent(const Range1& r) { return r.extent(); }

template <std::size_t I>
constexpr auto get(const Range1& r) {
  static_assert(I < 2, "");
  if constexpr (I == 0)
    return r.first;
  else
    return r.second;
}

/// constructs `[0,end)`
/// \param[in] end the value immediately past the last value in the range
/// \pre 0 <= end
/// \note syntactic sugar a la boost::irange
template <typename I>
Range1 irange(I end) {
  return Range1{end};
}

/// constructs `[begin,end)`
/// \param[in] begin the first value in the range
/// \param[in] end the value immediately past the last value in the range
/// \pre begin <= end
/// \note syntactic sugar a la boost::irange
template <typename I1, typename I2,
          typename = std::enable_if_t<std::is_integral_v<I1> &&
                                      std::is_integral_v<I2>>>
Range1 irange(I1 begin, I2 end) {
  return Range1{begin, end};
}

}  // namespace TiledArray

// Range1 is tuple-like
namespace std {
template <>
struct tuple_size<::TiledArray::Range1> {
  static constexpr std::size_t value = 2;
};

template <>
struct tuple_element<0, ::TiledArray::Range1> {
  using type = decltype(::TiledArray::Range1::first);
};
template <>
struct tuple_element<1, ::TiledArray::Range1> {
  using type = decltype(::TiledArray::Range1::second);
};

}  // namespace std

namespace boost {
template <std::size_t I>
constexpr auto get(const ::TiledArray::Range1& r) {
  static_assert(I < 2, "");
  if constexpr (I == 0)
    return r.first;
  else
    return r.second;
}
}  // namespace boost

#endif  // TILEDARRAY_RANGE1_H__INCLUDED
