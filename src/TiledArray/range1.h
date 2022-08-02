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

/// an integer range
/// @note previously represented by std::pair, hence the design
struct Range1 {
  typedef TA_1INDEX_TYPE index1_type;
  index1_type first = 0;
  index1_type second = 0;  //< N.B. second >= first

  Range1() = default;
  Range1(Range1 const&) = default;
  Range1(Range1&&) = default;
  Range1& operator=(Range1 const&) = default;
  Range1& operator=(Range1&&) = default;

  /// \pre first <= second
  template <typename U1, typename U2>
  explicit Range1(U1&& u1, U2&& u2)
      : first(_VSTD::forward<U1>(u1)), second(_VSTD::forward<U2>(u2)) {
    TA_ASSERT(second >= first);
  }

  /// @return the lower bound of this range, i.e. first
  auto lobound() const noexcept { return first; }

  /// @return the upper bound of this range, i.e. second
  auto upbound() const noexcept { return second; }

  /// @return the extent of this range, i.e. second - first
  auto extent() const noexcept { return second - first; }

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
  virtual const_iterator begin() const { return Iterator{first}; }

  /// End local element iterator

  /// \return An iterator that points to the beginning of the local element set
  virtual const_iterator end() const { return Iterator{second}; }

  /// Begin local element iterator

  /// \return An iterator that points to the beginning of the local element set
  const const_iterator cbegin() const { return begin(); }

  /// End local element iterator

  /// \return An iterator that points to the beginning of the local element set
  const const_iterator cend() const { return end(); }

  /// @}

  template <typename Archive,
            typename std::enable_if<madness::is_input_archive_v<
                std::decay_t<Archive>>>::type* = nullptr>
  void serialize(Archive& ar) {
    ar& first& second;
  }

  template <typename Archive,
            typename std::enable_if<madness::is_output_archive_v<
                std::decay_t<Archive>>>::type* = nullptr>
  void serialize(Archive& ar) const {
    ar& first& second;
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
