/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
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

#ifndef TILEDARRAY_TILED_RANGE_H__INCLUDED
#define TILEDARRAY_TILED_RANGE_H__INCLUDED

#include <TiledArray/range.h>
#include <TiledArray/tiled_range1.h>

namespace TiledArray {

/// Range data of a tiled array

/// TiledRange is a direct (Cartesian) product of 1-dimensional tiled ranges,
/// represented as TiledRange1 objects. Thus TiledRange is a semantically
/// contiguous (C++) range of TiledRange1 objects.
class TiledRange {
 private:
  /// Constructed with a set of ranges pointed to by [ first, last ).
  void init() {
    const std::size_t rank = this->rank();

    // Indices used to store range start and finish.
    using index_type = Range::index_type;
    index_type start;
    index_type finish;
    index_type start_element;
    index_type finish_element;

    start.reserve(rank);
    finish.reserve(rank);
    start_element.reserve(rank);
    finish_element.reserve(rank);

    // Find the start and finish of the over all tiles and element ranges.
    for (unsigned int i = 0; i < rank; ++i) {
      start.push_back(ranges_[i].tiles_range().first);
      finish.push_back(ranges_[i].tiles_range().second);

      start_element.push_back(ranges_[i].elements_range().first);
      finish_element.push_back(ranges_[i].elements_range().second);
    }
    range_type(start, finish).swap(range_);
    range_type(start_element, finish_element).swap(elements_range_);
  }

 public:
  // typedefs
  typedef TiledRange TiledRange_;
  typedef Range range_type;  // represents elements/tiles range
  typedef range_type::index_type index_type;
  typedef range_type::ordinal_type ordinal_type;
  typedef range_type::index1_type index1_type;
  static_assert(std::is_same_v<TiledRange1::index1_type, index1_type>);
  typedef container::svector<TiledRange1> Ranges;

  /// TiledRange is a contiguous C++ range of TiledRange1 objects
  using const_iterator = typename Ranges::const_iterator;
  /// TiledRange is a contiguous C++ range of TiledRange1 objects
  using value_type = typename Ranges::value_type;

  /// Default constructor
  TiledRange() : range_(), elements_range_(), ranges_() {}

  /// Constructs using a range of TiledRange1 objects
  /// \param first the iterator pointing to the front of the range
  /// \param last the iterator pointing past the back of the range
  template <typename InIter>
  explicit TiledRange(InIter first, InIter last)
      : range_(), elements_range_(), ranges_(first, last) {
    init();
  }

  /// Constructs using a range of TiledRange1 objects
  /// \param range_of_trange1s a range of TiledRange1 objects
  template <typename TRange1Range,
            typename = std::enable_if_t<
                detail::is_range_v<TRange1Range> &&
                std::is_same_v<detail::value_t<TRange1Range>, TiledRange1>>>
  explicit TiledRange(const TRange1Range& range_of_trange1s)
      : range_(),
        elements_range_(),
        ranges_(std::begin(range_of_trange1s), std::end(range_of_trange1s)) {
    init();
  }

  /// Constructs using a list of lists convertible to TiledRange1

  /// \tparam Integer An integral type
  /// \param list a list of lists of integers that can be converted to
  /// TiledRange1
  template <typename Integer,
            typename = std::enable_if_t<std::is_integral_v<Integer>>>
  explicit TiledRange(
      const std::initializer_list<std::initializer_list<Integer>>& list)
      : range_(), elements_range_() {
    ranges_.reserve(size(list));
    for (auto&& l : list) {
      ranges_.emplace_back(l);
    }
    init();
  }

  /// Constructed with an initializer_list of TiledRange1's
  explicit TiledRange(const std::initializer_list<TiledRange1>& list)
      : range_(), elements_range_(), ranges_(list.begin(), list.end()) {
    init();
  }

  /// Copy constructor
  TiledRange(const TiledRange_& other)
      : range_(other.range_),
        elements_range_(other.elements_range_),
        ranges_(other.ranges_) {}

  /// TiledRange assignment operator

  /// \return A reference to this object
  TiledRange_& operator=(const TiledRange_& other) {
    if (this != &other) TiledRange_(other).swap(*this);
    return *this;
  }

  /// In place permutation of this range.

  /// \return A reference to this object
  TiledRange_& operator*=(const Permutation& p) {
    TA_ASSERT(p.size() == range_.rank());
    Ranges temp = p * ranges_;
    TiledRange(temp.begin(), temp.end()).swap(*this);
    return *this;
  }

  /// Access the tile range

  /// \return A const reference to the tile range object
  const range_type& tiles_range() const { return range_; }

  /// Access the tile range

  /// \return A const reference to the tile range object
  /// \deprecated use TiledRange::tiles_range() instead
  [[deprecated]] const range_type& tiles() const { return tiles_range(); }

  /// Access the element range

  /// \return A const reference to the element range object
  const range_type& elements_range() const { return elements_range_; }

  /// Access the element range

  /// \return A const reference to the element range object
  /// \deprecated use TiledRange::elements_range() instead
  [[deprecated]] const range_type& elements() const { return elements_range(); }

  /// Constructs a range for the tile indexed by the given ordinal index.

  /// \param i The ordinal index of the tile range to be constructed
  /// \throw std::runtime_error Throws if i is not included in the range
  /// \return The constructed range object
  /// \note alias to TiledRange::make_tile_range() , introduced for consistency
  ///       with TiledRange1::tile()
  range_type tile(const index1_type& i) const { return make_tile_range(i); }

  /// Construct a range for the tile indexed by the given ordinal index.

  /// \param ord The ordinal index of the tile range to be constructed
  /// \throw std::runtime_error Throws if i is not included in the range
  /// \return The constructed range object
  range_type make_tile_range(const ordinal_type& ord) const {
    TA_ASSERT(tiles_range().includes_ordinal(ord));
    return make_tile_range(tiles_range().idx(ord));
  }

  /// Construct a range for the tile indexed by the given index.

  /// \tparam Index An integral range type
  /// \param index The index of the tile range to be constructed
  /// \throw std::runtime_error Throws if i is not included in the range
  /// \return The constructed range object
  /// \note alias to TiledRange::make_tile_range() , introduced for consistency
  ///       with TiledRange1::tile()
  template <typename Index>
  typename std::enable_if_t<detail::is_integral_range_v<Index>, range_type>
  tile(const Index& index) const {
    return make_tile_range(index);
  }

  /// Construct a range for the tile indexed by the given index.

  /// \tparam Index An integral range type
  /// \param index The index of the tile range to be constructed
  /// \throw std::runtime_error Throws if i is not included in the range
  /// \return The constructed range object
  template <typename Index>
  std::enable_if_t<detail::is_integral_range_v<Index>, range_type>
  make_tile_range(const Index& index) const {
    const auto rank = range_.rank();
    TA_ASSERT(range_.includes(index));
    typename range_type::index_type lower;
    typename range_type::index_type upper;
    lower.reserve(rank);
    upper.reserve(rank);
    unsigned d = 0;
    for (auto&& index_d : index) {
      lower.push_back(data()[d].tile(index_d).first);
      upper.push_back(data()[d].tile(index_d).second);
      ++d;
    }

    return range_type(lower, upper);
  }

  /// Construct a range for the tile indexed by the given index.

  /// \tparam Integer An integral type
  /// \param index The tile index, given as a \c std::initializer_list
  /// \throw std::runtime_error Throws if i is not included in the range
  /// \return The constructed range object
  /// \note alias to TiledRange::make_tile_range() , introduced for consistency
  ///       with TiledRange1::tile()
  template <typename Integer,
            typename = std::enable_if_t<std::is_integral_v<Integer>>>
  range_type tile(const std::initializer_list<Integer>& index) const {
    return make_tile_range(index);
  }

  /// Construct a range for the tile indexed by the given index.

  /// \tparam Integer An integral type
  /// \param index The tile index, given as a \c std::initializer_list
  /// \throw std::runtime_error Throws if i is not included in the range
  /// \return The constructed range object
  template <typename Integer,
            typename = std::enable_if_t<std::is_integral_v<Integer>>>
  range_type make_tile_range(
      const std::initializer_list<Integer>& index) const {
    return make_tile_range<std::initializer_list<Integer>>(index);
  }

  /// Convert an element index to a tile index

  /// \tparam Index An integral range type
  /// \param index The element index to convert
  /// \return The tile index that corresponds to the given element index
  template <typename Index>
  std::enable_if_t<detail::is_integral_range_v<Index>,
                   typename range_type::index>
  element_to_tile(const Index& index) const {
    const unsigned int rank = range_.rank();
    typename range_type::index result;
    result.reserve(rank);
    unsigned int d = 0;
    for (auto&& index_d : index) {
      TA_ASSERT(d < rank);
      result.push_back(ranges_[d].element_to_tile(index_d));
      ++d;
    }
    TA_ASSERT(d == rank);

    return result;
  }

  /// Convert an element index to a tile index

  /// \tparam Integer An integral type
  /// \param index The element index to convert
  /// \return The tile index that corresponds to the given element index
  template <typename Integer,
            typename = std::enable_if_t<std::is_integral_v<Integer>>>
  typename range_type::index element_to_tile(
      const std::initializer_list<Integer>& index) const {
    return this->element_to_tile<std::initializer_list<Integer>>(index);
  }

  /// The rank accessor

  /// \return the rank (=number of dimensions) of this object
  std::size_t rank() const { return ranges_.size(); }

  /// Accessor of the tiled range for one of the dimensions

  /// \param d the dimension index, a nonnegative integer less than rank()
  /// \return TiledRange1 object for dimension \c d
  const TiledRange1& dim(std::size_t d) const {
    TA_ASSERT(d < rank());
    return ranges_[d];
  }

  /// \return iterator pointing to the beginning of the range of TiledRange1
  /// objects
  const_iterator begin() const { return ranges_.begin(); }

  /// \return iterator pointing to the end of the range of TiledRange1 objects
  const_iterator end() const { return ranges_.end(); }

  /// \param[in] d mode index
  /// \return const reference to the TiledRange1 object for mode \p d
  const TiledRange1& at(size_t d) const { return ranges_.at(d); }

  /// \return A reference to the array of TiledRange1 objects.
  /// \throw nothing
  const Ranges& data() const { return ranges_; }

  void swap(TiledRange_& other) {
    range_.swap(other.range_);
    elements_range_.swap(other.elements_range_);
    std::swap(ranges_, other.ranges_);
  }

  /// Shifts the lower and upper bounds of this range

  /// \tparam Index An integral range type
  /// \param bound_shift The shift to be applied to the range
  /// \return A reference to this range
  template <typename Index,
            typename = std::enable_if_t<detail::is_integral_range_v<Index>>>
  TiledRange_& inplace_shift(const Index& bound_shift) {
    elements_range_.inplace_shift(bound_shift);
    using std::begin;
    auto bound_shift_it = begin(bound_shift);
    for (std::size_t d = 0; d != rank(); ++d, ++bound_shift_it) {
      ranges_[d].inplace_shift(*bound_shift_it);
    }
    return *this;
  }

  /// Shifts the lower and upper bound of this range

  /// \tparam Index An integral type
  /// \param bound_shift The shift to be applied to the range
  /// \return A reference to this range
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  TiledRange_& inplace_shift(const std::initializer_list<Index>& bound_shift) {
    return inplace_shift<std::initializer_list<Index>>(bound_shift);
  }

  /// Create a TiledRange with shifted lower and upper bounds

  /// \tparam Index An integral range type
  /// \param bound_shift The shift to be applied to the range
  /// \return A shifted copy of this range
  template <typename Index,
            typename = std::enable_if_t<detail::is_integral_range_v<Index>>>
  [[nodiscard]] TiledRange_ shift(const Index& bound_shift) const {
    TiledRange_ result(*this);
    result.inplace_shift(bound_shift);
    return result;
  }

  /// Create a TiledRange with shifted lower and upper bounds

  /// \tparam Index An integral type
  /// \param bound_shift The shift to be applied to the range
  /// \return A shifted copy of this range
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  [[nodiscard]] TiledRange_ shift(
      const std::initializer_list<Index>& bound_shift) const {
    TiledRange_ result(*this);
    result.inplace_shift(bound_shift);
    return result;
  }

  template <typename Archive,
            typename std::enable_if<madness::is_input_archive_v<
                std::decay_t<Archive>>>::type* = nullptr>
  void serialize(Archive& ar) {
    ar & range_ & elements_range_ & ranges_;
  }

  template <typename Archive,
            typename std::enable_if<madness::is_output_archive_v<
                std::decay_t<Archive>>>::type* = nullptr>
  void serialize(Archive& ar) const {
    ar & range_ & elements_range_ & ranges_;
  }

 private:
  range_type range_;           ///< Range of tile indices
  range_type elements_range_;  ///< Range of element indices
  Ranges ranges_;  ///< tiled (1d) range, aka TiledRange1, for each mode
                   ///< `*this` is a direct product of these tilings
};

/// TiledRange permutation operator.

/// This function will permute the range. Note: only tiles that are not
/// being used by other objects will be permuted. The owner of those
/// objects are
inline TiledRange operator*(const Permutation& p, const TiledRange& r) {
  TA_ASSERT(r.tiles_range().rank() == p.size());
  TiledRange::Ranges data = p * r.data();

  return TiledRange(data.begin(), data.end());
}

/// Exchange the content of the two given TiledRange's.
inline void swap(TiledRange& r0, TiledRange& r1) { r0.swap(r1); }

/// Returns true when all tile and element ranges are the same.
inline bool operator==(const TiledRange& r1, const TiledRange& r2) {
  return (r1.tiles_range().rank() == r2.tiles_range().rank()) &&
         (r1.tiles_range() == r2.tiles_range()) &&
         (r1.elements_range() == r2.elements_range()) &&
         std::equal(r1.data().begin(), r1.data().end(), r2.data().begin());
}

inline bool operator!=(const TiledRange& r1, const TiledRange& r2) {
  return !operator==(r1, r2);
}

inline std::ostream& operator<<(std::ostream& out, const TiledRange& rng) {
  out << "("
      << " tiles = " << rng.tiles_range()
      << ", elements = " << rng.elements_range() << " )";
  return out;
}

}  // namespace TiledArray

#endif  // TILEDARRAY_TILED_RANGE_H__INCLUDED
