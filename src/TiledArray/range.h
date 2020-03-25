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

#ifndef TILEDARRAY_RANGE_H__INCLUDED
#define TILEDARRAY_RANGE_H__INCLUDED

#include <TiledArray/permutation.h>
#include <TiledArray/range_iterator.h>
#include <TiledArray/size_array.h>
#include <TiledArray/util/container.h>

namespace TiledArray {

/// \brief A (hyperrectangular) interval on \f$ Z^n \f$, space of integer
/// n-indices

/// This object represents an n-dimensional, hyperrectangular array
/// of integers. It provides information on the rank (number of dimensions),
/// (nonnegative) lower bound, upper bound, extent (size), and stride of
/// each dimension. It can also be used to
/// test if an element is included in the range with a coordinate index or
/// ordinal offset. Finally, it can be used to convert coordinate indices to
/// ordinal offsets and vice versa.
/// TODO add Range support for negative indices
class Range {
 public:
  typedef Range Range_;                         ///< This object type
  typedef std::size_t size_type;                ///< Size type
  typedef container::svector<size_type> index;  ///< Coordinate index type
  typedef index index_type;  ///< Coordinate index type, to conform Tensor
                             ///< Working Group spec
  typedef detail::SizeArray<const size_type> size_array;  ///< Size array type
  typedef size_array
      extent_type;  ///< Range extent type, to conform Tensor Working Group spec
  typedef std::size_t
      ordinal_type;  ///< Ordinal type, to conform Tensor Working Group spec
  typedef detail::RangeIterator<size_type, Range_>
      const_iterator;  ///< Coordinate iterator
  friend class detail::RangeIterator<size_type, Range_>;

 protected:
  size_type* data_ = nullptr;
  ///< An array that holds the dimension information of the
  ///< range. The layout of the array is:
  ///< \code
  ///< { lobound[0], ..., lobound[rank_ - 1],
  ///<   upbound[0], ..., upbound[rank_ - 1],
  ///<   extent[0],  ..., extent[rank_ - 1],
  ///<   stride[0],  ..., stride[rank_ - 1] }
  ///< \endcode
  size_type offset_ = 0ul;  ///< Ordinal index offset correction
  size_type volume_ = 0ul;  ///< Total number of elements
  unsigned int rank_ = 0u;  ///< The rank (or number of dimensions) in the range

 private:
  /// Initialize range data from sequences of lower and upper bounds

  /// \tparam Index1 An array type
  /// \tparam Index2 An array type
  /// \param lower_bound The lower bound of the range
  /// \param upper_bound The upper bound of the range
  /// \pre Assume \c rank_ is initialized to the rank of the range and
  /// \c data_ has been allocated to hold 4*rank_ elements
  /// \post \c data_ and \c volume_ are initialized with range dimension
  /// information.
  template <typename Index1, typename Index2>
  void init_range_data(const Index1& lower_bound, const Index2& upper_bound) {
    // Construct temp pointers
    size_type* MADNESS_RESTRICT const lower = data_;
    size_type* MADNESS_RESTRICT const upper = lower + rank_;
    size_type* MADNESS_RESTRICT const extent = upper + rank_;
    size_type* MADNESS_RESTRICT const stride = extent + rank_;
    using std::data;
    const auto* MADNESS_RESTRICT const lower_data = data(lower_bound);
    const auto* MADNESS_RESTRICT const upper_data = data(upper_bound);

    // Set the volume seed
    volume_ = 1ul;
    offset_ = 0ul;

    // Compute range data
    for (int i = int(rank_) - 1; i >= 0; --i) {
      // Check input dimensions
      TA_ASSERT(lower_data[i] <= upper_data[i]);

      // Compute data for element i of lower, upper, and extent
      const size_type lower_bound_i = lower_data[i];
      const size_type upper_bound_i = upper_data[i];
      const size_type extent_i = upper_bound_i - lower_bound_i;

      lower[i] = lower_bound_i;
      upper[i] = upper_bound_i;
      extent[i] = extent_i;
      stride[i] = volume_;
      offset_ += lower_bound_i * volume_;
      volume_ *= extent_i;
    }
  }

  /// Initialize range data from a sequence of {lower,upper} bound pairs

  /// \tparam Index An array type
  /// \param lower_bound The lower bound of the range
  /// \param upper_bound The upper bound of the range
  /// \pre Assume \c rank_ is initialized to the rank of the range and
  /// \c data_ has been allocated to hold 4*rank_ elements
  /// \post \c data_ and \c volume_ are initialized with range dimension
  /// information.
  template <typename Index,
            typename std::enable_if<detail::is_pair<
                typename Index::value_type>::value>::type* = nullptr>
  void init_range_data(const Index& bound) {
    // Construct temp pointers
    size_type* MADNESS_RESTRICT const lower = data_;
    size_type* MADNESS_RESTRICT const upper = lower + rank_;
    size_type* MADNESS_RESTRICT const extent = upper + rank_;
    size_type* MADNESS_RESTRICT const stride = extent + rank_;
    using std::data;
    const auto* MADNESS_RESTRICT const bound_data = data(bound);

    // Set the volume seed
    volume_ = 1ul;
    offset_ = 0ul;

    // Compute range data
    for (int i = int(rank_) - 1; i >= 0; --i) {
      // Compute data for element i of lower, upper, and extent
      const size_type lower_bound_i = bound_data[i].first;
      const size_type upper_bound_i = bound_data[i].second;
      const size_type extent_i = upper_bound_i - lower_bound_i;

      // Check input dimensions
      TA_ASSERT(lower_bound_i >= 0ul);
      TA_ASSERT(lower_bound_i <= upper_bound_i);

      lower[i] = lower_bound_i;
      upper[i] = upper_bound_i;
      extent[i] = extent_i;
      stride[i] = volume_;
      offset_ += lower_bound_i * volume_;
      volume_ *= extent_i;
    }
  }

  /// Initialize range data from a sequence of extents

  /// \tparam Index An array type
  /// \param extents A sequence of extents for each dimension
  /// \pre Assume \c rank_ is initialized to the rank of the range and
  /// \c data_ has been allocated to hold 4*rank_ elements
  /// \post \c data_ and \c volume_ are initialized with range dimension
  /// information.
  template <typename Index,
            typename std::enable_if<detail::is_integral_list<
                typename Index::value_type>::value>::type* = nullptr>
  void init_range_data(const Index& extents) {
    // Construct temp pointers
    size_type* MADNESS_RESTRICT const lower = data_;
    size_type* MADNESS_RESTRICT const upper = lower + rank_;
    size_type* MADNESS_RESTRICT const extent = upper + rank_;
    size_type* MADNESS_RESTRICT const stride = extent + rank_;
    using std::data;
    const auto* MADNESS_RESTRICT const extent_data = data(extents);

    // Set the offset and volume initial values
    volume_ = 1ul;
    offset_ = 0ul;

    // Compute range data
    for (int i = int(rank_) - 1; i >= 0; --i) {
      // Check bounds of the input extent
      TA_ASSERT(extent_data[i] >= 0);

      // Get extent i
      const size_type extent_i = extent_data[i];

      lower[i] = 0ul;
      upper[i] = extent_i;
      extent[i] = extent_i;
      stride[i] = volume_;
      volume_ *= extent_i;
    }
  }

  /// Initialize range data from a tuple of extents

  /// \tparam Index An array type
  /// \param extents A tuple of extents for each dimension
  /// \pre Assume \c rank_ is initialized to the rank of the range and
  /// \c data_ has been allocated to hold 4*rank_ elements
  /// \post \c data_ and \c volume_ are initialized with range dimension
  /// information.
  template <typename... Indices,
            typename std::enable_if<
                detail::is_integral_list<Indices...>::value>::type* = nullptr>
  void init_range_data(const std::tuple<Indices...>& extents) {
    const constexpr std::size_t rank =
        std::tuple_size<std::tuple<Indices...>>::value;
    TA_ASSERT(rank_ == rank);

    // Set the offset and volume initial values
    volume_ = 1ul;
    offset_ = 0ul;

    // initialize by recursion
    init_range_data_helper(extents, std::make_index_sequence<rank>{});
  }

  template <typename... Indices, std::size_t... Is>
  void init_range_data_helper(const std::tuple<Indices...>& extents,
                              std::index_sequence<Is...>) {
    int workers[] = {0, (init_range_data_helper_iter<Is>(extents), 0)...};
    ++workers[0];
  }

  template <std::size_t I, typename... Indices>
  void init_range_data_helper_iter(const std::tuple<Indices...>& extents) {
    // Check bounds of the input extent
    TA_ASSERT(std::get<I>(extents) >= 0ul);

    // Get extent i
    const size_type extent_i = std::get<I>(extents);

    size_type* MADNESS_RESTRICT const lower = data_;
    size_type* MADNESS_RESTRICT const upper = lower + rank_;
    size_type* MADNESS_RESTRICT const extent = upper + rank_;
    size_type* MADNESS_RESTRICT const stride = extent + rank_;

    lower[I] = 0ul;
    upper[I] = extent_i;
    extent[I] = extent_i;
    stride[I] = volume_;
    volume_ *= extent_i;
  }

  /// Initialize permuted range data from lower and upper bounds

  /// \param other_lower_bound The lower bound of the unpermuted range
  /// \param other_upper_bound The upper bound of the unpermuted range
  /// \pre Assume \c rank_ is initialized to the rank of the range and
  /// \c data_ has been allocated to hold 4*rank_ elements
  /// \post \c data_, \c offset_, and \c volume_ are initialized with the
  /// permuted range dimension information from \c other_lower_bound and
  /// \c other_upper_bound.
  void init_range_data(
      const Permutation& perm,
      const size_type* MADNESS_RESTRICT const other_lower_bound,
      const size_type* MADNESS_RESTRICT const other_upper_bound) {
    // Create temporary pointers to this range data
    auto* MADNESS_RESTRICT const lower = data_;
    auto* MADNESS_RESTRICT const upper = lower + rank_;
    auto* MADNESS_RESTRICT const extent = upper + rank_;
    auto* MADNESS_RESTRICT const stride = extent + rank_;

    // Copy the permuted lower, upper, and extent into this range.
    for (unsigned int i = 0u; i < rank_; ++i) {
      const auto perm_i = perm[i];

      // Get the lower bound, upper bound, and extent from other for rank i.
      const auto other_lower_bound_i = other_lower_bound[i];
      const auto other_upper_bound_i = other_upper_bound[i];
      const auto other_extent_i = other_upper_bound_i - other_lower_bound_i;

      // Store the permuted lower bound, upper bound, and extent
      lower[perm_i] = other_lower_bound_i;
      upper[perm_i] = other_upper_bound_i;
      extent[perm_i] = other_extent_i;
    }

    // Recompute stride, offset, and volume
    volume_ = 1ul;
    offset_ = 0ul;
    for (int i = int(rank_) - 1; i >= 0; --i) {
      const auto lower_i = lower[i];
      const auto extent_i = extent[i];
      stride[i] = volume_;
      offset_ += lower_i * volume_;
      volume_ *= extent_i;
    }
  }

 public:
  /// Default constructor

  /// Construct a range that has zero rank, volume, and size.
  Range() {}

  /// Construct range defined by upper and lower bound sequences

  /// Construct a range defined by \c lower_bound and \c upper_bound.
  /// \tparam Index An array type
  /// \param lower_bound A sequence of lower bounds for each dimension
  /// \param upper_bound A sequence of upper bounds for each dimension
  /// \throw TiledArray::Exception When the size of \c lower_bound is not
  /// equal to that of \c upper_bound.
  /// \throw TiledArray::Exception When lower_bound[i] >= upper_bound[i]
  /// \throw std::bad_alloc When memory allocation fails.
  template <
      typename Index1, typename Index2,
      typename std::enable_if_t<!std::is_integral<Index1>::value &&
                                !std::is_integral<Index2>::value>* = nullptr>
  Range(const Index1& lower_bound, const Index2& upper_bound) {
    using std::size;
    const size_type n = size(lower_bound);
    TA_ASSERT(n == size(upper_bound));
    if (n) {
      // Initialize array memory
      data_ = new size_type[n << 2];
      rank_ = n;
      init_range_data(lower_bound, upper_bound);
    }
  }

  /// Construct range defined by the upper and lower bound sequences

  /// Construct a range defined by \c lower_bound and \c upper_bound.
  /// \param lower_bound An initializer list of lower bounds for each dimension
  /// \param upper_bound An initializer list of upper bounds for each dimension
  /// \throw TiledArray::Exception When the size of \c lower_bound is not
  /// equal to that of \c upper_bound.
  /// \throw TiledArray::Exception When lower_bound[i] >= upper_bound[i]
  /// \throw std::bad_alloc When memory allocation fails.
  template <
      typename Index1,
      typename std::enable_if<std::is_integral<Index1>::value>::type* = nullptr>
  Range(const std::initializer_list<Index1>& lower_bound,
        const std::initializer_list<Index1>& upper_bound) {
    using std::size;
    const size_type n = size(lower_bound);
    TA_ASSERT(n == size(upper_bound));
    if (n) {
      // Initialize array memory
      data_ = new size_type[n << 2];
      rank_ = n;
      init_range_data(lower_bound, upper_bound);
    }
  }

  /// Range constructor from a sequence of extents

  /// Construct a range with a lower bound of zero and an upper bound equal to
  /// \c extents.
  /// \tparam Index A vector type
  /// \param extent A vector that defines the size of each dimension
  /// \throw std::bad_alloc When memory allocation fails.
  template <
      typename Index,
      typename std::enable_if<
          !std::is_integral<Index>::value &&
          std::is_integral<typename Index::value_type>::value>::type* = nullptr>
  explicit Range(const Index& extent) {
    using std::size;
    const size_type n = size(extent);
    if (n) {
      // Initialize array memory
      data_ = new size_type[n << 2];
      rank_ = n;
      init_range_data(extent);
    }
  }

  /// Range constructor from an initializer list of extents

  /// Construct a range with a lower bound of zero and an upper bound equal to
  /// \c extent.
  /// \param extent An initializer list that defines the size of each dimension
  /// \throw std::bad_alloc When memory allocation fails.
  template <
      typename Index1,
      typename std::enable_if<std::is_integral<Index1>::value>::type* = nullptr>
  explicit Range(const std::initializer_list<Index1>& extent) {
    using std::size;
    const size_type n = size(extent);
    if (n) {
      // Initialize array memory
      data_ = new size_type[n << 2];
      rank_ = n;
      init_range_data(extent);
    }
  }

  /// Construct range defined by a sequence of {lower,upper} bound pairs

  /// \tparam Index An array (contiguous sequence) type
  /// \param lower_bound A sequence of lower bounds for each dimension
  /// \param upper_bound A sequence of upper bounds for each dimension
  /// \throw TiledArray::Exception When the size of \c lower_bound is not
  /// equal to that of \c upper_bound.
  /// \throw TiledArray::Exception When lower_bound[i] >= upper_bound[i]
  /// \throw std::bad_alloc When memory allocation fails.
  template <
      typename Index,
      typename std::enable_if<
          !std::is_integral<Index>::value &&
          detail::is_pair<typename Index::value_type>::value>::type* = nullptr>
  Range(const Index& bounds) {
    using std::size;
    const size_type n = size(bounds);
    if (n) {
      // Initialize array memory
      data_ = new size_type[n << 2];
      rank_ = n;
      init_range_data(bounds);
    }
  }

  /// Construct range defined by an initializer_list of {lower,upper} bound
  /// pairs

  /// \tparam Index An initializer list type
  /// \param lower_bound A sequence of lower bounds for each dimension
  /// \param upper_bound A sequence of upper bounds for each dimension
  /// \throw TiledArray::Exception When the size of \c lower_bound is not
  /// equal to that of \c upper_bound.
  /// \throw TiledArray::Exception When lower_bound[i] >= upper_bound[i]
  /// \throw std::bad_alloc When memory allocation fails.
  template <typename Index1, typename Index2>
  Range(const std::initializer_list<std::pair<Index1, Index2>>& bounds) {
    using std::size;
    const size_type n = size(bounds);
    if (n) {
      // Initialize array memory
      data_ = new size_type[n << 2];
      rank_ = n;
      init_range_data(bounds);
    }
  }

  /// Range constructor from a pack of extents for each dimension

  /// \tparam Index Pack of integer types
  /// \param extents A pack of extents for each dimension
  /// \post Range has a lower bound of 0, and an upper bound of \c (extents...).
  /// \throw std::bad_alloc When memory allocation fails.
  template <typename... Index, typename std::enable_if<detail::is_integral_list<
                                   Index...>::value>::type* = nullptr>
  explicit Range(const Index... extents)
      : Range(std::array<size_t, sizeof...(Index)>{
            {static_cast<std::size_t>(extents)...}}) {}

  /// Range constructor from a pack of {lo,up} bounds for each dimension

  /// \tparam IndexPairs Pack of std::pair's of integer types
  /// \param extents A pack of pairs of lobound and upbound for each dimension
  /// \throw std::bad_alloc When memory allocation fails.
  template <typename... IndexPairs,
            std::enable_if_t<detail::is_integral_pair_list_v<IndexPairs...>>* =
                nullptr>
  explicit Range(const IndexPairs... bounds)
      : Range(std::array<std::pair<std::size_t, std::size_t>,
                         sizeof...(IndexPairs)>{
            {static_cast<std::pair<std::size_t, std::size_t>>(bounds)...}}) {}

  /// Copy Constructor

  /// \param other The range to be copied
  /// \throw std::bad_alloc When memory allocation fails.
  Range(const Range_& other) {
    if (other.rank_ > 0ul) {
      data_ = new size_type[other.rank_ << 2];
      offset_ = other.offset_;
      volume_ = other.volume_;
      rank_ = other.rank_;
      memcpy(data_, other.data_, (sizeof(size_type) << 2) * other.rank_);
    }
  }

  /// Copy Constructor

  /// \param other The range to be copied
  /// \throw std::bad_alloc When memory allocation fails.
  Range(Range_&& other)
      : data_(other.data_),
        offset_(other.offset_),
        volume_(other.volume_),
        rank_(other.rank_) {
    other.data_ = nullptr;
    other.offset_ = 0ul;
    other.volume_ = 0ul;
    other.rank_ = 0u;
  }

  /// Permuting copy constructor

  /// \param perm The permutation applied to other
  /// \param other The range to be permuted and copied
  /// \throw std::bad_alloc When memory allocation fails.
  Range(const Permutation& perm, const Range_& other) {
    TA_ASSERT(perm.dim() == other.rank_);

    if (other.rank_ > 0ul) {
      data_ = new size_type[other.rank_ << 2];
      rank_ = other.rank_;

      if (perm) {
        init_range_data(perm, other.data_, other.data_ + rank_);
      } else {
        // Simple copy will do
        memcpy(data_, other.data_, (sizeof(size_type) << 2) * rank_);
        offset_ = other.offset_;
        volume_ = other.volume_;
      }
    }
  }

  /// Destructor
  ~Range() { delete[] data_; }

  /// Copy assignment operator

  /// \param other The range to be copied
  /// \return A reference to this object
  /// \throw std::bad_alloc When memory allocation fails.
  Range_& operator=(const Range_& other) {
    if (rank_ != other.rank_) {
      delete[] data_;
      data_ = (other.rank_ > 0ul ? new size_type[other.rank_ << 2] : nullptr);
      rank_ = other.rank_;
    }
    memcpy(data_, other.data_, (sizeof(size_type) << 2) * rank_);
    offset_ = other.offset_;
    volume_ = other.volume_;

    return *this;
  }

  /// Move assignment operator

  /// \param other The range to be copied
  /// \return A reference to this object
  /// \throw nothing
  Range_& operator=(Range_&& other) {
    data_ = other.data_;
    offset_ = other.offset_;
    volume_ = other.volume_;
    rank_ = other.rank_;

    other.data_ = nullptr;
    other.offset_ = 0ul;
    other.volume_ = 0ul;
    other.rank_ = 0u;

    return *this;
  }

  /// Rank accessor

  /// \return The rank (number of dimensions) of this range
  /// \throw nothing
  unsigned int rank() const { return rank_; }

  /// Accessor of the \c d-th dimension of the range

  /// \param d the dimension index, a nonnegative integer less than rank()
  /// \return the pair of {lower,upper} bounds for dimension \c d
  std::pair<size_type, size_type> dim(std::size_t d) const {
    TA_ASSERT(d < rank());
    return std::make_pair(lobound(d), upbound(d));
  }

  /// Range lower bound data accessor

  /// \return A pointer to the lower bound data (see <tt>lobound()</tt>)
  /// \throw nothing
  const size_type* lobound_data() const { return data_; }

  /// Range lower bound accessor

  /// \return A \c size_array that contains the lower bounds for each
  /// dimension of the block range.
  /// \throw nothing
  size_array lobound() const { return size_array(lobound_data(), rank_); }

  /// Range lower bound element accessor

  /// \return The lower bound of dimension \c dim.
  /// \throw nothing
  size_type lobound(size_t dim) const {
    TA_ASSERT(dim < rank_);
    return *(lobound_data() + dim);
  }

  /// Range upper bound data accessor

  /// \return A pointer to the upper bound data (see <tt>upbound()</tt>)
  /// \throw nothing
  const size_type* upbound_data() const { return data_ + rank_; }

  /// Range upper bound accessor

  /// \return A \c size_array that contains the upper bounds for each
  /// dimension of the block range.
  /// \throw nothing
  size_array upbound() const { return size_array(upbound_data(), rank_); }

  /// Range upped bound element accessor

  /// \return The upper bound of dimension \c dim.
  /// \throw nothing
  size_type upbound(size_t dim) const {
    TA_ASSERT(dim < rank_);
    return *(upbound_data() + dim);
  }

  /// Range extent data accessor

  /// \return A pointer to the extent data (see <tt>extent()</tt>)
  /// \throw nothing
  const size_type* extent_data() const { return data_ + (rank_ + rank_); }

  /// Range extent accessor

  /// \return A \c extent_type that contains the extent for each dimension of
  /// the block range.
  /// \throw nothing
  extent_type extent() const { return size_array(extent_data(), rank_); }

  /// Range extent element accessor

  /// \return The extent of dimension \c dim.
  /// \throw nothing
  size_type extent(size_t dim) const {
    TA_ASSERT(dim < rank_);
    return *(extent_data() + dim);
  }

  /// Range stride data accessor

  /// \return A pointer to the stride data (see <tt>stride()</tt>)
  /// \throw nothing
  const size_type* stride_data() const {
    return data_ + (rank_ + rank_ + rank_);
  }

  /// Range stride accessor

  /// \return A \c size_array that contains the stride for each dimension of
  /// the block range.
  /// \throw nothing
  size_array stride() const { return size_array(stride_data(), rank_); }

  /// Range stride element accessor

  /// \return The stride of dimension \c dim.
  /// \throw nothing
  size_type stride(size_t dim) const {
    TA_ASSERT(dim < rank_);
    return *(stride_data() + dim);
  }

  /// Range volume accessor

  /// \return The total number of elements in the range.
  /// \throw nothing
  ordinal_type volume() const { return volume_; }

  /// alias to volume() to conform to the Tensor Working Group specification
  /// \return The total number of elements in the range.
  /// \throw nothing
  ordinal_type area() const { return volume_; }

  /// Range offset

  /// The range ordinal offset is equal to the dot product of the lower bound
  /// and stride vector. It is used internally to compute ordinal offsets.
  /// \return The ordinal index offset
  ordinal_type offset() const { return offset_; }

  /// Index iterator factory

  /// The iterator dereferences to an index. The order of iteration matches
  /// the data layout of a row-major tensor.
  /// \return An iterator that holds the lower bound index of a tensor (unless
  /// it has zero volume, then it returns same result as end()) \throw nothing
  const_iterator begin() const {
    return (volume_ > 0) ? const_iterator(data_, this) : end();
  }

  /// Index iterator factory

  /// The iterator dereferences to an index. The order of iteration matches
  /// the data layout of a row-major tensor.
  /// \return An iterator that holds the upper bound element index of a tensor
  /// \throw nothing
  const_iterator end() const { return const_iterator(data_ + rank_, this); }

  /// Check the coordinate to make sure it is within the range.

  /// \tparam Index The coordinate index array type
  /// \param index The coordinate index to check for inclusion in the range
  /// \return \c true when <tt>i >= lobound</tt> and <tt>i < upbound</tt>,
  /// otherwise \c false
  /// \throw TiledArray::Exception When the rank of this range is not
  /// equal to the size of the index.
  template <typename Index,
            typename std::enable_if<!std::is_integral<Index>::value,
                                    bool>::type* = nullptr>
  bool includes(const Index& index) const {
    using std::size;
    TA_ASSERT(size(index) == rank_);
    const size_type* MADNESS_RESTRICT const lower = data_;
    const size_type* MADNESS_RESTRICT const upper = lower + rank_;

    bool result = (rank_ > 0u);
    using std::cbegin;
    auto it = cbegin(index);
    for (unsigned int i = 0u; result && (i < rank_); ++i, ++it) {
      const size_type index_i = *it;
      const size_type lower_i = lower[i];
      const size_type upper_i = upper[i];
      result = result && (index_i >= lower_i) && (index_i < upper_i);
    }

    return result;
  }

  /// Check the coordinate to make sure it is within the range.

  /// \tparam Integer An integer type
  /// \param index The element index to check for inclusion in the range,
  ///              as an \c std::initializer_list<Integer>
  /// \return \c true when <tt>i >= lobound</tt> and <tt>i < upbound</tt>,
  /// otherwise \c false
  /// \throw TiledArray::Exception When the rank of this range is not
  /// equal to the size of the index.
  template <typename Integer>
  bool includes(const std::initializer_list<Integer>& index) const {
    return includes<std::initializer_list<Integer>>(index);
  }

  /// Check the ordinal index to make sure it is within the range.

  /// \param i The ordinal index to check for inclusion in the range
  /// \return \c true when \c i \c >= \c 0 and \c i \c < \c volume
  /// \throw nothing
  template <typename Ordinal>
  typename std::enable_if<std::is_integral<Ordinal>::value, bool>::type
  includes(Ordinal i) const {
    return include_ordinal_(i);
  }

  template <typename... Index>
  typename std::enable_if<(sizeof...(Index) > 1ul), size_type>::type includes(
      const Index&... index) const {
    const size_type i[sizeof...(Index)] = {static_cast<size_type>(index)...};
    return includes(i);
  }

  /// Permute this range

  /// \param perm The permutation to be applied to this range
  /// \return A reference to this range
  /// \throw TiledArray::Exception When the rank of this range is not
  /// equal to the rank of the permutation.
  /// \throw std::bad_alloc When memory allocation fails.
  Range_& operator*=(const Permutation& perm);

  /// Resize range to a new upper and lower bound

  /// \tparam Index An array type
  /// \param lower_bound The lower bounds of the N-dimensional range
  /// \param upper_bound The upper bound of the N-dimensional range
  /// \return A reference to this range
  /// \throw TiledArray::Exception When the size of \c lower_bound is not
  /// equal to that of \c upper_bound.
  /// \throw TiledArray::Exception When <tt>lower_bound[i] >=
  /// upper_bound[i]</tt> \throw std::bad_alloc When memory allocation fails.
  template <typename Index>
  Range_& resize(const Index& lower_bound, const Index& upper_bound) {
    using std::size;
    const size_type n = size(lower_bound);
    TA_ASSERT(n == size(upper_bound));

    // Reallocate memory for range arrays
    if (rank_ != n) {
      delete[] data_;
      data_ = (n > 0ul ? new size_type[n << 2] : nullptr);
      rank_ = n;
    }
    if (n > 0ul)
      init_range_data(lower_bound, upper_bound);
    else
      volume_ = 0ul;

    return *this;
  }

  /// Shift the lower and upper bound of this range

  /// \tparam Index The shift array type
  /// \param bound_shift The shift to be applied to the range
  /// \return A reference to this range
  template <typename Index>
  Range_& inplace_shift(const Index& bound_shift) {
    using std::size;
    TA_ASSERT(size(bound_shift) == rank_);

    using std::data;
    const auto* MADNESS_RESTRICT const bound_shift_data = data(bound_shift);
    size_type* MADNESS_RESTRICT const lower = data_;
    size_type* MADNESS_RESTRICT const upper = data_ + rank_;
    const size_type* MADNESS_RESTRICT const stride = upper + rank_ + rank_;

    offset_ = 0ul;
    for (unsigned i = 0u; i < rank_; ++i) {
      // Load range data
      const auto bound_shift_i = bound_shift_data[i];
      auto lower_i = lower[i];
      auto upper_i = upper[i];
      const auto stride_i = stride[i];

      // Compute new range bounds
      lower_i += bound_shift_i;
      upper_i += bound_shift_i;

      // Update range data
      offset_ += lower_i * stride_i;
      lower[i] = lower_i;
      upper[i] = upper_i;
    }

    return *this;
  }

  /// Shift the lower and upper bound of this range

  /// \tparam Index The shift array type
  /// \param bound_shift The shift to be applied to the range
  /// \return A reference to this range
  template <typename I>
  Range_& inplace_shift(const std::initializer_list<I>& bound_shift) {
    return inplace_shift<std::initializer_list<I>>(bound_shift);
  }

  /// Shift the lower and upper bound of this range

  /// \tparam Index The shift array type
  /// \param bound_shift The shift to be applied to the range
  /// \return A shifted copy of this range
  template <typename Index>
  Range_ shift(const Index& bound_shift) {
    Range_ result(*this);
    result.inplace_shift(bound_shift);
    return result;
  }

  /// calculate the ordinal index of \c i

  /// This function is just a pass-through so the user can call \c ordinal() on
  /// a template parameter that can be a coordinate index or an integral.
  /// \param index Ordinal index
  /// \return \c index (unchanged)
  /// \throw When \c index is not included in this range
  ordinal_type ordinal(const ordinal_type index) const {
    TA_ASSERT(includes(index));
    return index;
  }

  /// calculate the ordinal index of \c index

  /// Convert a coordinate index to an ordinal index.
  /// \tparam Index A coordinate index type (array type)
  /// \param index The index to be converted to an ordinal index
  /// \return The ordinal index of \c index
  /// \throw When \c index is not included in this range.
  template <
      typename Index,
      typename std::enable_if<!std::is_integral<Index>::value>::type* = nullptr>
  ordinal_type ordinal(const Index& index) const {
    using std::size;
    TA_ASSERT(size(index) == rank_);
    TA_ASSERT(includes(index));

    size_type* MADNESS_RESTRICT const stride = data_ + rank_ + rank_ + rank_;

    size_type result = 0ul;
    using std::cbegin;
    auto index_it = cbegin(index);
    for (unsigned int i = 0u; i < rank_; ++i, ++index_it) {
      const size_type stride_i = stride[i];
      result += *(index_it)*stride_i;
    }

    return result - offset_;
  }

  /// calculate the ordinal index of \c index

  /// Convert a coordinate index to an ordinal index.
  /// \tparam Index A coordinate index type (array type)
  /// \param index The index to be converted to an ordinal index
  /// \return The ordinal index of \c index
  /// \throw When \c index is not included in this range.
  template <typename... Index,
            typename std::enable_if<(sizeof...(Index) > 1ul)>::type* = nullptr>
  size_type ordinal(const Index&... index) const {
    const size_type temp_index[sizeof...(Index)] = {
        static_cast<size_type>(index)...};
    return ordinal(temp_index);
  }

  /// calculate the coordinate index of the ordinal index, \c index.

  /// Convert an ordinal index to a coordinate index.
  /// \param index Ordinal index
  /// \return The index of the ordinal index
  /// \throw TiledArray::Exception When \c index is not included in this range
  /// \throw std::bad_alloc When memory allocation fails
  index idx(size_type index) const {
    // Check that index is contained by range.
    // N.B. this will fail if any extent is zero
    TA_ASSERT(includes(index));

    // Construct result coordinate index object and allocate its memory.
    Range_::index result(rank_, 0);

    // Get pointers to the data
    size_type* MADNESS_RESTRICT const result_data = result.data();
    size_type const* MADNESS_RESTRICT const lower = data_;
    size_type const* MADNESS_RESTRICT const size = data_ + rank_ + rank_;

    // Compute the coordinate index of index in range.
    for (int i = int(rank_) - 1; i >= 0; --i) {
      const size_type lower_i = lower[i];
      const size_type size_i = size[i];

      // Compute result index element i
      const size_type result_i = (index % size_i) + lower_i;
      index /= size_i;

      // Store result
      result_data[i] = result_i;
    }

    return result;
  }

  /// calculate the index of \c i

  /// This function is just a pass-through so the user can call \c idx() on
  /// a template parameter that can be an index or an ordinal_type.
  /// \param i The index
  /// \return \c i (unchanged)
  template <
      typename Index,
      typename std::enable_if<!std::is_integral<Index>::value>::type* = nullptr>
  const Index& idx(const Index& i) const {
    TA_ASSERT(includes(i));
    return i;
  }

  template <typename Archive,
            typename std::enable_if<madness::archive::is_input_archive<
                Archive>::value>::type* = nullptr>
  void serialize(const Archive& ar) {
    // Get rank
    unsigned int rank = 0ul;
    ar& rank;

    // Reallocate the array
    const unsigned int four_x_rank = rank << 2;
    if (rank_ != rank) {
      delete[] data_;
      data_ = (rank > 0u ? new size_type[four_x_rank] : nullptr);
      rank_ = rank;
    }

    // Get range data
    ar& madness::archive::wrap(data_, four_x_rank) & offset_& volume_;
  }

  template <typename Archive,
            typename std::enable_if<madness::archive::is_output_archive<
                Archive>::value>::type* = nullptr>
  void serialize(const Archive& ar) const {
    ar& rank_& madness::archive::wrap(data_, rank_ << 2) & offset_& volume_;
  }

  void swap(Range_& other) {
    // Get temp data
    std::swap(data_, other.data_);
    std::swap(offset_, other.offset_);
    std::swap(volume_, other.volume_);
    std::swap(rank_, other.rank_);
  }

 private:
  /// Check that a signed integral value is include in this range

  /// \tparam Index A signed integral type
  /// \param i The ordinal index to check
  /// \return \c true when <tt>i >= 0</tt> and <tt>i < volume_</tt>, otherwise
  /// \c false.
  template <typename Index>
  typename std::enable_if<std::is_signed<Index>::value, bool>::type
  include_ordinal_(Index i) const {
    return (i >= Index(0)) && (i < Index(volume_));
  }

  /// Check that an unsigned integral value is include in this range

  /// \tparam Index An unsigned integral type
  /// \param i The ordinal index to check
  /// \return \c true when  <tt>i < volume_</tt>, otherwise \c false.
  template <typename Index>
  typename std::enable_if<!std::is_signed<Index>::value, bool>::type
  include_ordinal_(Index i) const {
    return i < volume_;
  }

  /// Increment the coordinate index \c i in this range

  /// \param[in,out] i The coordinate index to be incremented
  /// \throw TiledArray::Exception When the rank of i is not equal to
  /// the rank of this range
  /// \throw TiledArray::Exception When \c i or \c i+n is outside this range
  void increment(index& i) const {
    TA_ASSERT(includes(i));

    size_type const* MADNESS_RESTRICT const lower = data_;
    size_type const* MADNESS_RESTRICT const upper = data_ + rank_;

    for (int d = int(rank_) - 1; d >= 0; --d) {
      // increment coordinate
      ++i[d];

      // break if done
      if (i[d] < upper[d]) return;

      // Reset current index to lower bound.
      i[d] = lower[d];
    }

    // if the current location was set to lower then it was at the end and
    // needs to be reset to equal upper.
    std::copy(upper, upper + rank_, i.begin());
  }

  /// Advance the coordinate index \c i by \c n in this range

  /// \param[in,out] i The coordinate index to be advanced
  /// \param n The distance to advance \c i
  /// \throw TiledArray::Exception When the rank of i is not equal to
  /// the rank of this range
  /// \throw TiledArray::Exception When \c i or \c i+n is outside this range
  void advance(index& i, std::ptrdiff_t n) const {
    TA_ASSERT(includes(i));
    const size_type o = ordinal(i) + n;
    TA_ASSERT(includes(o));
    i = idx(o);
  }

  /// Compute the distance between the coordinate indices \c first and \c last

  /// \param first The starting position in the range
  /// \param last The ending position in the range
  /// \return The difference between first and last, in terms of range positions
  /// \throw TiledArray::Exception When the size of \c first or \c last
  /// is not equal to the rank of this range
  /// \throw TiledArray::Exception When \c first or \c last is outside this
  /// range
  std::ptrdiff_t distance_to(const index& first, const index& last) const {
    TA_ASSERT(includes(first));
    TA_ASSERT(includes(last));
    return ordinal(last) - ordinal(first);
  }

};  // class Range

inline Range& Range::operator*=(const Permutation& perm) {
  TA_ASSERT(perm.dim() == rank_);
  if (rank_ > 1ul) {
    // Copy the lower and upper bound data into a temporary array
    size_type* MADNESS_RESTRICT const temp_lower = new size_type[rank_ << 1];
    const size_type* MADNESS_RESTRICT const temp_upper = temp_lower + rank_;
    std::memcpy(temp_lower, data_, (sizeof(size_type) << 1) * rank_);

    init_range_data(perm, temp_lower, temp_upper);

    // Cleanup old memory.
    delete[] temp_lower;
  }
  return *this;
}

/// Exchange the values of the give two ranges.
inline void swap(Range& r0, Range& r1) {  // no throw
  r0.swap(r1);
}

/// Create a permuted range

/// \param perm The permutation to be applied to the range
/// \param r The range to be permuted
/// \return A permuted copy of \c r.
inline Range operator*(const Permutation& perm, const Range& r) {
  return Range(perm, r);
}

/// Range equality comparison

/// \param r1 The first range to be compared
/// \param r2 The second range to be compared
/// \return \c true when \c r1 represents the same range as \c r2, otherwise
/// \c false.
inline bool operator==(const Range& r1, const Range& r2) {
  return (r1.rank() == r2.rank()) &&
         !std::memcmp(r1.lobound_data(), r2.lobound_data(),
                      r1.rank() * (2u * sizeof(Range::size_type)));
}
/// Range inequality comparison

/// \param r1 The first range to be compared
/// \param r2 The second range to be compared
/// \return \c true when \c r1 does not represent the same range as \c r2,
/// otherwise \c false.
inline bool operator!=(const Range& r1, const Range& r2) {
  return (r1.rank() != r2.rank()) ||
         std::memcmp(r1.lobound_data(), r2.lobound_data(),
                     r1.rank() * (2u * sizeof(Range::size_type)));
}

/// Range output operator

/// \param os The output stream that will be used to print \c r
/// \param r The range to be printed
/// \return A reference to the output stream
inline std::ostream& operator<<(std::ostream& os, const Range& r) {
  os << "[ ";
  detail::print_array(os, r.lobound_data(), r.rank());
  os << ", ";
  detail::print_array(os, r.upbound_data(), r.rank());
  os << " )";
  return os;
}

/// Test the two ranges are congruent

/// This function tests that the rank and extent of
/// \c r1 are equal to those of \c r2.
/// \param r1 The first Range to compare
/// \param r2 The second Range to compare
inline bool is_congruent(const Range& r1, const Range& r2) {
  return (r1.rank() == r2.rank()) &&
         std::equal(r1.extent_data(), r1.extent_data() + r1.rank(),
                    r2.extent_data());
}

}  // namespace TiledArray
#endif  // TILEDARRAY_RANGE_H__INCLUDED
