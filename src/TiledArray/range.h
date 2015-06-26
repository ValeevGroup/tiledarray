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

#include <TiledArray/range_iterator.h>
#include <TiledArray/permutation.h>
#include <TiledArray/size_array.h>

namespace TiledArray {

  /// Range data of an N-dimensional tensor.
  class Range {
  public:
    typedef Range Range_; ///< This object type
    typedef std::size_t size_type; ///< Size type
    typedef std::vector<size_type> index; ///< Coordinate index type
    typedef index index_type; ///< Coordinate index type, to conform Tensor Working Group spec
    typedef detail::SizeArray<const size_type> size_array; ///< Size array type
    typedef size_array extent_type;    ///< Range extent type, to conform Tensor Working Group spec
    typedef std::size_t ordinal_type; ///< Ordinal type, to conform Tensor Working Group spec
    typedef detail::RangeIterator<size_type, Range_> const_iterator; ///< Coordinate iterator
    friend class detail::RangeIterator<size_type, Range_>;

  protected:

    size_type* data_ = nullptr;
                      ///< An array that holds the dimension information of the
                      ///< range. The layout of the array is:
                      ///< \code
                      ///< { lower[0],  ..., lower[rank_ - 1],
                      ///<   upper[0],  ..., upper[rank_ - 1],
                      ///<   extent[0], ..., extent[rank_ - 1],
                      ///<   stride[0], ..., stride[rank_ - 1] }
                      ///< \endcode
    size_type offset_ = 0ul; ///< Ordinal index offset correction
    size_type volume_ = 0ul; ///< Total number of elements
    unsigned int rank_ = 0u; ///< The rank (or number of dimensions) in the range

  private:
    struct Enabler {};

    /// Initialize range data from lower and upper bounds

    /// \tparam Index An array type
    /// \param lower_bound The lower bound of the range
    /// \param upper_bound The upper bound of the range
    /// \pre Assume \c rank_ is initialized to the rank of the range and
    /// \c data_ has been allocated to hold 4*rank_ elements
    /// \post \c data_ and \c volume_ are initialized with range dimension
    /// information.
    template <typename Index>
    void init_range_data(const Index& lower_bound, const Index& upper_bound) {
      // Construct temp pointers
      size_type* restrict const lower  = data_;
      size_type* restrict const upper  = lower + rank_;
      size_type* restrict const extent = upper + rank_;
      size_type* restrict const stride = extent + rank_;

      // Set the volume seed
      volume_ = 1ul;
      offset_ = 0ul;

      // Compute range data
      for(int i = int(rank_) - 1; i >= 0; --i) {
        // Check input dimensions
        TA_ASSERT(lower_bound[i] >= 0ul);
        TA_ASSERT(lower_bound[i] < upper_bound[i]);

        // Compute data for element i of lower, upper, and extent
        const size_type lower_bound_i = lower_bound[i];
        const size_type upper_bound_i = upper_bound[i];
        const size_type extent_i      = upper_bound_i - lower_bound_i;

        lower[i]  = lower_bound_i;
        upper[i]  = upper_bound_i;
        extent[i] = extent_i;
        stride[i] = volume_;
        offset_  += lower_bound_i * volume_;
        volume_  *= extent_i;
      }
    }

    /// Initialize range data from a size array

    /// \tparam Index An array type
    /// \param upper_bound The upper bound of the range
    /// \pre Assume \c rank_ is initialized to the rank of the range and
    /// \c data_ has been allocated to hold 4*rank_ elements
    /// \post \c data_ and \c volume_ are initialized with range dimension
    /// information.
    template <typename Index>
    void init_range_data(const Index& upper_bound) {
      // Construct temp pointers
      size_type* restrict const lower  = data_;
      size_type* restrict const upper  = lower + rank_;
      size_type* restrict const extent = upper + rank_;
      size_type* restrict const stride = extent + rank_;

      // Set the offset and volume initial values
      volume_ = 1ul;
      offset_ = 0ul;

      // Compute range data
      for(int i = int(rank_) - 1; i >= 0; --i) {
        // Check bounds of the input extent
        TA_ASSERT(upper_bound[i] > 0ul);

        // Get extent i
        const size_type extent_i = upper_bound[i];

        lower[i]  = 0ul;
        upper[i]  = extent_i;
        extent[i] = extent_i;
        stride[i] = volume_;
        volume_  *= extent_i;
      }
    }

    /// Initialize permuted range data from lower and upper bounds

    /// \param other_lower_bound The lower bound of the unpermuted range
    /// \param other_upper_bound The upper bound of the unpermuted range
    /// \pre Assume \c rank_ is initialized to the rank of the range and
    /// \c data_ has been allocated to hold 4*rank_ elements
    /// \post \c data_, \c offset_, and \c volume_ are initialized with the
    /// permuted range dimension information from \c other_lower_bound and
    /// \c other_upper_bound.
    void init_range_data(const Permutation& perm,
        const size_type* restrict const other_lower_bound,
        const size_type* restrict const other_upper_bound)
    {
      // Create temporary pointers to this range data
      size_type* restrict const lower  = data_;
      size_type* restrict const upper  = lower + rank_;
      size_type* restrict const extent = upper + rank_;
      size_type* restrict const stride = extent + rank_;

      // Copy the permuted lower, upper, and extent into this range.
      for(unsigned int i = 0u; i < rank_; ++i) {
        const size_type perm_i = perm[i];

        // Get the lower bound, upper bound, and extent from other for rank i.
        const size_type other_lower_bound_i = other_lower_bound[i];
        const size_type other_upper_bound_i = other_upper_bound[i];
        const size_type other_extent_i = other_upper_bound_i - other_lower_bound_i;

        // Store the permuted lower bound, upper bound, and extent
        lower[perm_i]  = other_lower_bound_i;
        upper[perm_i] = other_upper_bound_i;
        extent[perm_i]   = other_extent_i;
      }

      // Recompute stride, offset, and volume
      volume_ = 1ul;
      offset_ = 0ul;
      for(int i = int(rank_) - 1; i >= 0; --i) {
        const size_type lower_i = lower[i];
        stride[i] = volume_;
        offset_ += lower_i * volume_;
        volume_ *= extent[i];
      }
    }

  public:

    /// Default constructor

    /// \post Range has zero rank, volume, and size.
    Range() { }

    /// Construct range defined by an upper and lower bound

    /// \tparam Index An array type
    /// \param lower_bound The lower bound of the N-dimensional range
    /// \param upper_bound The upper bound of the N-dimensional range
    /// \post Range has an lower and upper bound of \c lower_bound and
    /// \c upper_bound.
    /// \throw TiledArray::Exception When the size of \c lower_bound is not
    /// equal to that of \c upper_bound.
    /// \throw TiledArray::Exception When lower_bound[i] >= upper_bound[i]
    /// \throw std::bad_alloc When memory allocation fails.
    template <typename Index,
        typename std::enable_if<! std::is_integral<Index>::value>::type* = nullptr>
    Range(const Index& lower_bound, const Index& upper_bound) {
      const size_type n = detail::size(lower_bound);
      TA_ASSERT(n == detail::size(upper_bound));
      if(n) {
        // Initialize array memory
        data_ = new size_type[n << 2];
        rank_ = n;
        init_range_data(lower_bound, upper_bound);
      }
    }

    /// Range constructor from size array

    /// \tparam Index An array type
    /// \param upper_bound The upper bound of the N-dimensional range
    /// \post Range has an lower bound of 0, and an upper bound of
    /// \c upper_bound.
    /// \throw std::bad_alloc When memory allocation fails.
    template <typename Index,
        typename std::enable_if<! std::is_integral<Index>::value>::type* = nullptr>
    explicit Range(const Index& upper_bound) {
      const size_type n = detail::size(upper_bound);
      if(n) {
        // Initialize array memory
        data_ = new size_type[n << 2];
        rank_ = n;
        init_range_data(upper_bound);
      }
    }

    /// Range constructor from a pack of sizes for each dimension

    /// \tparam Index An array type
    /// \param upper_bound The upper bound of the N-dimensional range
    /// \post Range has an lower bound of 0, and an upper bound of \c (sizes...).
    /// \throw std::bad_alloc When memory allocation fails.
    template<typename... Index,
        typename std::enable_if<detail::is_integral_list<Index...>::value>::type* = nullptr>
    explicit Range(const Index... upper_bound) :
      Range(std::array<size_t, sizeof...(Index)>{{upper_bound...}})
    { }

    /// Copy Constructor

    /// \param other The range to be copied
    /// \throw std::bad_alloc When memory allocation fails.
    Range(const Range_& other) {
      if(other.rank_ > 0ul) {
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
    Range(Range_&& other) :
      data_(other.data_), offset_(other.offset_), volume_(other.volume_),
      rank_(other.rank_)
    {
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

      if(other.rank_ > 0ul) {
        data_ = new size_type[other.rank_ << 2];
        rank_ = other.rank_;

        if(perm) {
          init_range_data(perm, other.data_, other.data_ + rank_);
        } else {
          // Simple copy will due.
          memcpy(data_, other.data_, (sizeof(size_type) << 2) * rank_);
          offset_ = other.offset_;
          volume_ = other.volume_;
        }
      }
    }

    /// Destructor
    ~Range() { delete [] data_; }

    /// Copy assignment operator

    /// \param other The range to be copied
    /// \return A reference to this object
    /// \throw std::bad_alloc When memory allocation fails.
    Range_& operator=(const Range_& other) {
      if(rank_ != other.rank_) {
        delete [] data_;
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

    /// Range lower bound coordinate accessor

    /// \return A pointer to an array that contains the lower bound of this range
    /// \throw nothing
    const size_type* lobound_data() const { return data_; }

    /// Range lower bound accessor

    /// Provided to conform to the Tensor Working Group specification
    /// \return A \c size_array that contains the lower bound of this range
    /// \throw nothing
    size_array lobound() const { return size_array(data_, rank_); }

    /// Range upper bound coordinate accessor

    /// \return A pointer to an array that contains the upper bound of this range
    /// \throw nothing
    const size_type* upbound_data() const { return data_ + rank_; }

    /// Upper bound accessor

    /// \return A \c size_array that contains the upper bound of this range
    /// \throw nothing
    size_array upbound() const {
      return size_array(data_ + rank_, rank_);
    }

    /// Size accessor

    /// \return A pointer to an array that contains the extents of this range
    /// \throw nothing
    const size_type* extent_data() const { return data_ + (rank_ + rank_); }

    /// Size accessor

    /// \return An \c extent_type that contains the extent of each rank
    /// \throw nothing
    extent_type extent() const {
      return size_array(data_ + (rank_ + rank_), rank_);
    }

    /// Stride accessor

    /// \return A pointer to an array that contains the strides of this range
    /// \throw nothing
    size_array stride() const {
      return size_array(data_ + (rank_ + rank_ + rank_), rank_);
    }

    /// Range stride accessor

    /// \return A pointer that contains the strides of each rank
    /// \throw nothing
    const size_type* stride_data() const { return data_ + (rank_ + rank_ + rank_); }


    /// Range volume accessor

    /// \return The total number of elements in the range.
    /// \throw nothing
    size_type volume() const { return volume_; }

    /// alias to volume() to conform to the Tensor Working Group specification
    /// \return The total number of elements in the range.
    /// \throw nothing
    size_type area() const { return volume_; }

    size_type offset() const { return offset_; }

    /// Index iterator factory

    /// The iterator dereferences to an index. The order of iteration matches
    /// the data layout of a dense tensor.
    /// \return An iterator that holds the lower bound index of a tensor
    /// \throw nothing
    const_iterator begin() const { return const_iterator(data_, this); }

    /// Index iterator factory

    /// The iterator dereferences to an index. The order of iteration matches
    /// the data layout of a dense tensor.
    /// \return An iterator that holds the lower bound element index of a tensor
    /// \throw nothing
    const_iterator end() const { return const_iterator(data_ + rank_, this); }



    /// Check the coordinate to make sure it is within the range.

    /// \tparam Index The coordinate index array type
    /// \param index The coordinate index to check for inclusion in the range
    /// \return \c true when <tt>i >= start</tt> and <tt>i < finish</tt>,
    /// otherwise \c false
    /// \throw TildedArray::Exception When the rank of this range is not
    /// equal to the size of the index.
    template <typename Index,
        typename std::enable_if<! std::is_integral<Index>::value, bool>::type* = nullptr>
    bool includes(const Index& index) const {
      TA_ASSERT(detail::size(index) == rank_);
      const size_type* restrict const lower  = data_;
      const size_type* restrict const upper = lower + rank_;

      bool result = (rank_ > 0u);
      for(unsigned int i = 0u; result && (i < rank_); ++i) {
        const size_type index_i = index[i];
        const size_type lower_i = lower[i];
        const size_type upper_i = upper[i];
        result = result && (index_i >= lower_i) && (index_i < upper_i);
      }

      return result;
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
    typename std::enable_if<(sizeof...(Index) > 1ul), size_type>::type
    includes(const Index&... index) const {
      const size_type i[sizeof...(Index)] = {index...};
      return includes(i);
    }



    /// Permute this range

    /// \param perm The permutation to be applied to this range
    /// \return A reference to this range
    /// \throw TildedArray::Exception When the rank of this range is not
    /// equal to the rank of the permutation.
    /// \throw std::bad_alloc When memory allocation fails.
    Range_& operator *=(const Permutation& perm);

    /// Resize range to a new upper and lower bound

    /// \tparam Index An array type
    /// \param lower_bound The lower bounds of the N-dimensional range
    /// \param upper_bound The upper bound of the N-dimensional range
    /// \return A reference to this range
    /// \throw TiledArray::Exception When the size of \c lower_bound is not
    /// equal to that of \c upper_bound.
    /// \throw TiledArray::Exception When <tt>lower_bound[i] >= upper_bound[i]</tt>
    /// \throw std::bad_alloc When memory allocation fails.
    template <typename Index>
    Range_& resize(const Index& lower_bound, const Index& upper_bound) {
      const size_type n = detail::size(lower_bound);
      TA_ASSERT(n == detail::size(upper_bound));

      // Reallocate memory for range arrays
      if(rank_ != n) {
        delete [] data_;
        data_ = (n > 0ul ? new size_type[n << 2] : nullptr);
        rank_ = n;
      }
      if(n > 0ul)
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
      const unsigned int n = detail::size(bound_shift);
      TA_ASSERT(n == rank_);

      const auto* restrict const bound_shift_data = detail::data(bound_shift);
      size_type* restrict const lower = data_;
      size_type* restrict const upper = data_ + rank_;

      for(unsigned i = 0u; i < rank_; ++i) {
        const auto bound_shift_i = bound_shift_data[i];
        lower[i] += bound_shift_i;
        upper[i] += bound_shift_i;
      }

      return *this;
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
    size_type ordinal(const size_type index) const {
      TA_ASSERT(includes(index));
      return index;
    }

    /// calculate the ordinal index of \c index

    /// Convert a coordinate index to an ordinal index.
    /// \tparam Index A coordinate index type (array type)
    /// \param index The index to be converted to an ordinal index
    /// \return The ordinal index of \c index
    /// \throw When \c index is not included in this range.
    template <typename Index,
        typename std::enable_if<! std::is_integral<Index>::value>::type* = nullptr>
    size_type ordinal(const Index& index) const {
      TA_ASSERT(detail::size(index) == rank_);
      TA_ASSERT(includes(index));

      size_type* restrict const stride = data_ + rank_ + rank_ + rank_;

      size_type result = 0ul;
      for(unsigned int i = 0u; i < rank_; ++i) {
        const auto index_i = index[i];
        const size_type stride_i = stride[i];
        result += index_i * stride_i;
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
      const size_type temp_index[sizeof...(Index)] = { index... };
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
      TA_ASSERT(includes(index));

      // Construct result coordinate index object and allocate its memory.
      Range_::index result(rank_, 0);

      // Get pointers to the data
      size_type * restrict const result_data = result.data();
      size_type const * restrict const lower = data_;
      size_type const * restrict const size = data_ + rank_ + rank_;

      // Compute the coordinate index of index in range.
      for(int i = int(rank_) - 1; i >= 0; --i) {
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
    /// a template parameter that can be an index or a size_type.
    /// \param i The index
    /// \return \c i (unchanged)
    template <typename Index,
        typename std::enable_if<! std::is_integral<Index>::value>::type* = nullptr>
    const Index& idx(const Index& i) const {
      TA_ASSERT(includes(i));
      return i;
    }

    template <typename Archive,
        typename std::enable_if<madness::archive::is_input_archive<Archive>::value>::type* = nullptr>
    void serialize(const Archive& ar) {
      // Get rank
      unsigned int rank = 0ul;
      ar & rank;

      // Reallocate the array
      const unsigned int four_x_rank = rank << 2;
      if(rank_ != rank) {
        delete [] data_;
        data_ = (rank > 0u ? new size_type[four_x_rank] : nullptr);
        rank_ = rank;
      }

      // Get range data
      ar & madness::archive::wrap(data_, four_x_rank) & offset_ & volume_;
    }

    template <typename Archive,
        typename std::enable_if<madness::archive::is_output_archive<Archive>::value>::type* = nullptr>
    void serialize(const Archive& ar) const {
      ar & rank_ & madness::archive::wrap(data_, rank_ << 2) & offset_ & volume_;
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
    include_ordinal_(Index i) const { return (i >= Index(0)) && (i < Index(volume_)); }

    /// Check that an unsigned integral value is include in this range

    /// \tparam Index An unsigned integral type
    /// \param i The ordinal index to check
    /// \return \c true when  <tt>i < volume_</tt>, otherwise \c false.
    template <typename Index>
    typename std::enable_if<! std::is_signed<Index>::value, bool>::type
    include_ordinal_(Index i) const { return i < volume_; }

    /// Increment the coordinate index \c i in this range

    /// \param[in,out] i The coordinate index to be incremented
    /// \throw TiledArray::Exception When the rank of i is not equal to
    /// the rank of this range
    /// \throw TiledArray::Exception When \c i or \c i+n is outside this range
    void increment(index& i) const {
      TA_ASSERT(includes(i));

      size_type const * restrict const lower = data_;
      size_type const * restrict const upper = data_ + rank_;

      for(int d = int(rank_) - 1; d >= 0; --d) {
        // increment coordinate
        ++i[d];

        // break if done
        if(i[d] < upper[d])
          return;

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
    /// \throw TiledArray::Exception When \c first or \c last is outside this range
    std::ptrdiff_t distance_to(const index& first, const index& last) const {
      TA_ASSERT(includes(first));
      TA_ASSERT(includes(last));
      return ordinal(last) - ordinal(first);
    }

  }; // class Range

  inline Range& Range::operator *=(const Permutation& perm) {
    TA_ASSERT(perm.dim() == rank_);
    if(rank_ > 1ul) {
      // Copy the lower and upper bound data into a temporary array
      size_type* restrict const temp_lower = new size_type[rank_ << 1];
      const size_type* restrict const temp_upper = temp_lower + rank_;
      std::memcpy(temp_lower, data_, (sizeof(size_type) << 1) * rank_);

      init_range_data(perm, temp_lower, temp_upper);

      // Cleanup old memory.
      delete[] temp_lower;
    }
    return *this;
  }

  /// Exchange the values of the give two ranges.
  inline void swap(Range& r0, Range& r1) { // no throw
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
  inline bool operator ==(const Range& r1, const Range& r2) {
    return (r1.rank() == r2.rank()) && !std::memcmp(r1.lobound_data(), r2.lobound_data(),
        r1.rank() * (2u * sizeof(Range::size_type)));
  }
  /// Range inequality comparison

  /// \param r1 The first range to be compared
  /// \param r2 The second range to be compared
  /// \return \c true when \c r1 does not represent the same range as \c r2,
  /// otherwise \c false.
  inline bool operator !=(const Range& r1, const Range& r2) {
    return (r1.rank() != r2.rank()) || std::memcmp(r1.lobound_data(), r2.lobound_data(),
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

} // namespace TiledArray
#endif // TILEDARRAY_RANGE_H__INCLUDED
