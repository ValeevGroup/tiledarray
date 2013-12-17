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

#include <TiledArray/size_array.h>
#include <TiledArray/range_iterator.h>
#include <TiledArray/coordinates.h>
#include <TiledArray/utility.h>
#include <algorithm>
#include <vector>
#include <functional>

namespace TiledArray {

  // Forward declaration of TiledArray components.
  class Permutation;

  namespace detail {

    template <typename Index, typename WeightArray, typename StartArray>
    inline std::size_t calc_ordinal(const Index& index, const WeightArray& weight, const StartArray& start) {
      // Check that the dimensions of the arrays are equal.
      const std::size_t n = detail::size(index);
      TA_ASSERT(detail::size(weight) == n);
      TA_ASSERT(detail::size(start) == n);

      // Compute ordinal
      typename std::size_t o = 0ul;
      for(std::size_t i = 0ul; i < n; ++i)
        o += (index[i] - start[i]) * weight[i];

      return o;
    }

  }  // namespace detail

  /// Range data of an N-dimensional tensor.
  class Range {
  public:
    typedef Range Range_; ///< This object type
    typedef std::size_t size_type; ///< Size type
    typedef std::vector<std::size_t> index; ///< Coordinate index type
    typedef detail::SizeArray<std::size_t> size_array; ///< Size array type
    typedef detail::RangeIterator<index, Range_> const_iterator; ///< Coordinate iterator
    friend class detail::RangeIterator<index, Range_>;

  private:

    /// Initialize range arrays

    /// Use \c buffer to set the buffers for range arrays.
    /// \param buffer The buffer that will holds \c 4*n elements
    /// \param n The size of each of the range arrays
    /// \note If the range arrays reference a valid buffer, then calling this
    /// function will cause a memory leak.
    void init_arrays(size_type* const buffer, const size_type n) {
      start_.set(buffer, n);
      finish_.set(start_.end(), n);
      size_.set(finish_.end(), n);
      weight_.set(size_.end(), n);
    }

    /// Allocate and initialize range arrays

    /// \param n The size of each of the range arrays
    /// \throw std::bad_alloc When memory allocation fails
    void alloc_arrays(const size_type n) { init_arrays(new size_type[n * 4ul], n); }

    /// Reallocate and reinitialize range arrays

    /// If <tt>dim() != n</tt>, then a new buffer is allocated and it is used to
    /// reinitialize the range arrays. If \c n is zero, the range arrays are set
    /// to zero sized arrays.
    /// deallocated and the range arrays are set to zero size arrays.
    /// \param n The new size for the range arrays
    void realloc_arrays(const size_type n) {
      if(dim() != n) {
        delete_arrays();
        size_type* const buffer = (n > 0ul ? new size_type[n * 4ul] : NULL);
        init_arrays(buffer, n);
      }
    }
    /// delete array memory

    /// \throw nothing
    void delete_arrays() { delete [] start_.data(); }

    template <typename Index>
    void compute_range_data(const size_type n, const Index& start, const Index& finish) {
      // Set the volume seed
      volume_ = 1ul;

      // Compute range data
      for(int i = n - 1; i >= 0; --i) {
        TA_ASSERT(start[i] < finish[i]);
        start_[i] = start[i];
        finish_[i] = finish[i];
        size_[i] = finish[i] - start[i];
        weight_[i] = volume_;
        volume_ *= size_[i];
      }
    }

    template <typename Index>
    void compute_range_data(const size_type n, const Index& size) {
      // Set the volume seed
      volume_ = 1ul;

      // Compute range data
      for(int i = n - 1; i >= 0; --i) {
        TA_ASSERT(size[i] > 0ul);
        start_[i] = 0ul;
        finish_[i] = size_[i] = size[i];
        weight_[i] = volume_;
        volume_ *= size[i];
      }
    }

  public:

    /// Default constructor

    /// Construct a range with size and dimensions equal to zero.
    Range() :
      start_(), finish_(), size_(), weight_(), volume_(0ul)
    { }

    /// Constructor defined by an upper and lower bound

    /// \tparam Index An array type
    /// \param start The lower bounds of the N-dimensional range
    /// \param finish The upper bound of the N-dimensional range
    /// \throw TiledArray::Exception When the size of \c start is not equal to
    /// that of \c finish.
    /// \throw TiledArray::Exception When start[i] >= finish[i]
    /// \throw std::bad_alloc When memory allocation fails.
    template <typename Index>
    Range(const Index& start, const Index& finish) :
      start_(), finish_(), size_(), weight_(), volume_(0ul)
    {
      const size_type n = detail::size(start);
      TA_ASSERT(n == detail::size(finish));
      if(n > 0ul) {
        // Initialize array memory
        alloc_arrays(n);
        compute_range_data(n, start, finish);
      }
    }

    /// Range constructor from size array

    /// \tparam SizeArray An array type
    /// \param size An array with the size of each dimension
    /// \throw std::bad_alloc When memory allocation fails.
    template <typename SizeArray>
    explicit Range(const SizeArray& size) :
      start_(), finish_(), size_(), weight_(), volume_(0ul)
    {
      const size_type n = detail::size(size);
      if(n) {
        // Initialize array memory
        alloc_arrays(n);
        compute_range_data(n, size);
      }
    }

    /// Copy Constructor

    /// \param other The range to be copied
    /// \throw std::bad_alloc When memory allocation fails.
    Range(const Range_& other) :
      start_(), finish_(), size_(), weight_(), volume_(other.volume_)
    {
      const size_type n = other.dim();
      if(n > 0ul) {
        alloc_arrays(n);
        memcpy(start_.data(), other.start_.begin(), sizeof(size_type) * 4ul * n);
      }
    }

    /// Destructor
    ~Range() { delete_arrays(); }

    /// Copy assignment operator

    /// \param other The range to be copied
    /// \return A reference to this object
    /// \throw std::bad_alloc When memory allocation fails.
    Range_& operator=(const Range_& other) {
      const size_type n = other.dim();
      realloc_arrays(n);
      memcpy(start_.data(), other.start_.begin(), sizeof(size_type) * 4ul * n);
      volume_ = other.volume();

      return *this;
    }

    /// Dimension accessor

    /// \return The number of dimension of this range
    /// \throw nothing
    unsigned int dim() const { return size_.size(); }

    /// Range start coordinate accessor

    /// \return A \c size_array that contains the lower bound of this range
    /// \throw nothing
    const size_array& start() const { return start_; }

    /// Range finish coordinate accessor

    /// \return A \c size_array that contains the upper bound of this range
    /// \throw nothing
    const size_array& finish() const { return finish_; }

    /// Range size accessor

    /// \return A \c size_array that contains the sizes of each dimension
    /// \throw nothing
    const size_array& size() const { return size_; }

    /// Range weight accessor

    /// \return A \c size_array that contains the strides of each dimension
    /// \throw nothing
    const size_array& weight() const { return weight_; }


    /// Range volume accessor

    /// \return The total number of elements in the range.
    /// \throw nothing
    size_type volume() const { return volume_; }

    /// Index iterator factory

    /// The iterator dereferences to an index. The order of iteration matches
    /// the data layout of a dense tensor.
    /// \return An iterator that holds the start element index of a tensor
    /// \throw nothing
    const_iterator begin() const { return const_iterator(start_, this); }

    /// Index iterator factory

    /// The iterator dereferences to an index. The order of iteration matches
    /// the data layout of a dense tensor.
    /// \return An iterator that holds the finish element index of a tensor
    /// \throw nothing
    const_iterator end() const { return const_iterator(finish_, this); }

    /// Check the coordinate to make sure it is within the range.

    /// \tparam Index The coordinate index array type
    /// \param index The coordinate index to check for inclusion in the range
    /// \return \c true when \c i \c >= \c start and \c i \c < \c f, otherwise
    /// \c false
    /// \throw TildedArray::Exception When the dimension of this range is not
    /// equal to the size of the index.
    template <typename Index>
    typename madness::disable_if<std::is_integral<Index>, bool>::type
    includes(const Index& index) const {
      TA_ASSERT(detail::size(index) == dim());
      const unsigned int end = dim();
      for(unsigned int i = 0ul; i < end; ++i)
        if((index[i] < start_[i]) || (index[i] >= finish_[i]))
          return false;

      return true;
    }

    /// Check the ordinal index to make sure it is within the range.

    /// \param i The ordinal index to check for inclusion in the range
    /// \return \c true when \c i \c >= \c 0 and \c i \c < \c volume
    /// \throw nothing
    template <typename Ordinal>
    typename madness::enable_if<std::is_integral<Ordinal>, bool>::type
    includes(Ordinal i) const {
      return include_ordinal_(i);
    }

    /// Permute this range

    /// \param perm The permutation to be applied to this range
    /// \return A reference to this range
    /// \throw TildedArray::Exception When the dimension of this range is not
    /// equal to the dimension of the permutation.
    /// \throw std::bad_alloc When memory allocation fails.
    Range_& operator ^=(const Permutation& perm) {
      const size_type n = dim();
      TA_ASSERT(perm.dim() == n);

      if(n > 1ul) {
        // Create a permuted copy of start and finish
        madness::ScopedArray<size_type> buffer(new size_type[n * 2ul]);
        register size_type* const perm_start = buffer.get();
        register size_type* const perm_finish = buffer.get() + n;
        for(size_type i = 0ul; i < n; ++i) {
          perm_start[perm[i]] = start_[i];
          perm_finish[perm[i]] = finish_[i];
        }

        compute_range_data(n, perm_start, perm_finish);
      }

      return *this;
    }

    /// Resize range to a new upper and lower bound

    /// \tparam Index An array type
    /// \param start The lower bounds of the N-dimensional range
    /// \param finish The upper bound of the N-dimensional range
    /// \throw TiledArray::Exception When the size of \c start is not equal to
    /// that of \c finish.
    /// \throw TiledArray::Exception When start[i] >= finish[i]
    /// \throw std::bad_alloc When memory allocation fails.
    template <typename Index>
    Range_& resize(const Index& start, const Index& finish) {
      const size_type n = detail::size(start);
      TA_ASSERT(n == detail::size(finish));

      // Reallocate memory for range arrays
      realloc_arrays(n);
      if(n > 0ul)
        compute_range_data(n, start, finish);
      else
        volume_ = 0ul;

      return *this;
    }

    /// calculate the ordinal index of \c i

    /// This function is just a pass-through so the user can call \c ord() on
    /// a template parameter that can be a coordinate index or an integral.
    /// \param index Ordinal index
    /// \return \c index (unchanged)
    /// \throw When \c index is not included in this range
    size_type ord(const size_type index) const {
      TA_ASSERT(includes(index));
      return index;
    }

    /// calculate the ordinal index of \c i

    /// Convert a coordinate index to an ordinal index.
    /// \tparam Index A coordinate index type (array type)
    /// \param index The index to be converted to an ordinal index
    /// \return The ordinal index of \c index
    /// \throw When \c index is not included in this range.
    template <typename Index>
    typename madness::disable_if<std::is_integral<Index>, size_type>::type
    ord(const Index& index) const {
      TA_ASSERT(detail::size(index) == dim());
      TA_ASSERT(includes(index));
      size_type o = 0;
      const unsigned int end = dim();
      for(unsigned int i = 0ul; i < end; ++i)
        o += (index[i] - start_[i]) * weight_[i];

      return o;
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
      Range_::index result;
      result.reserve(dim());

      // Compute the coordinate index of o in range.
      for(std::size_t i = 0ul; i < dim(); ++i) {
        const size_type s = index / weight_[i]; // Compute the size of result[i]
        index %= weight_[i];
        result.push_back(s + start_[i]);
      }

      return result;
    }

    /// calculate the index of \c i

    /// This function is just a pass-through so the user can call \c idx() on
    /// a template parameter that can be an index or a size_type.
    /// \param i The index
    /// \return \c i (unchanged)
    template <typename Index>
    typename madness::disable_if<std::is_integral<Index>, const index&>::type
    idx(const Index& i) const {
      TA_ASSERT(includes(i));
      return i;
    }

    template <typename Archive>
    typename madness::enable_if<madness::archive::is_input_archive<Archive> >::type
    serialize(const Archive& ar) {
      // Get number of dimensions
      size_type n = 0ul;
      ar & n;

      // Get range data
      realloc_arrays(n);
      ar & madness::archive::wrap(start_.data(), n * 4ul) & volume_;
    }

    template <typename Archive>
    typename madness::enable_if<madness::archive::is_output_archive<Archive> >::type
    serialize(const Archive& ar) const {
      const size_type n = dim();
      ar & n & madness::archive::wrap(start_.data(), n * 4ul) & volume_;
    }

    void swap(Range_& other) {
      // Get temp data
      size_type* temp_start = start_.data();
      const size_type n = start_.size();

      // Swap data
      init_arrays(other.start_.data(), n);
      other.init_arrays(temp_start, n);
      std::swap(volume_, other.volume_);
    }

  private:

    /// Check that a signed integral value is include in this range

    /// \tparam Index A signed integral type
    /// \param i The ordinal index to check
    /// \return \c true when <tt>i >= 0</tt> and <tt>i < volume_</tt>, otherwise
    /// \c false.
    template <typename Index>
    typename madness::enable_if<std::is_signed<Index>, bool>::type
    include_ordinal_(Index i) const { return (i >= Index(0)) && (i < Index(volume_)); }

    /// Check that an unsigned integral value is include in this range

    /// \tparam Index An unsigned integral type
    /// \param i The ordinal index to check
    /// \return \c true when  <tt>i < volume_</tt>, otherwise \c false.
    template <typename Index>
    typename madness::disable_if<std::is_signed<Index>, bool>::type
    include_ordinal_(Index i) const { return i < volume_; }

    /// Increment the coordinate index \c i in this range

    /// \param[in,out] i The coordinate index to be incremented
    /// \throw TiledArray::Exception When the dimension of i is not equal to
    /// that of this range
    /// \throw TiledArray::Exception When \c i or \c i+n is outside this range
    void increment(index& i) const {
      TA_ASSERT(includes(i));
      for(int d = int(dim()) - 1; d >= 0; --d) {
        // increment coordinate
        ++i[d];

        // break if done
        if(i[d] < finish_[d])
          return;

        // Reset current index to start value.
        i[d] = start_[d];
      }

      // if the current location was set to start then it was at the end and
      // needs to be reset to equal finish.
      std::copy(finish_.begin(), finish_.end(), i.begin());
    }

    /// Advance the coordinate index \c i by \c n in this range

    /// \param[in,out] i The coordinate index to be advanced
    /// \param n The distance to advance \c i
    /// \throw TiledArray::Exception When the dimension of i is not equal to
    /// that of this range
    /// \throw TiledArray::Exception When \c i or \c i+n is outside this range
    void advance(index& i, std::ptrdiff_t n) const {
      TA_ASSERT(includes(i));
      const size_type o = ord(i) + n;
      TA_ASSERT(includes(o));
      i = idx(o);
    }

    /// Compute the distance between the coordinate indices \c first and \c last

    /// \param first The starting position in the range
    /// \param last The ending position in the range
    /// \return The difference between first and last, in terms of range positions
    /// \throw TiledArray::Exception When the dimension of \c first or \c last
    /// is not equal to that of this range
    /// \throw TiledArray::Exception When \c first or \c last is outside this range
    std::ptrdiff_t distance_to(const index& first, const index& last) const {
      TA_ASSERT(includes(first));
      TA_ASSERT(includes(last));
      return ord(last) - ord(first);
    }

    size_array start_; ///< Tile origin
    size_array finish_; ///< Tile upper bound
    size_array size_; ///< Dimension sizes
    size_array weight_; ///< Dimension weights (strides)
    size_type volume_; ///< Total number of elements
  }; // class Range



  /// Exchange the values of the give two ranges.
  inline void swap(Range& r0, Range& r1) { // no throw
    r0.swap(r1);
  }


  /// Create a permuted range

  /// \param perm The permutation to be applied to the range
  /// \param r The range to be permuted
  /// \return A permuted copy of \c r.
  inline Range operator ^(const Permutation& perm, const Range& r) {
    const Range::size_type n = r.dim();
    TA_ASSERT(perm.dim() == n);
    Range result;

    if(n > 1ul) {
      // Get the start and finish of the original range.
      const Range::size_array& start = r.start();
      const Range::size_array& finish = r.finish();

      // Create a permuted copy of start and finish
      madness::ScopedArray<Range::size_type> buffer(new Range::size_type[n * 2ul]);
      register Range::size_type* const perm_start = buffer.get();
      register Range::size_type* const perm_finish = buffer.get() + n;
      for(Range::size_type i = 0ul; i < n; ++i) {
        perm_start[perm[i]] = start[i];
        perm_finish[perm[i]] = finish[i];
      }

      result.resize(Range::size_array(perm_start, perm_start + n),
          Range::size_array(perm_finish, perm_finish + n));
    } else {
      result = r;
    }

    return result;
  }

  /// No permutation function

  /// This function is used to allow generic code for \c Permutation or
  /// \c NoPermutation code.
  /// \param r The range not to be permuted
  /// \return A const reference to \c r
  inline const Range& operator ^(const detail::NoPermutation&, const Range& r) {
    return r;
  }

  /// Range equality comparison

  /// \param r1 The first range to be compared
  /// \param r2 The second range to be compared
  /// \return \c true when \c r1 represents the same range as \c r2, otherwise
  /// \c false.
  inline bool operator ==(const Range& r1, const Range& r2) {
    return ((r1.start() == r2.start()) && (r1.finish() == r2.finish()));
  }
  /// Range inequality comparison

  /// \param r1 The first range to be compared
  /// \param r2 The second range to be compared
  /// \return \c true when \c r1 does not represent the same range as \c r2,
  /// otherwise \c false.
  inline bool operator !=(const Range& r1, const Range& r2) {
    return ! operator ==(r1, r2);
  }

  /// Range output operator

  /// \param os The output stream that will be used to print \c r
  /// \param r The range to be printed
  /// \return A reference to the output stream
  inline std::ostream& operator<<(std::ostream& os, const Range& r) {
    os << "[ ";
    detail::print_array(os, r.start());
    os << ", ";
    detail::print_array(os, r.finish());
    os << " )";
    return os;
  }

} // namespace TiledArray
#endif // TILEDARRAY_RANGE_H__INCLUDED
