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

namespace TiledArray {

  /// Range data of an N-dimensional tensor.
  class Range {
  public:
    typedef Range Range_; ///< This object type
    typedef std::size_t size_type; ///< Size type
    typedef std::vector<size_type> index; ///< Coordinate index type
    typedef index index_type; ///< Coordinate index type, to conform Tensor Working Group spec
    typedef std::vector<size_type> size_array; ///< Size array type
    typedef index extent_type;    ///< Range extent type, to conform Tensor Working Group spec
    typedef std::size_t ordinal_type; ///< Ordinal type, to conform Tensor Working Group spec
    typedef detail::RangeIterator<size_type, Range_> const_iterator; ///< Coordinate iterator
    friend class detail::RangeIterator<size_type, Range_>;

  private:

    size_type* data_; ///< An array that holds the dimension information of the
                      ///< range. The layout of the array is:
                      ///< \code
                      ///< { start[0],  ..., start[rank_ - 1],
                      ///<   finish[0], ..., finish[rank_ - 1],
                      ///<   size[0],   ..., size[rank_ - 1],
                      ///<   stride[0], ..., stride[rank_ - 1] }
                      ///< \endcode
    size_type volume_; ///< Total number of elements
    unsigned int rank_; ///< The rank (or number of dimensions) in the range

    struct Enabler {};

    /// Initialize range data from lower and upper bounds

    /// \tparam Index An array type
    /// \param start The lower bound of the range
    /// \param finish The upper bound of the range
    /// \pre Assume \c rank_ is initialized to the rank of the range and
    /// \c data_ has been allocated to hold 4*rank_ elements
    /// \post \c data_ and \c volume_ are initialized with range dimension
    /// information.
    template <typename Index>
    void init_range_data(const Index& start, const Index& finish) {
      // Construct temp pointers
      size_type* restrict const start_ptr  = data_;
      size_type* restrict const finish_ptr = start_ptr + rank_;
      size_type* restrict const size_ptr   = finish_ptr + rank_;
      size_type* restrict const weight_ptr = size_ptr + rank_;

      // Set the volume seed
      volume_ = 1ul;

      // Compute range data
      for(int i = int(rank_) - 1; i >= 0; --i) {
        // Check input dimensions
        TA_ASSERT(start[i] >= 0ul);
        TA_ASSERT(start[i] < finish[i]);

        // Compute data for element i of start, finish, and size
        const size_type start_i  = start[i];
        const size_type finish_i = finish[i];
        const size_type size_i   = finish_i - start_i;

        start_ptr[i]  = start_i;
        finish_ptr[i] = finish_i;
        size_ptr[i]   = size_i;
        weight_ptr[i] = volume_;
        volume_      *= size_i;
      }
    }

    /// Initialize range data from a size array

    /// \tparam Index An array type
    /// \param size The upper bound of the range
    /// \pre Assume \c rank_ is initialized to the rank of the range and
    /// \c data_ has been allocated to hold 4*rank_ elements
    /// \post \c data_ and \c volume_ are initialized with range dimension
    /// information.
    template <typename Index>
    void init_range_data(const Index& size) {
      // Construct temp pointers
      size_type* restrict const start_ptr  = data_;
      size_type* restrict const finish_ptr = start_ptr + rank_;
      size_type* restrict const size_ptr   = finish_ptr + rank_;
      size_type* restrict const weight_ptr = size_ptr + rank_;

      // Set the volume seed
      volume_ = 1ul;

      // Compute range data
      for(int i = int(rank_) - 1; i >= 0; --i) {
        // Check bounds of the input size
        TA_ASSERT(size[i] > 0ul);

        // Get size i
        const size_type size_i = size[i];

        start_ptr[i]  = 0ul;
        finish_ptr[i] = size_i;
        size_ptr[i]   = size_i;
        weight_ptr[i] = volume_;
        volume_      *= size_i;
      }
    }

    /// Initialize permuted range data from lower and upper bounds

    /// \param other_start The lower bound of the unpermuted range
    /// \param other_finish The upper bound of the unpermuted range
    /// \pre Assume \c rank_ is initialized to the rank of the range and
    /// \c data_ has been allocated to hold 4*rank_ elements
    /// \post \c data_ and \c volume_ are initialized with the permuted range
    /// dimension information from \c other_start and \c other_finish.
    void init_range_data(const Permutation& perm,
        const size_type* restrict const other_start,
        const size_type* restrict const other_finish)
    {
      // Create temporary pointers to this range data
      size_type* restrict const start_ptr  = data_;
      size_type* restrict const finish_ptr = start_ptr + rank_;
      size_type* restrict const size_ptr   = finish_ptr + rank_;
      size_type* restrict const weight_ptr = size_ptr + rank_;

      // Copy the permuted start, finish, and size into this range.
      for(unsigned int i = 0u; i < rank_; ++i) {
        const size_type perm_i = perm[i];

        // Get the start, finish, and size from other for rank i.
        const size_type other_start_i = other_start[i];
        const size_type other_finish_i = other_finish[i];
        const size_type other_size_i = other_finish_i - other_start_i;

        // Store the permuted start, finish and size
        start_ptr[perm_i]  = other_start_i;
        finish_ptr[perm_i] = other_finish_i;
        size_ptr[perm_i]   = other_size_i;
      }

      // Recompute weight and volume
      volume_ = 1ul;
      for(int i = int(rank_) - 1; i >= 0; --i) {
        weight_ptr[i] = volume_;
        volume_ *= size_ptr[i];
      }
    }

  public:

    /// Default constructor

    /// \post Range has zero rank, volume, and size.
    Range() : data_(nullptr), volume_(0ul), rank_(0u) { }

    /// Construct range defined by an upper and lower bound

    /// \tparam Index An array type
    /// \param start The lower bound of the N-dimensional range
    /// \param finish The upper bound of the N-dimensional range
    /// \post Range has an lower and upper bound of \c start and \c finish.
    /// \throw TiledArray::Exception When the size of \c start is not equal to
    /// that of \c finish.
    /// \throw TiledArray::Exception When start[i] >= finish[i]
    /// \throw std::bad_alloc When memory allocation fails.
    template <typename Index,
        enable_if_t<! std::is_integral<Index>::value>* = nullptr>
    Range(const Index& start, const Index& finish) :
      data_(nullptr), volume_(0ul), rank_(0u)
    {
      const size_type n = detail::size(start);
      TA_ASSERT(n == detail::size(finish));
      if(n) {
        // Initialize array memory
        data_ = new size_type[n << 2];
        rank_ = n;
        init_range_data(start, finish);
      }
    }

    /// Range constructor from size array

    /// \tparam SizeArray An array type
    /// \param size An array with the size of each dimension
    /// \post Range has an lower bound of 0, and an upper bound of \c size.
    /// \throw std::bad_alloc When memory allocation fails.
    template <typename SizeArray,
        enable_if_t<! std::is_integral<SizeArray>::value>* = nullptr>
    explicit Range(const SizeArray& size) :
      data_(nullptr), volume_(0ul), rank_(0u)
    {
      const size_type n = detail::size(size);
      if(n) {
        // Initialize array memory
        data_ = new size_type[n << 2];
        rank_ = n;
        init_range_data(size);
      }
    }

    /// Range constructor from a pack of sizes for each dimension

    /// \tparam Sizes A pack of unsigned integers
    /// \param sizes A pack of sizes for dimensions
    /// \post Range has an lower bound of 0, and an upper bound of \c (sizes...).
    /// \throw std::bad_alloc When memory allocation fails.
    template<typename... Sizes,
        enable_if_t<detail::is_integral_list<Sizes...>::value>* = nullptr>
    explicit Range(const Sizes... sizes) :
      data_(nullptr), volume_(0ul), rank_(0u)
    {
      constexpr size_type n = sizeof...(Sizes);
      size_type s[n] = {sizes...};

      // Initialize array memory
      data_ = new size_type[n << 2];
      rank_ = n;
      init_range_data(s);
    }

    /// Copy Constructor

    /// \param other The range to be copied
    /// \throw std::bad_alloc When memory allocation fails.
    Range(const Range_& other) :
      data_(nullptr), volume_(0ul), rank_(0u)
    {
      if(other.rank_ > 0ul) {
        data_ = new size_type[other.rank_ << 2];
        rank_ = other.rank_;
        volume_ = other.volume_;
        memcpy(data_, other.data_, (sizeof(size_type) << 2) * other.rank_);
      }
    }

    /// Copy Constructor

    /// \param other The range to be copied
    /// \throw std::bad_alloc When memory allocation fails.
    Range(Range_&& other) :
      data_(other.data_), volume_(other.volume_), rank_(other.rank_)
    {
      other.data_ = nullptr;
      other.volume_ = 0ul;
      other.rank_ = 0u;
    }

    /// Permuting copy constructor

    /// \param perm The permutation applied to other
    /// \param other The range to be permuted and copied
    /// \throw std::bad_alloc When memory allocation fails.
    Range(const Permutation& perm, const Range_& other) :
      data_(nullptr), volume_(0ul), rank_(0u)
    {
      TA_ASSERT(perm.dim() == other.rank_);

      if(other.rank_ > 0ul) {
        data_ = new size_type[other.rank_ << 2];
        rank_ = other.rank_;

        if(perm) {
          init_range_data(perm, other.data_, other.data_ + rank_);
        } else {
          // Simple copy will due.
          memcpy(data_, other.data_, (sizeof(size_type) << 2) * rank_);
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
      volume_ = other.volume();

      return *this;
    }

    /// Move assignment operator

    /// \param other The range to be copied
    /// \return A reference to this object
    /// \throw nothing
    Range_& operator=(Range_&& other) {
      data_ = other.data_;
      volume_ = other.volume_;
      rank_ = other.rank_;

      other.data_ = nullptr;
      other.volume_ = 0ul;
      other.rank_ = 0u;

      return *this;
    }

    /// Dimension accessor

    /// \return The rank (number of dimensions) of this range
    /// \throw nothing
    /// \note Equivalent to \c rank()
    unsigned int dim() const { return rank_; }

    /// Rank accessor

    /// \return The rank (number of dimensions) of this range
    /// \throw nothing
    /// \note Provided to satisfy the requirements of Tensor Working Group
    /// specification.
    unsigned int rank() const { return rank_; }

    /// Range start coordinate accessor

    /// \return A pointer to an array that contains the lower bound of this range
    /// \throw nothing
    const size_type* start() const { return data_; }

    /// Range lower bound accessor

    /// Provided to conform to the Tensor Working Group specification
    /// \return A \c size_array that contains the lower bound of this range
    /// \throw nothing
    /// \note Provided to satisfy the requirements of Tensor Working Group
    /// specification.
    size_array lobound() const { return size_array(data_, data_ + rank_); }

    /// Range finish coordinate accessor

    /// \return A pointer to an array that contains the upper bound of this range
    /// \throw nothing
    const size_type* finish() const { return data_ + rank_; }

    /// Upper bound accessor

    /// \return A \c size_array that contains the upper bound of this range
    /// \throw nothing
    /// \note Provided to satisfy the requirements of Tensor Working Group
    /// specification.
    size_array upbound() const {
      const size_type* const finish = data_ + rank_;
      return size_array(finish, finish + rank_);
    }

    /// Size accessor

    /// \return A pointer to an array that contains the lower bound of this range
    /// \throw nothing
    const size_type* size() const { return data_ + (rank_ + rank_); }

    /// Size accessor

    /// \return An \c extent_type that contains the extent of each dimension
    /// \throw nothing
    /// \note Provided to satisfy the requirements of Tensor Working Group
    /// specification.
    extent_type extent() const {
      const size_type* const size = data_ + rank_ + rank_;
      return size_array(size, size + rank_);
    }

    /// Range weight accessor

    /// \return A \c size_array that contains the strides of each dimension
    /// \throw nothing
    const size_type* weight() const { return data_ + (rank_ + rank_ + rank_); }


    /// Range volume accessor

    /// \return The total number of elements in the range.
    /// \throw nothing
    size_type volume() const { return volume_; }

    /// alias to volume() to conform to the Tensor Working Group specification
    /// \return The total number of elements in the range.
    /// \throw nothing
    size_type area() const { return volume_; }

    /// Index iterator factory

    /// The iterator dereferences to an index. The order of iteration matches
    /// the data layout of a dense tensor.
    /// \return An iterator that holds the start element index of a tensor
    /// \throw nothing
    const_iterator begin() const { return const_iterator(data_, this); }

    /// Index iterator factory

    /// The iterator dereferences to an index. The order of iteration matches
    /// the data layout of a dense tensor.
    /// \return An iterator that holds the finish element index of a tensor
    /// \throw nothing
    const_iterator end() const { return const_iterator(data_ + rank_, this); }

    /// Check the coordinate to make sure it is within the range.

    /// \tparam Index The coordinate index array type
    /// \param index The coordinate index to check for inclusion in the range
    /// \return \c true when \c i \c >= \c start and \c i \c < \c f, otherwise
    /// \c false
    /// \throw TildedArray::Exception When the dimension of this range is not
    /// equal to the size of the index.
    template <typename Index,
        enable_if_t<! std::is_integral<Index>::value, bool>* = nullptr>
    bool includes(const Index& index) const {
      TA_ASSERT(detail::size(index) == rank_);
      size_type* restrict const start_ptr  = data_;
      size_type* restrict const finish_ptr = start_ptr + rank_;

      bool result = (rank_ > 0u);
      for(unsigned int i = 0u; result && (i < rank_); ++i) {
        const size_type index_i = index[i];
        result = result && (index_i >= start_ptr[i]) && (index_i < finish_ptr[i]);
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
      size_type i[sizeof...(Index)] = {index...};
      return includes(i);
    }



    /// Permute this range

    /// \param perm The permutation to be applied to this range
    /// \return A reference to this range
    /// \throw TildedArray::Exception When the dimension of this range is not
    /// equal to the dimension of the permutation.
    /// \throw std::bad_alloc When memory allocation fails.
    Range_& operator *=(const Permutation& perm) {
      TA_ASSERT(perm.dim() == rank_);

      if(rank_ > 1ul) {
        // Copy the start and finish data into a temporary array
        size_type* restrict const start = new size_type[rank_ << 1];
        const size_type* restrict const finish = start + rank_;
        std::memcpy(start, data_, (sizeof(size_type) << 1) * rank_);

        init_range_data(perm, start, finish);

        // Cleanup old memory.
        delete [] start;
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
      if(rank_ != n) {
        delete [] data_;
        data_ = (n > 0ul ? new size_type[n << 2] : nullptr);
        rank_ = n;
      }
      if(n > 0ul)
        init_range_data(start, finish);
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
    template <typename Index,
        enable_if_t<! std::is_integral<Index>::value>* = nullptr>
    size_type ord(const Index& index) const {
      TA_ASSERT(detail::size(index) == rank_);
      TA_ASSERT(includes(index));

      size_type* restrict const start_ptr  = data_;
      size_type* restrict const weight_ptr = data_ + rank_ + rank_ + rank_;

      size_type result = 0ul;
      for(unsigned int i = 0u; i < rank_; ++i)
        result += (index[i] - start_ptr[i]) * weight_ptr[i];

      return result;
    }

    template <typename... Index,
        enable_if_t<(sizeof...(Index) > 1ul)>* = nullptr>
    size_type ord(const Index&... index) const {
      size_type i[sizeof...(Index)] = { index... };
      return ord(i);
    }

    /// alias to ord<Index>(), to conform with the Tensor Working Group spec \sa ord()
    template <typename... Index>
    size_type ordinal(const Index&... index) const { return ord(index...); }

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
      size_type * restrict const result_data = & result.front();
      size_type const * restrict const weight_ptr = data_ + rank_ + rank_ + rank_;
      size_type const * restrict const start_ptr = data_;

      // Compute the coordinate index of o in range.
      for(unsigned int i = 0u; i < rank_; ++i) {
        const size_type weight_i = weight_ptr[i];
        const size_type start_i = start_ptr[i];

        // Compute result index element i
        const size_type result_i = (index / weight_i) + start_i;
        index %= weight_i;

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
        enable_if_t<! std::is_integral<Index>::value>* = nullptr>
    const Index& idx(const Index& i) const {
      TA_ASSERT(includes(i));
      return i;
    }

    template <typename Archive,
        enable_if_t<madness::archive::is_input_archive<Archive>::value>* = nullptr>
    void serialize(const Archive& ar) {
      // Get number of dimensions
      unsigned int n = 0ul;
      ar & n;

      // Reallocate the array
      const unsigned int four_n = n << 2;
      if(rank_ != n) {
        delete [] data_;
        data_ = (n > 0u ? new size_type[four_n] : nullptr);
        rank_ = n;
      }

      // Get range data
      ar & madness::archive::wrap(data_, four_n) & volume_;
    }

    template <typename Archive,
        enable_if_t<madness::archive::is_output_archive<Archive>::value>* = nullptr>
    void serialize(const Archive& ar) const {
      ar & rank_ & madness::archive::wrap(data_, rank_ << 2) & volume_;
    }

    void swap(Range_& other) {
      // Get temp data
      std::swap(data_, other.data_);
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
    /// \throw TiledArray::Exception When the dimension of i is not equal to
    /// that of this range
    /// \throw TiledArray::Exception When \c i or \c i+n is outside this range
    void increment(index& i) const {
      TA_ASSERT(includes(i));

      size_type const * restrict const start_ptr = data_;
      size_type const * restrict const finish_ptr = data_ + rank_;

      for(int d = int(rank_) - 1; d >= 0; --d) {
        // increment coordinate
        ++i[d];

        // break if done
        if(i[d] < finish_ptr[d])
          return;

        // Reset current index to start value.
        i[d] = start_ptr[d];
      }

      // if the current location was set to start then it was at the end and
      // needs to be reset to equal finish.
      std::copy(finish_ptr, finish_ptr + rank_, i.begin());
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

  }; // class Range



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
    return (r1.rank() == r2.rank()) && !std::memcmp(r1.start(), r2.start(),
        r1.rank() * (2u * sizeof(Range::size_type)));
  }
  /// Range inequality comparison

  /// \param r1 The first range to be compared
  /// \param r2 The second range to be compared
  /// \return \c true when \c r1 does not represent the same range as \c r2,
  /// otherwise \c false.
  inline bool operator !=(const Range& r1, const Range& r2) {
    return (r1.rank() != r2.rank()) || std::memcmp(r1.start(), r2.start(),
        r1.rank() * (2u * sizeof(Range::size_type)));
  }

  /// Range output operator

  /// \param os The output stream that will be used to print \c r
  /// \param r The range to be printed
  /// \return A reference to the output stream
  inline std::ostream& operator<<(std::ostream& os, const Range& r) {
    os << "[ ";
    detail::print_array(os, r.start(), r.rank());
    os << ", ";
    detail::print_array(os, r.finish(), r.rank());
    os << " )";
    return os;
  }

} // namespace TiledArray
#endif // TILEDARRAY_RANGE_H__INCLUDED
