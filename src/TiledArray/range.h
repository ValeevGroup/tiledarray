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
#include <algorithm>
#include <vector>
#include <functional>

namespace TiledArray {

  // Forward declaration of TiledArray components.
  class Permutation;

  namespace detail {

    template <typename SizeArray>
    inline std::size_t calc_volume(const SizeArray& size) {
      return size.size() ? std::accumulate(size.begin(), size.end(), typename SizeArray::value_type(1),
          std::multiplies<typename SizeArray::value_type>()) : 0ul;
    }

    template<typename InIter, typename OutIter>
    inline void calc_weight_helper(InIter first, InIter last, OutIter result) { // no throw
      typedef typename std::iterator_traits<OutIter>::value_type value_type;

      for(value_type weight = 1; first != last; ++first, ++result) {
        *result = (*first != 0ul ? weight : 0);
        weight *= *first;
      }
    }

    template <typename WArray, typename SArray>
    inline void calc_weight(WArray& weight, const SArray& size) {
      TA_ASSERT(weight.size() == size.size());
      calc_weight_helper(size.rbegin(), size.rend(), weight.rbegin());
    }

    template <typename Index, typename WeightArray, typename StartArray>
    inline typename Index::value_type calc_ordinal(const Index& index, const WeightArray& weight, const StartArray& start) {
      typename Index::value_type o = 0;
      const typename Index::value_type dim = index.size();
      for(std::size_t i = 0ul; i < dim; ++i)
        o += (index[i] - start[i]) * weight[i];

      return o;
    }


    template <typename Index, typename Rng>
    inline void calc_index(Index& index, std::size_t o, const Rng& range) {
      const std::size_t dim = index.size();

      for(std::size_t i = 0ul; i < dim; ++i) {
        const typename Index::value_type x = (o / range.weight()[i]);
        o -= x * range.weight()[i];
        index[i] = x + range.start()[i];
      }
    }

    template <typename ForIter, typename InIterStart, typename InIterFinish>
    inline void increment_coordinate_helper(ForIter first_cur, ForIter last_cur, InIterStart start, InIterFinish finish) {
      for(; first_cur != last_cur; ++first_cur, ++start, ++finish) {
        // increment coordinate
        ++(*first_cur);

        // break if done
        if( *first_cur < *finish)
          return;

        // Reset current index to start value.
        *first_cur = *start;
      }
    }

    template <typename Index, typename Rng>
    inline void increment_coordinate(Index& index, const Rng& range) {
      for(int dim = int(range.dim() - 1); dim >= 0; --dim) {
        // increment coordinate
        ++index[dim];

        // break if done
        if(index[dim] < range.finish()[dim])
          return;

        // Reset current index to start value.
        index[dim] = range.start()[dim];
      }

      // if the current location was set to start then it was at the end and
      // needs to be reset to equal finish.
      std::copy(range.finish().begin(), range.finish().end(), index.begin());
    }

  }  // namespace detail

  /// Range data of an N-dimensional tensor.
  class Range {
  public:
    typedef Range Range_;
    typedef std::size_t size_type;
    typedef std::vector<std::size_t> index;
    typedef detail::SizeArray<std::size_t> size_array;
    typedef detail::RangeIterator<index, Range_> const_iterator;
    friend class detail::RangeIterator<index, Range_>;

    /// Default constructor. The range has 0 size and the origin is set at 0.
    Range() :
        start_(), finish_(), size_(), weight_(), volume_(0ul)
    {}

    /// Constructor defined by an upper and lower bound. All elements of
    /// finish must be greater than or equal to those of start.
    template <typename Index>
    Range(const Index& start, const Index& finish) :
        start_(),
        finish_(),
        size_(),
        weight_(),
        volume_(start.size() ? 1ul : 0ul)
    {
      TA_ASSERT(start.size() == finish.size());
      TA_ASSERT( (std::equal(start.begin(), start.end(), finish.begin(),
          std::less_equal<std::size_t>())) );

      // Initialize array memory
      const size_type n = start.size();
      init_arrays(new size_type[n * 4], n);

      // Set array data
      for(int i = n - 1; i >= 0; --i) {
        start_[i] = start[i];
        finish_[i] = finish[i];
        size_[i] = finish[i] - start[i];
        weight_[i] = (size_[i] != 0ul ? volume_ : 0);
        volume_ *= size_[i];
      }
    }

    template <typename SizeArray>
    explicit Range(const SizeArray& size) :
        start_(),
        finish_(),
        size_(),
        weight_(),
        volume_(size.size() ? 1ul : 0ul)
    {
      // Initialize array memory
      const size_type n = size.size();
      init_arrays(new size_type[n * 4], n);

      // Set array data
      for(int i = n - 1; i >= 0; --i) {
        start_[i] = 0ul;
        finish_[i] = size[i];
        size_[i] = size[i];
        weight_[i] = (size[i] != 0ul ? volume_ : 0);
        volume_ *= size[i];
      }
    }

    /// Copy Constructor
    Range(const Range_& other) : // no throw
        start_(), finish_(), size_(), weight_(), volume_(other.volume_)
    {
      const size_type n = other.start_.size();
      init_arrays(new size_type[n * 4], n);
      memcpy(start_.data(), other.start_.begin(), sizeof(size_type) * 4 * n);
    }

    ~Range() {
      delete_arrays();
    }

    Range_& operator=(const Range_& other) {
      const size_type n = other.start_.size();

      if(start_.size() != n) {
        delete_arrays();
        init_arrays(new size_type[n * 4], n);
      }

      memcpy(start_.data(), other.start_.begin(), sizeof(size_type) * 4 * n);

      return *this;
    }

    unsigned int dim() const { return size_.size(); }

    /// Returns the lower bound of the range
    const size_array& start() const { return start_; } // no throw

    /// Returns the upper bound of the range
    const size_array& finish() const { return finish_; } // no throw

    /// Returns an array with the size of each dimension.
    const size_array& size() const { return size_; } // no throw

    const size_array& weight() const { return weight_; } // no throw


    /// Range volume accessor

    /// \return The total number of elements in the range.
    size_type volume() const { return volume_; }

    /// Index iterator factory

    /// The iterator dereferences to an index. The order of iteration matches
    /// the data layout of a dense tensor.
    /// \return An iterator that holds the start element index of a tensor.
    const_iterator begin() const { return const_iterator(start_, this); }

    /// Index iterator factory

    /// The iterator dereferences to an index. The order of iteration matches
    /// the data layout of a dense tensor.
    /// \return An iterator that holds the finish element index of a tensor.
    const_iterator end() const { return const_iterator(finish_, this); }

    /// Check the coordinate to make sure it is within the range.

    /// \tparam Index The coordinate index array type
    /// \param index The coordinate index to check for inclusion in the range
    /// \return \c true when \c i \c >= \c start and \c i \c < \c f, otherwise
    /// \c false
    template <typename Index>
    typename madness::disable_if<std::is_integral<Index>, bool>::type
    includes(const Index& index) const {
      TA_ASSERT(index.size() == dim());
      const unsigned int end = dim();
      for(unsigned int i = 0ul; i < end; ++i)
        if((index[i] < start_[i]) || (index[i] >= finish_[i]))
          return false;

      return true;
    }

    /// Check the ordinal index to make sure it is within the range.

    /// \param i The ordinal index to check for inclusion in the range
    /// \return \c true when \c i \c >= \c 0 and \c i \c < \c volume
    template <typename Ordinal>
    typename madness::enable_if<std::is_integral<Ordinal>, bool>::type
    includes(Ordinal i) const {
      return include_ordinal_(i);
    }

    /// Permute the tile given a permutation.
    Range_& operator ^=(const Permutation& p) {
      TA_ASSERT(p.dim() == dim());
      Range_ temp(p ^ start_, p ^ finish_);
      temp.swap(*this);

      return *this;
    }

    /// Change the dimensions of the range.
    template <typename Index>
    Range_& resize(const Index& start, const Index& finish) {
      Range_ temp(start, finish);
      temp.swap(*this);

      return *this;
    }

    /// calculate the ordinal index of \c i

    /// This function is just a pass-through so the user can call \c ord() on
    /// a template parameter that can be an index or a size_type.
    /// \param i Ordinal index
    /// \return \c i (unchanged)
    template <typename Ordinal>
    typename madness::enable_if<std::is_integral<Ordinal>, Ordinal>::type
    ord(Ordinal i) const {
      TA_ASSERT(includes(i));
      return i;
    }

    /// calculate the ordinal index of \c i

    /// Convert an index to an ordinal index.
    /// \param index The index to be converted to an ordinal index
    /// \return The ordinal index of the index \c i
    template <typename Index>
    typename madness::disable_if<std::is_integral<Index>, size_type>::type
    ord(const Index& index) const {
      TA_ASSERT(index.size() == dim());
      TA_ASSERT(includes(index));
      size_type o = 0;
      const unsigned int end = dim();
      for(unsigned int i = 0ul; i < end; ++i)
        o += (index[i] - start_[i]) * weight_[i];

      return o;
    }

    /// calculate the index of \c i

    /// Convert an ordinal index to an index.
    /// \param o Ordinal index
    /// \return The index of the ordinal index
    template <typename Ordinal>
    typename madness::enable_if<std::is_integral<Ordinal>, index>::type
    idx(Ordinal o) const {
      TA_ASSERT(includes(o));
      index i;
      detail::calc_index(size_index_(i), o, *this);
      return i;
    }

    /// calculate the index of \c i

    /// This function is just a pass-through so the user can call \c idx() on
    /// a template parameter that can be an index or a size_type.
    /// \param i The index
    /// \return \c i (unchanged)
    template <typename Index>
    typename madness::disable_if<std::is_integral<Index>, const Index&>::type
    idx(const Index& i) const {
      TA_ASSERT(includes(i));
      return i;
    }

    template <typename Archive>
    typename madness::enable_if<madness::archive::is_input_archive<Archive> >::type
    serialize(const Archive& ar) {
      size_type n = 0ul;

      // Get number of dimensions
      ar & n;
      const size_type n4 = n * 4;
      if(start_.size() != n) {
        delete_arrays();
        init_arrays(new size_type[n4], n);
      }
      ar & madness::archive::wrap(start_.data(), n4) & volume_;
    }

    template <typename Archive>
    typename madness::enable_if<madness::archive::is_output_archive<Archive> >::type
    serialize(const Archive& ar) const {
      const size_type n = start_.size();
      ar & n & madness::archive::wrap(start_.data(), n * 4) & volume_;
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

    void init_arrays(size_type* data, const size_type n) {
      start_.set(data, n);
      finish_.set(start_.end(), n);
      size_.set(finish_.end(), n);
      weight_.set(size_.end(), n);
    }

    void delete_arrays() {
      if(! start_.empty())
        delete [] start_.data();
    }

    template <typename T>
    std::vector<T>& size_index_(std::vector<T>& i) const {
      if(i.size() != dim())
        i.resize(dim());

      return i;
    }

    template <typename T>
    T& size_index_(T& i) const {
      TA_ASSERT(i.size() == dim());
      return i;
    }

    template <typename Ordinal>
    typename madness::enable_if<std::is_signed<Ordinal>, bool>::type
    include_ordinal_(Ordinal i) const {
      return (i >= 0ul) && (i < volume_);
    }

    template <typename Ordinal>
    typename madness::disable_if<std::is_signed<Ordinal>, bool>::type
    include_ordinal_(Ordinal i) const {
      return i < volume_;
    }

    void increment(index& i) const {
      detail::increment_coordinate(i, *this);
    }

    void advance(index& i, std::ptrdiff_t n) const {
      const size_type o = ord(i) + n;

      if(n >= volume_)
        std::copy(finish_.begin(), finish_.end(), i.begin());
      else
        i = idx(o);
    }

    std::ptrdiff_t distance_to(const index& first, const index& last) const {
      return ord(last) - ord(first);
    }

    size_array start_;     ///< Tile origin
    size_array finish_;    ///< Tile upper bound
    size_array size_;      ///< Dimension sizes
    size_array weight_;    ///< Dimension weights
    size_type volume_;///< Total number of elements
  }; // class Range



  /// Exchange the values of the give two ranges.
  inline void swap(Range& r0, Range& r1) { // no throw
    r0.swap(r1);
  }

  /// Return the union of two range (i.e. the overlap). If the ranges do not
  /// overlap, then a 0 size range will be returned.
//  template <typename Derived1, typename Derived2>
//  Range<CS> operator &(const Range<CS>& b1, const Range<CS>& b2) {
//    Range<CS> result;
//    typename Range<CS>::index start, finish;
//    typename Range<CS>::index::value_type s1, s2, f1, f2;
//    for(unsigned int d = 0; d < CS::dim; ++d) {
//      s1 = b1.start()[d];
//      f1 = b1.finish()[d];
//      s2 = b2.start()[d];
//      f2 = b2.finish()[d];
//      // check for overlap
//      if( (s2 < f1 && s2 >= s1) || (f2 < f1 && f2 >= s1) ||
//          (s1 < f2 && s1 >= s2) || (f1 < f2 && f1 >= s2) )
//      {
//        start[d] = std::max(s1, s2);
//        finish[d] = std::min(f1, f2);
//      } else {
//        return result; // no overlap for this index
//      }
//    }
//    result.resize(start, finish);
//    return result;
//  }

  /// Returns a permuted range.
  inline Range operator ^(const Permutation& perm, const Range& r) {
    TA_ASSERT(perm.dim() == r.dim());
    return Range(perm ^ r.start(), perm ^ r.finish());
  }

  inline const Range& operator ^(const detail::NoPermutation& perm, const Range& r) {
    return r;
  }

  /// Returns true if the start and finish are equal.
  inline bool operator ==(const Range& r1, const Range& r2) {
#ifdef NDEBUG
    return (r1.dim() == r2.dim()) &&
        ( std::equal(r1.start().begin(), r1.start().end(), r2.start().begin()) ) &&
        ( std::equal(r1.finish().begin(), r1.finish().end(), r2.finish().begin()) );
#else
    return (r1.dim() == r2.dim()) &&
        ( std::equal(r1.start().begin(), r1.start().end(), r2.start().begin()) ) &&
        ( std::equal(r1.finish().begin(), r1.finish().end(), r2.finish().begin()) ) &&
        ( std::equal(r1.size().begin(), r1.size().end(), r2.size().begin()) ) &&
        ( std::equal(r1.weight().begin(), r1.weight().end(), r2.weight().begin()) ); // do an extra size check to catch bugs.
#endif
  }

  /// Returns true if the start and finish are not equal.
  inline bool operator !=(const Range& r1, const Range& r2) {
    return ! operator ==(r1, r2);
  }

  /// range output operator.
  inline std::ostream& operator<<(std::ostream& out, const Range& r) {
    out << "[ ";
    detail::print_array(out, r.start());
    out << ", ";
    detail::print_array(out, r.finish());
    out << " )";
    return out;
  }

} // namespace TiledArray
#endif // TILEDARRAY_RANGE_H__INCLUDED
