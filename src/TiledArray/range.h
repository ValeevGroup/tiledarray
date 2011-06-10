#ifndef TILEDARRAY_RANGE_H__INCLUDED
#define TILEDARRAY_RANGE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/coordinate_system.h>
#include <TiledArray/coordinates.h>
#include <TiledArray/range_iterator.h>
#include <TiledArray/type_traits.h>
#include <boost/utility/enable_if.hpp>
#include <algorithm>

namespace TiledArray {

  // Forward declaration of TiledArray components.
  template <unsigned int>
  class Permutation;
  template <typename>
  class Range;
  template <typename I, unsigned int DIM, typename Tag>
  void swap(ArrayCoordinate<I,DIM,Tag>& c1, ArrayCoordinate<I,DIM,Tag>& c2);
  template <typename CS>
  void swap(Range<CS>&, Range<CS>&);
  template <typename CS>
  Range<CS> operator &(const Range<CS>&, const Range<CS>&);
  template <unsigned int DIM, typename CS>
  Range<CS> operator ^(const Permutation<DIM>&, const Range<CS>&);
  template <typename CS>
  bool operator ==(const Range<CS>&, const Range<CS>&);
  template <typename CS>
  bool operator !=(const Range<CS>&, const Range<CS>&);
  template <typename CS>
  Range<CS> operator ^(const Permutation<1>&, const Range<CS>&);
  template <typename CS>
  std::ostream& operator<<(std::ostream&, const Range<CS>&);

  /// Range stores dimension information for a block of tiles or elements.

  /// Range is used to obtain and/or store start, finish, size, and volume
  /// information. It also provides index iteration over its range.
  template <typename CS>
  class Range {
  public:
    typedef Range<CS> Range_;
    typedef CS coordinate_system;

    typedef typename CS::volume_type volume_type;
    typedef typename CS::index index;
    typedef typename CS::ordinal_index ordinal_index;
    typedef typename CS::size_array size_array;

    typedef detail::RangeIterator<index, Range_> const_iterator;
    friend class detail::RangeIterator< index , Range_ >;

    /// Default constructor. The range has 0 size and the origin is set at 0.
    Range() :
        start_(), finish_(), size_(), weight_()
    {}

    /// Constructor defined by an upper and lower bound. All elements of
    /// finish must be greater than or equal to those of start.
    Range(const index& start, const index& finish) :
        start_(start), finish_(finish), size_(finish - start), weight_(coordinate_system::calc_weight(size_.data()))
    {
      TA_ASSERT( (std::equal(start_.begin(), start_.end(), finish_.begin(), std::less_equal<typename coordinate_system::ordinal_index>())) ,
          std::runtime_error, "Finish is less than start.");

    }

    /// Copy Constructor
    Range(const Range_& other) : // no throw
        start_(other.start_), finish_(other.finish_), size_(other.size_),
        weight_(other.weight_)
    {}

    ~Range() {}

    // iterator factory functions
    const_iterator begin() const { return const_iterator(start_, this); }
    const_iterator end() const { return const_iterator(finish_, this); }

    /// Returns the lower bound of the range
    const index& start() const { return start_; } // no throw

    /// Returns the upper bound of the range
    const index& finish() const { return finish_; } // no throw

    /// Returns an array with the size of each dimension.
    const size_array& size() const { return size_.data(); } // no throw

    const size_array& weight() const { return weight_.data(); } // no throw

    /// Returns the number of elements in the range.
    volume_type volume() const {
      return coordinate_system::calc_volume(size_.data());
    }

    /// Check the coordinate to make sure it is within the range.

    /// \param i The coordinate index to check for inclusion in the range
    /// \return \c true when \c i \c >= \c start and \c i \c < \c f, otherwise
    /// \c false
    bool includes(const index& i) const {
      typename size_array::const_iterator start_it = start_.begin();
      typename size_array::const_iterator finish_it = finish_.begin();
      for(typename index::const_iterator it = i.begin(); it != i.end(); ++it, ++start_it, ++finish_it)
        if((*it < *start_it) || (*it >= *finish_it))
          return false;

      return true;
    }

    /// Check the ordinal index to make sure it is within the range.

    /// \param i The ordinal index to check for inclusion in the range
    /// \return \c true when \c i \c >= \c 0 and \c i \c < \c volume
    bool includes(const ordinal_index& i) const {
      return include_ordinal_(i);
    }

    /// Assignment Operator.
    Range_& operator =(const Range_& other) {
      start_ = other.start_;
      finish_ = other.finish_;
      size_ = other.size_;
      weight_ = other.weight_;
      return *this;
    }

    /// Permute the tile given a permutation.
    Range_& operator ^=(const Permutation<coordinate_system::dim>& p) {
      Range_ temp(p ^ start_, p ^ finish_);
  	  swap(temp);

  	  return *this;
    }

    /// Change the dimensions of the range.
    Range_& resize(const index& start, const index& finish) {
      Range_ temp(start, finish);
      swap(temp);

      return *this;
    }

    template <typename Archive>
    void serialize(const Archive& ar) {
      ar & start_ & finish_ & size_ & weight_;
    }

    void swap(Range_& other) {
      TiledArray::swap(start_, other.start_);
      TiledArray::swap(finish_, other.finish_);
      TiledArray::swap(size_, other.size_);
      TiledArray::swap(weight_, other.weight_);
    }

  private:

    template <typename T>
    typename boost::enable_if<std::is_signed<T>, bool>::type
    include_ordinal_(const T& i) const {
      return (i >= 0) && (i < volume());
    }

    template <typename T>
    typename boost::disable_if<std::is_signed<T>, bool>::type
    include_ordinal_(const T& i) const {
      return i < volume();
    }

    void increment(index& i) const {
      coordinate_system::increment_coordinate(i, start_, finish_);
    }

    index start_;    ///< Tile origin
    index finish_;   ///< Tile upper bound
    index size_;     ///< Dimension sizes
    index weight_;   ///< Dimension weights
  }; // class Range

  /// Range stores dimension information for a block of tiles or elements.

  /// Range is used to obtain and/or store start, finish, size, and volume
  /// information. It also provides index iteration over its range.
  template <unsigned int L, detail::DimensionOrderType O, typename I>
  class Range<CoordinateSystem<1u, L, O, I> > {
  public:
    typedef Range<CoordinateSystem<1u, L, O, I> > Range_;
    typedef CoordinateSystem<1u, L, O, I> coordinate_system;

    typedef typename coordinate_system::volume_type volume_type;
    typedef typename coordinate_system::ordinal_index index;
    typedef typename coordinate_system::ordinal_index ordinal_index;
    typedef typename coordinate_system::ordinal_index size_array;

    typedef detail::RangeIterator<index, Range_> const_iterator;
    friend class detail::RangeIterator< index , Range_ >;

    /// Default constructor. The range has 0 size and the origin is set at 0.
    Range() :
        start_(), finish_(), size_()
    {}

    /// Constructor defined by an upper and lower bound. All elements of
    /// finish must be greater than or equal to those of start.
    Range(const index start, const index finish) :
        start_(start), finish_(finish), size_(finish - start)
    {
      TA_ASSERT( start <= finish ,
          std::runtime_error, "Finish is less than start.");
    }

    /// Copy Constructor
    Range(const Range_& other) : // no throw
        start_(other.start_), finish_(other.finish_), size_(other.size_)
    {}

    ~Range() {}

    // iterator factory functions
    const_iterator begin() const { return const_iterator(start_, this); }
    const_iterator end() const { return const_iterator(finish_, this); }

    /// Returns the lower bound of the range
    index start() const { return start_; } // no throw

    /// Returns the upper bound of the range
    index finish() const { return finish_; } // no throw

    /// Returns an array with the size of each dimension.
    size_array size() const { return size_; } // no throw

    size_array weight() const { return 1; } // no throw

    /// Returns the number of elements in the range.
    volume_type volume() const {
      return size_;
    }

    /// Check the coordinate to make sure it is within the range.
    bool includes(const index& i) const {
      return (start_ <= i) && (i < finish_);
    }

    /// Assignment Operator.
    Range_& operator =(const Range_& other) {
      start_ = other.start_;
      finish_ = other.finish_;
      size_ = other.size_;
      return *this;
    }

    /// Permute the tile given a permutation.
    Range_& operator ^=(const Permutation<1>& p) { return *this; }

    /// Change the dimensions of the range.
    Range_& resize(const index& start, const index& finish) {
      Range_ temp(start, finish);
      swap(temp);

      return *this;
    }

    template <typename Archive>
    void serialize(const Archive& ar) {
      ar & start_ & finish_ & size_;
    }

    void swap(Range_& other) {
      std::swap(start_, other.start_);
      std::swap(finish_, other.finish_);
      std::swap(size_, other.size_);
    }

  private:

    void increment(index& i) const {
      ++i;
    }

    index start_;    ///< Tile origin
    index finish_;   ///< Tile upper bound
    index size_;     ///< Dimension sizes
  }; // class Range

  /// Exchange the values of the give two ranges.
  template <typename CS>
  void swap(Range<CS>& r0, Range<CS>& r1) { // no throw
    r0.swap(r1);
  }

  /// Return the union of two range (i.e. the overlap). If the ranges do not
  /// overlap, then a 0 size range will be returned.
  template <typename CS>
  Range<CS> operator &(const Range<CS>& b1, const Range<CS>& b2) {
    Range<CS> result;
    typename Range<CS>::index start, finish;
    typename Range<CS>::index::index s1, s2, f1, f2;
    for(unsigned int d = 0; d < CS::dim; ++d) {
      s1 = b1.start()[d];
      f1 = b1.finish()[d];
      s2 = b2.start()[d];
      f2 = b2.finish()[d];
      // check for overlap
      if( (s2 < f1 && s2 >= s1) || (f2 < f1 && f2 >= s1) ||
          (s1 < f2 && s1 >= s2) || (f1 < f2 && f1 >= s2) )
      {
        start[d] = std::max(s1, s2);
        finish[d] = std::min(f1, f2);
      } else {
        return result; // no overlap for this index
      }
    }
    result.resize(start, finish);
    return result;
  }

  /// Returns a permuted range.
  template <unsigned int DIM, typename CS>
  Range<CS> operator ^(const Permutation<DIM>& perm, const Range<CS>& r) {
    TA_STATIC_ASSERT(DIM == CS::dim);
    const typename Range<CS>::index s = perm ^ r.start();
    const typename Range<CS>::index f = perm ^ r.finish();
    Range<CS> result(s, f);
    return result;
  }

  /// Returns true if the start and finish are equal.
  template <typename CS>
  bool operator ==(const Range<CS>& r1, const Range<CS>& r2) {
#ifdef NDEBUG
    return ( r1.start() == r2.start() ) && ( r1.finish() == r2.finish() );
#else
    return ( r1.start() == r2.start() ) && ( r1.finish() == r2.finish() ) &&
        (r1.size() == r2.size()) && (r1.weight() == r2.weight()); // do an extra size check to catch bugs.
#endif
  }

  /// Returns true if the start and finish are not equal.
  template <typename CS>
  bool operator !=(const Range<CS>& r1, const Range<CS>& r2) {
    return ! operator ==(r1, r2);
  }

  /// Returns a permuted range.
  template <typename CS>
  Range<CS> operator ^(const Permutation<1>& perm, const Range<CS>& r) {
    TA_STATIC_ASSERT(CS::dim == 1);
    return r;
  }

  /// ostream output orperator.
  template<typename CS>
  std::ostream& operator<<(std::ostream& out, const Range<CS>& blk) {
    out << "[ " << blk.start() << ", " << blk.finish() << " )";
    return out;
  }

} // namespace TiledArray
#endif // TILEDARRAY_RANGE_H__INCLUDED
