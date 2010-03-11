#ifndef TILEDARRAY_RANGE_H__INCLUDED
#define TILEDARRAY_RANGE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/coordinates.h>
#include <TiledArray/array_util.h>
#include <TiledArray/iterator.h>
//#include <boost/array.hpp>


namespace TiledArray {

  // Forward declaration of TiledArray components.
  template <unsigned int DIM>
  class Permutation;
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  class Range;
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  void swap(ArrayCoordinate<I,DIM,Tag,CS>& c1, ArrayCoordinate<I,DIM,Tag,CS>& c2);
  template <typename T, std::size_t DIM>
  T volume(const boost::array<T,DIM>&);
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  void swap(Range<I,DIM,Tag,CS>&, Range<I,DIM,Tag,CS>&);
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  Range<I,DIM,Tag,CS> operator &(const Range<I,DIM,Tag,CS>&, const Range<I,DIM,Tag,CS>&);
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  Range<T,DIM,Tag,CS> operator ^(const Permutation<DIM>&, const Range<T,DIM,Tag,CS>&);
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  bool operator ==(const Range<I,DIM,Tag,CS>&, const Range<I,DIM,Tag,CS>&);
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  bool operator !=(const Range<I,DIM,Tag,CS>&, const Range<I,DIM,Tag,CS>&);
  template <typename T, typename Tag, typename CS>
  Range<T,1,Tag,CS> operator ^(const Permutation<1>&, const Range<T,1,Tag,CS>&);
  template<typename I, unsigned int DIM, typename Tag, typename CS>
  std::ostream& operator<<(std::ostream&, const Range<I,DIM,Tag,CS>&);

  /// Range stores dimension information for a block of tiles or elements.

  /// Range is used to obtain and/or store start, finish, size, and volume
  /// information. It also provides index iteration over its range.
  template <typename I, unsigned int DIM, typename Tag = LevelTag<0>, typename CS = CoordinateSystem<DIM> >
  class Range {
    BOOST_STATIC_ASSERT(DIM < TA_MAX_DIM);

  public:
    typedef Range<I,DIM,Tag,CS> Range_;
    typedef I ordinal_type;
    typedef ArrayCoordinate<I,DIM,Tag,CS> index_type;
    typedef I volume_type;
    typedef boost::array<I,DIM> size_array;
    typedef CS coordinate_system;

    typedef detail::IndexIterator<index_type, Range_> const_iterator;
    friend class detail::IndexIterator< index_type , Range_ >;

    static unsigned int dim() { return DIM; }

    /// Default constructor. The range has 0 size and the origin is set at 0.
    Range() :
        start_(), finish_(), size_()
    {}

    /// Construct a range of size with the origin set at 0 for all dimensions.
    Range(const size_array& size, const index_type& start = index_type()) :
        start_(start), finish_(start + size), size_(size)
    {
      TA_ASSERT( (detail::less_eq<I,DIM>(start_.data(), finish_.data())) ,
          std::runtime_error, "Finish is less than start.");
    }

    /// Constructor defined by an upper and lower bound. All elements of
    /// finish must be greater than or equal to those of start.
    Range(const index_type& start, const index_type& finish) :
        start_(start), finish_(finish), size_(finish - start)
    {
      TA_ASSERT( (detail::less_eq<I,DIM>(start_.data(), finish.data())) ,
          std::runtime_error, "Finish is less than start.");
    }

    /// Copy Constructor
    Range(const Range_& other) : // no throw
        start_(other.start_), finish_(other.finish_), size_(other.size_)
    {}

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// Move Constructor
    Range(Range_&& other) : // no throw
        start_(std::move(other.start_)), finish_(std::move(other.finish_)), size_(std::move(other.size_))
    {}
#endif // __GXX_EXPERIMENTAL_CXX0X__

    ~Range() {}

    // iterator factory functions
    const_iterator begin() const { return const_iterator(start_, this); }
    const_iterator end() const { return const_iterator(finish_, this); }

    /// Returns the lower bound of the range
    const index_type& start() const { return start_; } // no throw

    /// Returns the upper bound of the range
    const index_type& finish() const { return finish_; } // no throw

    /// Returns an array with the size of each dimension.
    const size_array& size() const { return size_.data(); } // no throw

    /// Returns the number of elements in the range.
    volume_type volume() const {
      return detail::volume(size_.data());
    }

    /// Check the coordinate to make sure it is within the range.
    bool includes(const index_type& i) const {
      return (detail::less_eq(start_.data(), i.data()) &&
          detail::less(i.data(), finish_.data()));
    }

    bool includes(const ordinal_type& o) const {
      return o < volume();
    }

    /// Assignment Operator.
    Range_& operator =(const Range_& other) {
      Range_ temp(other);
      swap(*this, temp);
      return *this;
    }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// Assignment Operator.
    Range_& operator =(Range_&& other) {
      start_ = std::move(other.start_);
      finish_ = std::move(other.finish_);
      size_ = std::move(other.size_);

      return *this;
    }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    /// Permute the tile given a permutation.
    Range_& operator ^=(const Permutation<DIM>& p) {
      Range_ temp(*this);
  	  temp.start_ ^= p;
  	  temp.finish_ ^= p;
  	  temp.size_ ^= p;

  	  swap(*this, temp);
      return *this;
    }

    /// Change the dimensions of the range.
    Range_& resize(const index_type& start, const index_type& finish) {
      Range_ temp(start, finish);
      swap(*this, temp);
      return *this;
    }

    /// Change the dimensions of the range.
    Range_& resize(const size_array& size) {
      Range_ temp(size, start_);
      swap(*this, temp);
      return *this;
    }

    template <typename Archive>
    void serialize(const Archive& ar) {
      ar & start_ & finish_ & size_;
    }

  private:

    void increment(index_type& i) const {
      detail::IncrementCoordinate<index_type,coordinate_system>(i, start_, finish_);
    }

    friend   void swap<>(Range<I,DIM,Tag,CS>&, Range<I,DIM,Tag,CS>&);

    index_type start_;              // Tile origin
    index_type finish_;             // Tile upper bound
    index_type size_;               // Dimension sizes

  }; // class Range

  /// compute the volume of the orthotope bounded by the origin and a
  template <typename T, std::size_t DIM>
  T volume(const boost::array<T,DIM>& a) {
    T result = 1;
    for(std::size_t d = 0; d < DIM; ++d)
      result *= std::abs(static_cast<long int>(a[d]));
    return result;
  }

  /// Exchange the values of the give two ranges.
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  void swap(Range<I,DIM,Tag,CS>& r0, Range<I,DIM,Tag,CS>& r1) { // no throw
    TiledArray::swap(r0.start_, r1.start_);
    TiledArray::swap(r0.finish_, r1.finish_);
    TiledArray::swap(r0.size_, r1.size_);
  }

  /// Return the union of two range (i.e. the overlap). If the ranges do not
  /// overlap, then a 0 size range will be returned.
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  Range<I,DIM,Tag,CS> operator &(const Range<I,DIM,Tag,CS>& b1, const Range<I,DIM,Tag,CS>& b2) {
    Range<I,DIM,Tag,CS> result;
    typename Range<I,DIM,Tag,CS>::index_type start, finish;
    typename Range<I,DIM,Tag,CS>::index_type::index s1, s2, f1, f2;
    for(unsigned int d = 0; d < DIM; ++d) {
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
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  Range<T,DIM,Tag,CS> operator ^(const Permutation<DIM>& perm, const Range<T,DIM,Tag,CS>& r) {
    const typename Range<T,DIM,Tag,CS>::index_type s = perm ^ r.start();
    const typename Range<T,DIM,Tag,CS>::index_type f = perm ^ r.finish();
    Range<T,DIM,Tag,CS> result(s, f);
    return result;
  }

  /// Returns true if the start and finish are equal.
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  bool operator ==(const Range<I,DIM,Tag,CS>& r1, const Range<I,DIM,Tag,CS>& r2) {
#ifdef NDEBUG
    return ( r1.start() == r2.start() ) && ( r1.finish() == r2.finish() );
#else
    return ( r1.start() == r2.start() ) && ( r1.finish() == r2.finish() ) &&
        (r1.size() == r2.size()); // do an extra size check to catch bugs.
#endif
  }

  /// Returns true if the start and finish are not equal.
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  bool operator !=(const Range<I,DIM,Tag,CS>& r1, const Range<I,DIM,Tag,CS>& r2) {
    return ! operator ==(r1, r2);
  }

  /// Returns a permuted range.
  template <typename T, typename Tag, typename CS>
  Range<T,1,Tag,CS> operator ^(const Permutation<1>& perm, const Range<T,1,Tag,CS>& r) {
    return r;
  }

  /// ostream output orperator.
  template<typename I, unsigned int DIM, typename Tag, typename CS>
  std::ostream& operator<<(std::ostream& out, const Range<I,DIM,Tag,CS>& blk) {
    out << "[ " << blk.start() << ", " << blk.finish() << " )";
    return out;
  }

} // namespace TiledArray
#endif // TILEDARRAY_RANGE_H__INCLUDED
