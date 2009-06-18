#ifndef BLOCK_H__INCLUDED
#define BLOCK_H__INCLUDED

#include <coordinates.h>
#include <iterator.h>
#include <boost/array.hpp>
#include <cassert>

namespace TiledArray {

  // Forward declaration of TiledArray components.
  template <unsigned int>
  class LevelTag;
  template <unsigned int DIM>
  class Permutation;
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  void swap(ArrayCoordinate<I,DIM,Tag,CS>& c1, ArrayCoordinate<I,DIM,Tag,CS>& c2);

  template <typename I, std::size_t DIM, typename CS>
  boost::array<I,DIM> calc_weights(const boost::array<I,DIM>&);
  template <typename I, std::size_t DIM>
  I volume(const boost::array<I,DIM>&);

  /// Block stores dimension information for a block of tiles or elements.

  /// Block is used to obtain and/or store start, finish, size, and volume
  /// information. It also provides index iteration over its range.
  template <typename I, unsigned int DIM, typename Tag = LevelTag<0>, typename CS = CoordinateSystem<DIM> >
  class Block {
  public:
    typedef Block<I,DIM,Tag,CS> Block_;
    typedef ArrayCoordinate<I,DIM,Tag,CS> index_type;
    typedef I volume_type;
    typedef boost::array<I,DIM> size_array;
    typedef CS coordinate_system;

    typedef detail::IndexIterator<index_type, Block_> const_iterator;
    friend class detail::IndexIterator< index_type , Block_ >;

    static const unsigned int dim() { return DIM; }

    /// Default constructor. The block has 0 size and the origin is set at 0.
    Block() :
        start_(0), finish_(0), size_(0)
    {}

    /// Construct a block of size with the origin of the block set at 0 for all dimensions.
    Block(const size_array& size, const index_type& start = index_type(0)) :
        start_(start), finish_(start + size), size_(size)
    {
#ifndef NDEBUG
      bool valid = ! detail::less<I,DIM,CS>(finish_.data(), start_.data());
      assert( valid );
#endif
    }

    /// Constructor defined by an upper and lower bound. All elements of
    /// finish must be greater than or equal to those of start.
    Block(const index_type& start, const index_type& finish) :
        start_(start), finish_(finish), size_(finish - start)
    {
#ifndef NDEBUG
      bool valid = ! detail::less<I,DIM,CS>(finish_.data(), start_.data());
      assert( valid );
#endif
    }

    /// Copy Constructor
    Block(const Block_& other) : // no throw
        start_(other.start_), finish_(other.finish_), size_(other.size_)
    {}

    ~Block() {}

    // iterator factory functions
    const_iterator begin() const { return const_iterator(start_, this); }
    const_iterator end() const { return const_iterator(finish_, this); }

    /// Returns the lower bound of the block
    const index_type& start() const { return start_; }

    /// Returns the upper bound of the block
    const index_type& finish() const { return finish_; }

    /// Returns an array with the size of each dimension.
    const size_array& size() const { return size_.data(); }

    /// Returns the number of elements in the block.
    volume_type volume() const {
      volume_type result = 1;
      for(unsigned int d = 0; d < DIM; ++d)
        result *= ( size_[d] < 0 ? -size_[d] : size_[d] );
      return result;
    }

    /// Check the coordinate to make sure it is within the block range
    bool includes(const index_type& i) const {
      detail::Less<I,DIM,CS> l;
      detail::LessEq<I,DIM,CS> le;
      return (le(start_.data(), i.data()) && l(i.data(), finish_.data()));
    }

    /// Assignment Operator.
    Block_& operator =(const Block_& other) {
      Block temp(other);
      swap(temp);
      return *this;
    }

    /// Permute the tile given a permutation.
    Block_& operator ^=(const Permutation<DIM>& p) {
      Block temp(*this);
  	  temp.start_ ^= p;
  	  temp.finish_ ^= p;
  	  temp.size_ ^= p;

  	  swap(temp);
      return *this;
    }

    /// Change the dimensions of the block.
    Block& resize(const index_type& start, const index_type& finish) {
      Block temp(start, finish);
      swap(temp);
      return *this;
    }

    /// Change the dimensions of the block.
    Block& resize(const size_array& size) {
      Block temp(size, start_);
      swap(temp);
      return *this;
    }

    template <typename Archive>
    void serialize(const Archive& ar) {
      ar & start_ & finish_ & size_;
    }

    void swap(Block& other) { // no throw
      ::TiledArray::swap(start_, other.start_);
      ::TiledArray::swap(finish_, other.finish_);
      ::TiledArray::swap(size_, other.size_);
    }

  private:

    void increment(index_type& i) const {
      detail::IncrementCoordinate<index_type,coordinate_system>(i, start_, finish_);
    }

    index_type start_;              // Tile origin
    index_type finish_;             // Tile upper bound
    index_type size_;               // Dimension sizes

  }; // class Block

  /// compute the volume of the orthotope bounded by the origin and a
  template <typename T, std::size_t DIM>
  T volume(const boost::array<T,DIM>& a) {
    T result = 1;
    for(std::size_t d = 0; d < DIM; ++d)
      result *= std::abs(static_cast<long int>(a[d]));
    return result;
  }

  /// Return the union of two block (i.e. the overlap). If the blocks do not
  /// overlap, then a 0 size block will be returned.
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  Block<I,DIM,Tag,CS> operator &(const Block<I,DIM,Tag,CS>& b1, const Block<I,DIM,Tag,CS>& b2) {
    Block<I,DIM,Tag,CS> result;
    typename Block<I,DIM,Tag,CS>::index_type start, finish;
    register typename Block<I,DIM,Tag,CS>::index_type::index s1, s2, f1, f2;
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

  /// Returns a permuted block.
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  Block<T,DIM,Tag,CS> operator ^(const Permutation<DIM>& perm, const Block<T,DIM,Tag,CS>& b) {
    const typename Block<T,DIM,Tag,CS>::index_type s = perm ^ b.start();
    const typename Block<T,DIM,Tag,CS>::index_type f = perm ^ b.finish();
    Block<T,DIM,Tag,CS> result(s, f);
    return result;
  }

  /// Returns true if the start and finish are equal.
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  bool operator ==(const Block<I,DIM,Tag,CS>& b1, const Block<I,DIM,Tag,CS>& b2) {
    return ( b1.start() == b2.start() ) && ( b1.finish() == b2.finish() );
  }

  /// Returns true if the start and finish are not equal.
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  bool operator !=(const Block<I,DIM,Tag,CS>& b1, const Block<I,DIM,Tag,CS>& b2) {
    return ( b1.start() != b2.start() ) || ( b1.finish() != b2.finish() );
  }

  /// ostream output orperator.
  template<typename I, unsigned int DIM, typename Tag, typename CS>
  std::ostream& operator<<(std::ostream& out, const Block<I,DIM,Tag,CS>& blk) {
    out << "[ " << blk.start() << ", " << blk.finish() << " )";
    return out;
  }

} // namespace TiledArray
#endif // BLOCK_H__INCLUDED
