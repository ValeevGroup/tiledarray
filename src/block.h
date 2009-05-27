#ifndef BLOCK_H__INCLUDED
#define BLOCK_H__INCLUDED

#include <coordinate_system.h>
#include <iterator.h>
#include <boost/array.hpp>
#include <cassert>

namespace TiledArray {

  // Forward declaration of TiledArray components.
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  class ArrayCoordinate;

  template <typename T, std::size_t DIM, typename CS>
  boost::array<T,DIM> calc_weights(const boost::array<T,DIM>&);
  template <typename T, std::size_t DIM>
  T volume(const boost::array<T,DIM>&);

  template <typename T, unsigned int DIM, typename Tag = LevelTag<0>, typename CS = CoordinateSystem<DIM> >
  class Block {
  public:
    typedef Block<T,DIM,Tag,CS> Block_;
    typedef ArrayCoordinate<T,DIM,Tag,CS> index_type;
    typedef typename index_type::index ordinal_type;
    typedef typename index_type::volume volume_type;
    typedef typename index_type::Array size_array;
    typedef CS coordinate_system;

    typedef detail::IndexIterator<index_type, Block_> const_iterator;
    friend class detail::IndexIterator< index_type , Block_ >;

    static const unsigned int dim() { return DIM; }

    /// Default constructor. The block has 0 size and the origin is set at 0.
    Block() :
        start_(0), finish_(0), size_(0),
        weights_(calc_weights(size_.data())), n_(0)
    {}

    /// Construct a block of size with the origin of the block set at 0 for all dimensions.
    Block(const size_array& size, const index_type& start = index_type(0)) :
        start_(start), finish_(start + size), size_(size),
        weights_(calc_weights(size)), n_(volume(size))
    { assert( finish_ >= start_); }

    /// Constructor defined by an upper and lower bound. All elements of
    /// finish must be greater than or equal to those of start.
    Block(const index_type& start, const index_type& finish) :
        start_(start), finish_(finish), size_(finish - start),
        weights_(calc_weights(size_.data())), n_(volume(size_.data()))
    { assert( finish_ >= start_); }

    /// Copy Constructor
    Block(const Block_& other) :
        start_(other.start_), finish_(other.finish_), size_(other.size_),
        weights_(other.weights_), n_(other.n_)
    {}

    ~Block() {}

    // iterator factory functions
    const_iterator begin() const { return const_iterator(start_, this); }
    const_iterator end() const { return const_iterator(finish_, this); }

    /// Returns an array with the size of each dimension.
    const size_array& size() const { return size_.data(); }

    /// Returns the lower bound of the block
    const index_type& start() const { return start_; }

    /// Returns the upper bound of the block
    const index_type& finish() const { return finish_; }

    /// Returns the number of elements contained in the array.
    volume_type volume() const { return n_; }

    /// Returns the index weights which are used to calculate the
    const size_array& weights() const { return weights_; }

    /// Check the coordinate to make sure it is within the block range
    bool includes(const index_type& i) const {
      return (i >= start()) && (i < finish());
    }

    /// computes an ordinal index for a given an index_type
    ordinal_type ordinal(const index_type& i) const {
      assert(includes(i));
      ordinal_type result = dot_product( (i - start_).data(), weights_);
      return result;
    }

    /// Assignment Operator.
    Block_& operator =(const Block_& other) {
      start_ = other.start_;
      finish_ = other.finish_;
      size_ = other.size_;
      std::copy(other.weights_.begin(), other.weights_.end(), weights_.begin());
      n_ = other.n_;
    }

    /// Permute the tile given a permutation.
    Block_& operator ^=(const Permutation<DIM>& p) {
  	  start_ ^= p;
  	  finish_ ^= p;
  	  size_ ^= p;
  	  weights_ = calc_weights(size_.data());

      return *this;
    }

    template <typename Archive>
    void serialize(const Archive& ar) {
      ar & start_ & finish_ & size_ & weights_ & n_ ;
    }

  private:

    void increment(index_type& i) const {
      // increment least significant, and check to see if the iterator has
      // reached the end of that dimension
      for(typename coordinate_system::const_iterator order_it = coordinate_system::begin();
          order_it != coordinate_system::end(); ++order_it) {
        // increment and break if done.
        if( (++( i[ *order_it ] ) ) < finish_[ *order_it ] )
          return;

        // The iterator is at the end of the dimension, set it to the lower bound.
        i[*order_it] = start_[*order_it];

        // Increment the next higher bound.
      }

      // Check for end (i.e. i was reset to start)
      if(i == start_)
        i = finish_;
    }

    static size_array calc_weights(const size_array& sizes) {
      return ::TiledArray::calc_weights<ordinal_type, static_cast<std::size_t>(DIM), coordinate_system>(sizes);
    }

    static volume_type volume(const size_array& sizes) {
      return ::TiledArray::volume<ordinal_type, static_cast<std::size_t>(DIM)>(sizes);
    }

    index_type start_;              // Tile origin
    index_type finish_;             // Tile upper bound
    index_type size_;               // Dimension sizes
    size_array weights_;            // Index weights used for calculating ordinal indices
    volume_type n_;                // Number of elements

  }; // class Block

  template <typename T, std::size_t DIM, typename CS>
  boost::array<T,DIM> calc_weights(const boost::array<T,DIM>& sizes) {
    typedef  detail::DimensionOrder<DIM> DimOrder;
    const DimOrder order = CS::ordering();
    boost::array<T,DIM> result;
    T weight = 1;


    for(typename DimOrder::const_iterator d = order.begin(); d != order.end(); ++d) {
      // calc ordinal weights.
      result[*d] = weight;
      weight *= sizes[*d];
    }

    return result;
  }

  /// compute the volume of the orthotope bounded by the origin and a
  template <typename T, std::size_t DIM>
  T volume(const boost::array<T,DIM>& a) {
    T result = 1;
    for(std::size_t d = 0; d < DIM; ++d)
      result *= std::abs(static_cast<long int>(a[d]));
    return result;
  }

  /// Returns a permuted block.
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  Block<T,DIM,Tag,CS> operator ^(const Permutation<DIM>& perm, const Block<T,DIM,Tag,CS>& b) {
    return Block<T,DIM,Tag,CS>(perm ^ b.start(), perm ^ b.finish());
  }

  /// Returns true if the start and finish are equal.
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  bool operator ==(const Block<T,DIM,Tag,CS>& b1, const Block<T,DIM,Tag,CS>& b2) {
    return ( b1.start() == b2.start() ) && ( b1.finish() == b2.finish() );
  }

  /// Returns true if the start and finish are not equal.
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  bool operator !=(const Block<T,DIM,Tag,CS>& b1, const Block<T,DIM,Tag,CS>& b2) {
    return ( b1.start() != b2.start() ) || ( b1.finish() != b2.finish() );
  }

  /// ostream output orperator.
  template<typename T, unsigned int DIM, typename Tag, typename CS>
  std::ostream& operator<<(std::ostream& out, const Block<T,DIM,Tag,CS>& blk) {
    out << "[ " << blk.start() << " , " << blk.finish() << " )";
    return out;
  }
} // namespace TiledArray
#endif // BLOCK_H__INCLUDED
