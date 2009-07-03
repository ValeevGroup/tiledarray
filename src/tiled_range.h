#ifndef RANGE_H__INCLUDED
#define RANGE_H__INCLUDED

#include <coordinates.h>
#include <tiled_range1.h>
#include <array_storage.h>
#include <iosfwd>
#include <boost/array.hpp>
#include <boost/operators.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace TiledArray {

  // Forward declaration of TiledArray Permutation.
  template <unsigned int DIM>
  class Permutation;
  template <typename I>
  class TiledRange1;

  // need these forward declarations
  template<typename I, unsigned int DIM, typename CS> class TiledRange;
  template<typename I, unsigned int DIM, typename CS>
  TiledRange<I,DIM,CS> operator ^(const Permutation<DIM>&, const TiledRange<I,DIM,CS>&);
  template<typename I, unsigned int DIM, typename CS>
  bool operator ==(const TiledRange<I,DIM,CS>&, const TiledRange<I,DIM,CS>&);
  template<typename I, unsigned int DIM, typename CS>
  std::ostream& operator<<(std::ostream& out, const TiledRange<I,DIM,CS>& rng);

  /// TiledRange is a tiled DIM-dimensional range. It is immutable, to simplify API.
  template<typename I, unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class TiledRange : boost::equality_comparable1< TiledRange<I,DIM,CS> > {
	public:
      // typedefs
      typedef TiledRange<I,DIM,CS> TiledRange_;
      typedef CS coordinate_system;
      typedef TiledRange1<I> tiled_range1_type;
      typedef Range<I,DIM,LevelTag<1>,coordinate_system> range_type;
      typedef Range<I,DIM,LevelTag<0>,coordinate_system> element_range_type;
      typedef element_range_type tile_range_type;
      typedef typename range_type::size_array size_array;
      typedef typename range_type::index_type index_type;
      typedef typename range_type::volume_type volume_type;
      typedef typename tile_range_type::index_type element_index_type;
      typedef typename tile_range_type::index_type tile_index_type;
      typedef DenseArrayStorage<tile_range_type, DIM, LevelTag<1>, coordinate_system > tile_container;
      typedef typename tile_container::iterator iterator;
      typedef typename tile_container::const_iterator const_iterator;

      /// Returns the number of dimensions
      static unsigned int dim() { return DIM; }

      // Default constructor
      TiledRange() : block_(), element_block_(), tile_blocks_(), ranges_() { }

      // Constructed with a set of ranges pointed to by [ first, last ).
      template <typename InIter>
      TiledRange(InIter first, InIter last) {
        for(typename Ranges::iterator it = ranges_.begin(); it != ranges_.end(); ++it, ++first) {
          TA_ASSERT( (first != last),
              std::runtime_error("TiledRange<...>::TiledRange(...): iterator unexpectedly reached the end of the range.") );
          *it = *first;
        }
        init_();
      }

      /// Copy constructor
      TiledRange(const TiledRange& other) :
          block_(other.block_), element_block_(other.element_block_),
          tile_blocks_(other.tile_blocks_), ranges_(other.ranges_)
      { }

      /// Returns a const_iterator to the first tile range in TiledRange.
      const_iterator begin() const { return tile_blocks_.begin(); }
      /// Return a const_iterator to the end of the tile range.
      const_iterator end() const { return tile_blocks_.end(); }

      /// Return iterator to the tile range that contains the element index.
      const_iterator find(const tile_index_type e) const {
        const_iterator result;
        if(element_block_.includes(e)) {
          index_type t = element2tile_(e);
          result = tile_blocks_.begin() + tile_blocks_.ordinal(t);
        } else {
          result = tile_blocks_.end();
        }
        return result;
      }

      /// In place permutation of range.

      /// This function will permute the range. Note: only tiles that are not
      /// being used by other objects will be permuted. The owner of those
      /// objects are
      TiledRange& operator ^=(const Permutation<DIM>& p) {
        TiledRange temp = p ^ *this;
        swap(temp);
        return *this;
      }

      /// TiledRange assignment operator
      TiledRange& operator =(const TiledRange& other) {
        TiledRange temp(other);
        swap(temp);
        return *this;
      }

      /// Resize the range to the set of dimensions in [first, last) input
      /// iterators. The iterators must dereference to a tiled_range1_type.
      template <typename InIter>
      TiledRange& resize(InIter first, InIter last) {
        TiledRange temp(first, last);
        swap(temp);
        return *this;
      }

      /// Access the block information on the tiles contained by the range.
      const range_type& tiles() const {
        return block_;
      }

      /// Access the block information on the elements contained by the range.
      const element_range_type& elements() const {
        return element_block_;
      }

      /// Access the block information on the elements contained by tile t.
      const tile_range_type& tile(const index_type& t) const {
        TA_ASSERT( tile_blocks_.includes(t),
            std::out_of_range("TiledRange<...>::tile(...) const: Tile index is out of range."));
        return tile_blocks_[t - block_.start()];
      }

      void swap(TiledRange& other) {
        block_.swap(other.block_);
        element_block_.swap(other.element_block_);
        tile_blocks_.swap(other.tile_blocks_);
        boost::swap(ranges_, other.ranges_);
      }

    private:
      /// precomputes useful data listed below
      void init_() {
        // Indices used to store block start and finish.
        index_type start;
        index_type finish;
        tile_index_type start_element;
        tile_index_type finish_element;

        // Find the start and finish of the over all tiles and element blocks.
        for(unsigned int d=0; d < DIM; ++d) {
          start[d] = ranges_[d].tiles().start();
          finish[d] = ranges_[d].tiles().finish();

          start_element[d] = ranges_[d].elements().start();
          finish_element[d] = ranges_[d].elements().finish();
        }
        block_.resize(start, finish);
        element_block_.resize(start_element, finish_element);

        // Find the dimensions of each tile and store its block information.
        tile_index_type start_tile;
        tile_index_type finish_tile;
        tile_blocks_.resize(block_.size());
        for(typename range_type::const_iterator it = block_.begin(); it != block_.end(); ++it) {
          // Determine the start and finish of each tile.
          for(unsigned int d = 0; d < DIM; ++d) {
            start_tile[d] = ranges_[d].tile( (*it)[d] ).start();
            finish_tile[d] = ranges_[d].tile( (*it)[d] ).finish();
          }

          // Create and store the tile block.
          tile_blocks_[ *it - block_.start()] = tile_range_type(start_tile, finish_tile);
        }
      }

      /// Returns the tile index of the tile that contains the given element index.
      index_type element2tile_(const tile_index_type& e) const {
        index_type result;
        if(elements().includes(e)) {
          for(unsigned int d = 0; d < DIM; ++d) {
            result[d] = ranges_[d].element2tile(e[d]);
          }
        } else {
          result = block_.finish().data();
        }

        return result;
      }

      friend TiledRange operator ^ <>(const Permutation<DIM>&, const TiledRange<I,DIM,CS>&);
      friend bool operator == <>(const TiledRange&, const TiledRange&);

      range_type block_; ///< Stores information on tile indexing for the range.
      element_range_type element_block_; ///< Stores information on element indexing for the range.
      tile_container tile_blocks_; ///< Stores a indexing information for each tile in the range.
      typedef boost::array<tiled_range1_type,DIM> Ranges;
      Ranges ranges_; ///< Stores tile boundaries for each dimension.

  };

  /// TiledRange permutation operator.

  /// This function will permute the range. Note: only tiles that are not
  /// being used by other objects will be permuted. The owner of those
  /// objects are
  template<typename I, unsigned int DIM, typename CS>
  TiledRange<I,DIM,CS> operator ^(const Permutation<DIM>& p, const TiledRange<I,DIM,CS>& r) {
    TiledRange<I,DIM,CS> result(r);
    result.ranges_ ^= p;
    result.block_ ^= p;
    result.element_block_ ^= p;
    result.tile_blocks_ ^= p;
    for(typename TiledRange<I,DIM,CS>::range_type::const_iterator it = r.tiles().begin(); it != r.tiles().end(); ++it)
      result.tile_blocks_[ p ^ *it ] ^= p;

    return result;
  }

  /// Returns true when all tile and element ranges are the same.
  template<typename I, unsigned int DIM, typename CS>
  bool operator ==(const TiledRange<I, DIM, CS>& r1, const TiledRange<I, DIM, CS>& r2) {
    return
#ifndef NDEBUG
        // Do some extra equality checking while debugging. If everything is
        // working properly, the block data will be consistent with the data in
        // ranges.
        (r1.block_ == r2.block_) && (r1.element_block_ == r2.element_block_) &&
        std::equal(r1.tile_blocks_.begin(), r1.tile_blocks_.end(), r2.tile_blocks_.begin()) &&
#endif
        std::equal(r1.ranges_.begin(), r1.ranges_.end(), r2.ranges_.begin());
  }

  template<typename I, unsigned int DIM, typename CS>
  bool operator !=(const TiledRange<I, DIM, CS>& r1, const TiledRange<I, DIM, CS>& r2) {
    return ! operator ==(r1, r2);
  }

  template<typename I, unsigned int DIM, typename CS>
  std::ostream& operator<<(std::ostream& out, const TiledRange<I,DIM,CS>& rng) {
    out << "(" << " tiles = " << rng.tiles()
        << " elements = " << rng.elements() << " )";
    return out;
  }


}
; // end of namespace TiledArray


#endif // RANGE_H__INCLUDED
