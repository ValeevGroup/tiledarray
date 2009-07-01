#ifndef RANGE_H__INCLUDED
#define RANGE_H__INCLUDED

#include <coordinates.h>
#include <range1.h>
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
  class Range1;

  // need these forward declarations
  template<typename I, unsigned int DIM, typename CS> class TiledRange;
  template<typename I, unsigned int DIM, typename CS>
  TiledRange<I,DIM,CS> operator ^(const Permutation<DIM>&, const TiledRange<I,DIM,CS>&);
  template<typename I, unsigned int DIM, typename CS>
  std::ostream& operator<<(std::ostream& out, const TiledRange<I,DIM,CS>& rng);

  /// TiledRange is a tiled DIM-dimensional range. It is immutable, to simplify API.
  template<typename I, unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class TiledRange : boost::equality_comparable1< TiledRange<I,DIM,CS> > {
	public:
      // typedefs
      typedef TiledRange<I,DIM,CS> Range_;
      typedef CS coordinate_system;
      typedef Range1<I> range1_type;
      typedef Block<I,DIM,LevelTag<1>,coordinate_system> block_type;
      typedef Block<I,DIM,LevelTag<0>,coordinate_system> element_block_type;
      typedef element_block_type tile_block_type;
      typedef typename block_type::size_array size_array;
      typedef typename block_type::index_type index_type;
      typedef typename tile_block_type::index_type tile_index_type;
      typedef DenseArrayStorage<tile_block_type, DIM, LevelTag<1>, coordinate_system > tile_container;
      typedef typename tile_container::iterator iterator;
      typedef typename tile_container::const_iterator const_iterator;

      /// Returns the number of dimensions
      static unsigned int dim() { return DIM; }

      // Default constructor
      TiledRange() : block_(), element_block_(), tile_blocks_(), ranges_() { }

      // Constructed with a set of ranges pointed to by [ first, last ).
      template <typename InIter>
      TiledRange(InIter first, InIter last) {
        for(typename Ranges::iterator it = ranges_.begin(); it != ranges_.end(); ++it) {
          TA_ASSERT( (first != last),
              std::runtime_error("Range<...>::Range(...): iterator unexpectedly reached the end of the range.") );
          *it = *first;
        }
        init_();
      }

      TiledRange(const TiledRange& other) :
          block_(other.block_), element_block_(other.element_block_),
          tile_blocks_(other.tile_blocks_), ranges_(other.ranges_)
      { }

      /// Return iterator to the tile that contains an element index.
      const_iterator find(const tile_index_type e) const {
        if(element_block_.includes(e)) {
          index_type t = element2tile(e);
          const_iterator result = tile_blocks_.find(t);
          return result;
        } else {
          const_iterator result = tile_blocks_.end();
          return result;
        }
      }

      /// In place permutation of range.

      /// This function will permute the range. Note: only tiles that are not
      /// being used by other objects will be permuted. The owner of those
      /// objects are
      TiledRange& operator ^=(const Permutation<DIM>& perm) {
        TiledRange temp(*this);
        temp.ranges_ ^= perm;
        temp.block_ ^= perm;
        temp.element_block_ ^= perm;
        iterator temp_it = temp.tiles().begin();
        for(iterator it = tiles().begin(); it != tiles().end(); ++it, ++temp_it) {
          if(it->unique())
            (* temp.tile_blocks_[ *it ]) ^= perm;
          else
            *temp_it = *it; // someone else is using this pointer, so we want to save its link.
        }
        temp.tile_blocks_ ^= perm;
        return *this;
      }

      /// TiledRange assignment operator
      TiledRange& operator =(const TiledRange& other) {
        TiledRange temp(other);
        swap(temp);
        return *this;
      }

      /// Resize the range to the set of dimensions in [first, last) input
      /// iterators.
      template <typename InIter>
      TiledRange& resize(InIter first, InIter last) {
        TiledRange temp(first, last);
        swap(temp);
        return *this;
      }

      /// Equality operator
      bool operator ==(const TiledRange& rng) const {
        return std::equal(ranges_.begin(), ranges_.end(), rng.ranges_.begin());
      }

      /// Access the block information on the tiles contained by the range.
      const block_type& tiles() const {
        return block_;
      }

      /// Access the block information on the elements contained by the range.
      const element_block_type& elements() const {
        return element_block_;
      }

      /// Access the block information on the elements contained by tile t.
      const tile_block_type& tile(const index_type& t) const {
        return tile_blocks_[t];
      }

      void swap(TiledRange& other) {
        block_.swap(other.block_);
        element_block_.swap(other.element_block_);
        std::swap(tile_blocks_, other.tile_blocks_);
        std::swap(ranges_, other.ranges_);
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
        boost::shared_ptr<tile_block_type> tile_block;
        tile_index_type start_tile;
        tile_index_type finish_tile;
        tile_blocks_.resize(block_.size());
        for(typename block_type::const_iterator it = block_.begin(); it != block_.end(); ++it) {
          // Determine the start and finish of each tile.
          for(unsigned int d = 0; d < DIM; ++d) {
            start_tile[d] = ranges_[d].tile( (*it)[d] ).start();
            finish_tile[d] = ranges_[d].tile( (*it)[d] ).finish();
          }

          // Create and store the tile block.
          tile_blocks_[ *it ] = tile_block_type(start_tile, finish_tile);
          tile_block.reset();
        }
      }

      index_type element2tile(const tile_index_type& e) const {
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

      /// Stores information on tile indexing for the range.
      block_type block_;
      /// Stores information on element indexing for the range.
      element_block_type element_block_;
      /// Stores a indexing information for each tile in the range.
      tile_container tile_blocks_;
      /// Stores tile boundaries for each dimension.
      typedef boost::array<range1_type,DIM> Ranges;
      Ranges ranges_;

  };

  /// TiledRange permutation operator.

  /// This function will permute the range. Note: only tiles that are not
  /// being used by other objects will be permuted. The owner of those
  /// objects are
  template<typename I, unsigned int DIM, typename CS>
  TiledRange<I,DIM,CS> operator ^(const Permutation<DIM>& perm, const TiledRange<I,DIM,CS>& r) {
    TiledRange<I,DIM,CS> result(r);
    result.ranges_ ^= perm;
    result.block_ ^= perm;
    result.element_block_ ^= perm;
    result.tile_blocks_ ^= perm;
    for(typename TiledRange<I,DIM,CS>::iterator it = r.tiles().begin(); it != r.tiles().end(); ++it)
      result.tile_blocks_[ *it ] ^= perm;

    return result;
  }

  template<typename I, unsigned int DIM, typename CS>
  std::ostream& operator<<(std::ostream& out, const TiledRange<I,DIM,CS>& rng) {
    out << "Range<" << DIM << ">(" << " @= " << &rng
        << " *begin_tile=" << (rng.tiles().start()) << " *end_tile=" << (rng.tiles().finish())
        << " start_element=" << rng.elements().start() << " finish_element=" << rng.elements().finish()
        << " nelements=" << rng.elements().volume() << " ntiles=" << rng.tiles().volume() << " )";
    return out;
  }


}
; // end of namespace TiledArray


#endif // RANGE_H__INCLUDED
