#ifndef RANGE_H__INCLUDED
#define RANGE_H__INCLUDED

#include <coordinates.h>
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
  class Range1;

  // need these forward declarations
  template<unsigned int DIM, typename CS> class Range;
  template<unsigned int DIM, typename CS> std::ostream& operator<<(std::ostream& out,
                                                                   const Range<DIM,CS>& rng);

  /// Range is a tiled DIM-dimensional range. It is immutable, to simplify API.
  template<unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class Range : boost::equality_comparable1< Range<DIM,CS> > {
	public:
      // typedefs
      typedef Range<DIM,CS> Range_;
      typedef CS coordinate_system;
      typedef size_t ordinal_index;
      typedef Block<ordinal_index,DIM,LevelTag<1>,coordinate_system> block_type;
      typedef Block<ordinal_index,DIM,LevelTag<0>,coordinate_system> element_block_type;
      typedef element_block_type tile_block_type;
      typedef typename block_type::size_array size_array;
      typedef typename block_type::index_type index_type;
      typedef typename tile_block_type::index_type tile_index_type;
      typedef DenseArrayStorage<boost::shared_ptr<tile_block_type>, DIM, LevelTag<1>, coordinate_system > tile_container;
      typedef typename tile_container::iterator iterator;
      typedef typename tile_container::const_iterator const_iterator;

      /// Returns the number of dimensions
      static unsigned int dim() { return DIM; }

      // Default constructor
      Range() : block_(), element_block_(), tile_blocks_(), ranges_() { }

      // Constructed with an array of ranges
      template <typename InIter>
      Range(InIter first, InIter last) {
        assert( (last - first) == DIM);
        std::copy(first, last, ranges_.begin());
        init_();
      }

      Range(const Range& other) :
          block_(other.block_), element_block_(other.element_block),
          tile_blocks_(other.block_.size()), ranges_(other.ranges_)
      {
        // We need to do an explicit copy of the tile blocks so the data is copied
        // and not a copy of the shared pointers.
        typename tile_container::const_iterator other_it = other.tile_blocks_.begin();
        typename tile_container::const_iterator it = tile_blocks_.begin();
        for(; other_it != other.tile_blocks_.end_tile(); ++other_it)
          *it = boost::make_shared<tile_block_type>(*(*other_it));
      }

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

      Range<DIM>& operator ^=(const Permutation<DIM>& perm) {
        Range temp(*this);
        temp.ranges_ ^= perm;
        temp.block_ ^= perm;
        temp.element_block_ ^= perm;
        for(const_iterator it = tiles().begin(); it != tiles().end(); ++it) {
          tile_blocks_[ *it ] ^= perm;
        }

        return *this;
      }

      // Equality operator
      bool operator ==(const Range<DIM>& rng) const {
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
      boost::shared_ptr<tile_block_type> tile(const index_type& t) const {
        return tile_blocks_[t];
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
          start[d] = ranges_[d].start();
          finish[d] = ranges_[d].finish();

          start_element[d] = ranges_[d].start_element();
          finish_element[d] = ranges_[d].finish_element();
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
            start_tile[d] = ranges_[d].start_element( (*it)[d] );
            finish_tile[d] = ranges_[d].finish_element( (*it)[d] );
          }

          // Create and store the tile block.
          tile_block = boost::make_shared<tile_block_type>(start_tile, finish_tile);
          tile_blocks_[ *it ] = tile_block;
          tile_block.reset();
        }
      }

      index_type element2tile(const tile_index_type& e) const {
        index_type result;
        if(elements().includes(e)) {
          for(unsigned int d = 0; d < DIM; ++d)
            result[d] = ranges_[d].find(e[d])->index;
        } else {
          result = block_.finish();
        }

        return result;
      }

      /// Stores information on tile indexing for the range.
      block_type block_;
      /// Stores information on element indexing for the range.
      element_block_type element_block_;
      /// Stores a indexing information for each tile in the range.
      tile_container tile_blocks_;
      /// Stores tile boundaries for each dimension.
      typedef boost::array<Range1,DIM> Ranges;
      Ranges ranges_;

  };

  template<unsigned int DIM, typename CS>
  std::ostream& operator<<(std::ostream& out, const Range<DIM,CS>& rng) {
    out << "Range<" << DIM << ">(" << " @= " << &rng
        << " *begin_tile=" << (rng.tiles().start()) << " *end_tile=" << (rng.tiles().finish())
        << " start_element=" << rng.elements().start() << " finish_element=" << rng.elements().finish()
        << " nelements=" << rng.elements().volume() << " ntiles=" << rng.tiles().volume() << " )";
    return out;
  }

  template<unsigned int DIM, typename CS> std::ostream& print(std::ostream& out,
                                                              const typename Range<DIM>::tile& tile) {
    tile.print(out);
    return out;
  }

}
; // end of namespace TiledArray


#endif // RANGE_H__INCLUDED
