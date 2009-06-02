	#ifndef RANGE_H__INCLUDED
#define RANGE_H__INCLUDED

#include <range1.h>
#include <coordinates.h>
#include <iterator.h>
#include <block.h>
#include <array_storage.h>
#include <debug.h>
#include <iosfwd>
#include <boost/array.hpp>
#include <boost/operators.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/make_shared.hpp>

namespace TiledArray {

  // Forward declaration of TiledArray Permutation.
  template <unsigned int DIM>
  class Permutation;

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
      typedef Block<ordinal_index,DIM,LevelTag<0>,coordinate_system> tile_block_type;
      typedef boost::shared_ptr<block_type> block_ptr;
      typedef boost::shared_ptr<tile_block_type> tile_block_ptr;
      typedef typename block_type::size_array size_array;
      typedef typename block_type::index_type index_type;
      typedef typename tile_block_type::index_type tile_index_type;
      typedef typename block_type::const_iterator const_iterator;
      typedef typename tile_block_type::const_iterator const_tile_iterator;
      typedef DenseArrayStorage<tile_block_ptr, DIM, LevelTag<1>, coordinate_system > tile_container;

      /// Returns the number of dimensions
      static unsigned int dim() { return DIM; }

      // Default constructor
      Range() : block_(), element_block_(), tile_blocks_(), ranges_() { }

      // Constructed with an array of ranges
      template <typename InIter>
      Range(InIter first, InIter last) {
        TA_ASSERT( (last - first) == DIM);
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

      /// Returns an index iterator that iterates over tile indices.
      const_iterator begin() const { return block_.begin(); }
      const_iterator end() const { return block_.end(); }

      /// Returns an index iterator that iterates over tile blocks.
      const_tile_iterator begin(const index_type& t) const { return tile_blocks_[t].begin(); }
      const_tile_iterator end(const index_type& t) const { return tile_blocks_[t].end(); }

      /// number of elements
      ordinal_index volume_elements() const {
        return detail::volume(element_block_.size());
      }

      /// number of elements of tile
      ordinal_index volume_elements(const index_type& t) const {
    	  return detail::volume(tile_blocks_[t].size());
      }

      /// number of tiles
      ordinal_index volume_tiles() const {
        return detail::volume(block_.size());
      }

      /// Returns true if tile_index_type is within the range
      bool includes(const tile_index_type& e) const {

        return element_block_.includes(e);
      }

      /// Returns true if index_type is within the range
      bool includes(const index_type& t) const {
        return block_.includes(t);
      }

      /// Return the tile that contains an element index.
      const_iterator find(const tile_index_type e) const {
        index_type tmp;
        if(includes(tmp)) {
          for(unsigned int d = 0; d < DIM; ++d)
            tmp[d] = ranges_[d].find(e[d]).index;
        } else {
          tmp = block_.finish();
        }

        const_iterator result(tmp, &block_);
        return result;
      }

      Range<DIM>& operator ^=(const Permutation<DIM>& perm) {
        Range temp(*this);
        temp.ranges_ ^= perm;
        temp.block_ ^= perm;
        temp.element_block_ ^= perm;
        for(const_iterator it = begin(); it != end(); ++it) {

        }

        return *this;
      }

      // Equality operator
      bool operator ==(const Range<DIM>& rng) const {
        if(&rng == this)
          return true;
        else
          return std::equal(ranges_.begin(), ranges_.end(), rng.ranges_.begin());
      }

      /// return element with the smallest indices in each dimension
      tile_index_type start_element() const {
        return element_block_.start();
      }

      /// return element past the one with the largest indices in each dimension
      tile_index_type finish_element() const {
        return element_block_.finish();
      }

      /// return element with the smallest indices in each dimension
      tile_index_type start_element(const index_type& t) const {
        assert(includes(t));
        tile_index_type result = tile_blocks_[t]->start();
        return result;
      }

      /// return element past the one with the largest indices in each dimension
      tile_index_type finish_element(const index_type& t) const {
        assert(includes(t));
    	tile_index_type result = tile_blocks_[t]->finish();
        return result;
      }

      /// return tile with the smallest indices in each dimension
      index_type start_tile() const {
        return block_.start();
      }

      /// return tile past the one with the largest indices in each dimension
      index_type finish_tile() const {
        return block_.finish();
      }


      /// given a tile index, return a size_array (boost::array) object containing its extents in each dimension
      size_array size(const index_type& t) const {
        return tile_blocks_[t]->size();
      }

      /// return a index_type::Array (boost::array) object containing its extents in each dimension
      size_array size() const {
        return block_->size();
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
          start[d] = ranges_[d].start_tile();
          finish[d] = ranges_[d].finish_tile();

          start_element[d] = ranges_[d].start_element();
          finish_element[d] = ranges_[d].finish_element();
        }
        block_.resize(start, finish);
        element_block_.resize(start_element, finish_element);

        // Find the dimensions of each tile and store its block information.
        tile_block_ptr tile_block;
        tile_index_type start_tile;
        tile_index_type finish_tile;
        tile_blocks_.resize(block_.size());
        for(const_iterator it = block_.begin(); it != block_.end(); ++it) {
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

      /// Stores information on tile indexing for the range.
      block_type block_;
      /// Stores information on element indexing for the range.
      tile_block_type element_block_;
      /// Stores a indexing information for each tile in the range.
      tile_container tile_blocks_;
      /// Stores tile boundaries for each dimension.
      typedef boost::array<Range1,DIM> Ranges;
      Ranges ranges_;

  };

  template<unsigned int DIM, typename CS>
  std::ostream& operator<<(std::ostream& out, const Range<DIM,CS>& rng) {
    out << "Range<" << DIM << ">(" << " @= " << &rng
        << " *begin_tile=" << (*rng.begin()) << " *end_tile=" << (*rng.end())
        << " start_element=" << rng.start_element() << " finish_element=" << rng.finish_element()
        << " nelements=" << rng.volume_elements() << " ntiles=" << rng.volume_tiles() << " )";
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
