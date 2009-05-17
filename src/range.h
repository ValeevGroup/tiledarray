	#ifndef RANGE_H__INCLUDED
#define RANGE_H__INCLUDED

#include <iostream>

#include <boost/array.hpp>
#include <boost/operators.hpp>
#include <boost/iterator/filter_iterator.hpp>

#include <range1.h>
#include <coordinates.h>
#include <iterator.h>
#include <permutation.h>

namespace TiledArray {

  // need these forward declarations
  template<unsigned int DIM, typename CS> class Range;
  template<unsigned int DIM, typename CS> std::ostream& operator<<(std::ostream& out,
                                                                   const Range<DIM,CS>& rng);


  /// Range is a tiled DIM-dimensional range. It is immutable, to simplify API.
  template<unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class Range : boost::equality_comparable1< Range<DIM,CS> > {

      typedef Range<DIM,CS> Range_;
      typedef boost::array<Range1,DIM> Ranges;
      typedef Range1::element_index index_t;

    public:
      // typedefs
      typedef CS coordinate_system;
      typedef ArrayCoordinate<index_t,DIM,LevelTag<1>,coordinate_system> tile_index;
      typedef ArrayCoordinate<index_t,DIM,LevelTag<0>,coordinate_system> element_index;
      typedef size_t ordinal_index;

      // ready to declare iterators
      /// iterates over Range1
      typedef typename Ranges::const_iterator range_iterator;
      /// iterates over tile indices
      typedef detail::IndexIterator< tile_index, Range_> tile_iterator;
      INDEX_ITERATOR_FRIENDSHIP(tile_index, Range_);

      /// Returns the number of dimensions
      static unsigned int dim() { return DIM; }

      // Default constructor
      Range() {
        init_();
      }

      // Constructed with an array of ranges
      Range(const Range1* ranges) {
        std::copy(ranges, ranges + DIM, ranges_.begin());
        init_();
      }

      /// Constructor from an iterator range of Range1
      template <typename RangeIterator>
      Range(const RangeIterator& ranges_begin, const RangeIterator& ranges_end) {
        std::copy(ranges_begin, ranges_end, ranges_.begin());
        init_();
      }

      /// Returns an iterator pointing to the first range.
      tile_iterator begin() const {
        tile_iterator result(start_tile(), this);
        return result;
      }

      /// Return an iterator pointing one past the last dimension.
      tile_iterator end() const {
        return tile_iterator(finish_tile(), this);
      }

      /// number of elements
      ordinal_index nelements() const {
        return nelems_;
      }

      /// number of elements of tile
      ordinal_index nelements(const tile_index& t) const {
    	  ordinal_index n = 1;
    	  for(size_t d = 0; d < DIM; ++d)
            n *= ranges_[d].size(t[d]);

    	  return n;
      }

      /// number of tiles
      ordinal_index ntiles() const {
        return ntiles_;
      }

      /// Returns true if element_index is within the range
      bool includes(const element_index& e) const {
        for(unsigned int d = 0; d < DIM; ++d)
          if ( !ranges_[d].includes_element(e[d]) )
            return false;
        return true;
      }

      /// Returns true if tile_index is within the range
      bool includes(const tile_index& t) const {
        for(unsigned int d=0; d<DIM; ++d)
          if ( !ranges_[d].includes_tile(t[d]) )
            return false;
        return true;
      }

      /// Return the tile that contains an element index.
      tile_iterator find(const element_index e) const {
        tile_index tmp;

        for (unsigned int dim = 0; dim < DIM; ++dim)
          tmp[dim] = *( ranges_[dim].find(e[dim]) );

        if (includes(tmp)) {
          tile_iterator result(tmp, this);
          return result;
        }
        else
          return end();
      }

      Range<DIM>& operator ^=(const Permutation<DIM>& perm) {
        ranges_ ^= perm;
        init_();
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
      element_index start_element() const {
        return start_element_;
      }

      /// return element past the one with the largest indices in each dimension
      element_index finish_element() const {
        return finish_element_;
      }

      /// return element with the smallest indices in each dimension
      element_index start_element(const tile_index& t) const {
        assert(includes(t));
        element_index result;
    	for(size_t d = 0; d < DIM; ++d)
          result[d] = ranges_[d].start_element(t[d]);
        return result;
      }

      /// return element past the one with the largest indices in each dimension
      element_index finish_element(const tile_index& t) const {
        assert(includes(t));
    	element_index result;
        for(size_t d = 0; d < DIM; ++d)
          result[d] = ranges_[d].finish_element(t[d]);
        return result;
      }

      /// return tile with the smallest indices in each dimension
      tile_index start_tile() const {
        return start_tile_;
      }

      /// return tile past the one with the largest indices in each dimension
      tile_index finish_tile() const {
        return finish_tile_;
      }


      /// given a tile index, return a tile_index::Array (boost::array) object containing its extents in each dimension
      typename tile_index::Array size(const tile_index& t) const {
        element_index extents = finish_element(t) - start_element(t);
        return extents.data();
      }

      /// return a tile_index::Array (boost::array) object containing its extents in each dimension
      typename tile_index::Array size() const {
        element_index extents = finish_element() - start_element();
        return extents.data();
      }

      /// computes an ordinal index for a given tile_index
      ordinal_index ordinal(const tile_index& t) const {
        assert(includes(t));
        tile_index relative_index = t - start_tile();
        ordinal_index result = dot_product(relative_index.data(),tile_ordinal_weights_);
        return result;
      }

    private:
      /// precomputes useful data listed below
      void init_() {
        // Get dim ordering iterator
        const detail::DimensionOrder<DIM>& dimorder = coordinate_system::ordering();
        typename detail::DimensionOrder<DIM>::const_iterator d;

        ordinal_index tile_weight = 1;
        ordinal_index element_weight = 1;
        for(d = dimorder.begin(); d != dimorder.end(); ++d) {
          // Init ordinal weights for tiles.
          tile_ordinal_weights_[*d] = tile_weight;
          tile_weight *= ranges_[*d].ntiles();

          // init ordinal weights for elements.
          element_ordinal_weights_[*d] = element_weight;
          element_weight *= ranges_[*d].size();
        }
        nelems_ = element_weight;
        ntiles_ = tile_weight;

        // init start & finish for elements and tiles
        for(unsigned int d=0; d<DIM; ++d) {
          // Store upper and lower element ranges
          start_element_[d] = ranges_[d].start_element();
          finish_element_[d] = ranges_[d].finish_element();
          // Store upper and lower tile ranges
          start_tile_[d] = ranges_[d].start_tile();
          finish_tile_[d] = ranges_[d].finish_tile();
        }
      }

      /// computes an ordinal index for a given tile_index
      ordinal_index ordinal(const element_index& e) const {
    	assert(includes(e));
    	element_index relative_index = e - start_element();
        ordinal_index result = dot_product(relative_index.data(),element_ordinal_weights_);
        return result;
      }

      /// Increment tile index.
      void increment(tile_index& t) const {
        detail::IncrementCoordinate<DIM,tile_index,coordinate_system>(t, start_tile(), finish_tile());
      }

      /// Increment element index.
      void increment(element_index& e) const {
        detail::IncrementCoordinate<DIM,element_index,coordinate_system>(e, start_element(), finish_element());
      }

      /// to compute ordinal dot tile_index with this.
      /// this automatically takes care of dimension ordering
      boost::array<ordinal_index,DIM> tile_ordinal_weights_;
      boost::array<ordinal_index,DIM> element_ordinal_weights_;
      ordinal_index ntiles_;
      ordinal_index nelems_;
      element_index start_element_;
      element_index finish_element_;
      tile_index start_tile_;
      tile_index finish_tile_;
      Ranges ranges_; // Vector of range data for each dimension

  };

  template<unsigned int DIM, typename CS>
  std::ostream& operator<<(std::ostream& out, const Range<DIM,CS>& rng) {
    out << "Range<" << DIM << ">(" << " @= " << &rng
        << " *begin_tile=" << (*rng.begin()) << " *end_tile=" << (*rng.end())
        << " start_element=" << rng.start_element() << " finish_element=" << rng.finish_element()
        << " nelements=" << rng.nelements() << " ntiles=" << rng.ntiles() << " )";
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
