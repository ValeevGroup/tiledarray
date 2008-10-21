#ifndef RANGE_H__INCLUDED
#define RANGE_H__INCLUDED

#include <iostream>

#include <boost/array.hpp>
#include <boost/operators.hpp>
#include <boost/iterator/filter_iterator.hpp>

#include <range1.h>
#include <coordinates.h>

namespace TiledArray {
  
  // Range is a tiled DIM-dimensional range
  template<unsigned int DIM> class Range :
      boost::equality_comparable1< Range<DIM> > {
      
      typedef Range1::element_index index_t;
      struct ElementTag {
      };
      struct TileTag {
      };

      typedef boost::array<Range1,DIM> Ranges;
      Ranges ranges_; // Vector of range data for each dimension

    public:
      // typedefs
      typedef ArrayCoordinate<index_t,DIM,ElementTag> tile_index;
      typedef ArrayCoordinate<index_t,DIM,TileTag> element_index;
      typedef size_t ordinal_index;

      /// used to implement Shape::iterator and Shape::const_iterator
      template <typename Value> class Iterator : public boost::iterator_facade<
      Iterator<Value>,
      Value,
      std::input_iterator_tag> {
        public:
          typedef Range<DIM> Container;

          Iterator(const Iterator& other) :
            container_(other.container_), current_(other.current_) {
          }
          ~Iterator() {
          }
          
        private:
          friend class boost::iterator_core_access;
          friend class Range<DIM>;
          Iterator(const Value& cur, const Container* container) :
            container_(container), current_(cur) {
          }
          
          bool equal(Iterator<Value> const& other) const {
            return current_ == other.current_;
          }
          
#if 0
          void increment() {
            // increment least significant -- see ArrayCoordinate
            ++current_;
            // if necessary, carry over
            const Tuple<DIM> low = container_->low();
            const Tuple<DIM> high = container_->high();
            while (lsindex >= high[lsdim]) {
              current_[lsdim] = low[lsdim];
              --lsdim;
              // if ran out of dimensions break out of the loop
              if (lsdim >= 0)
              lsindex = ++(current_[lsdim]);
              else
              break;
            }
          }
#endif
          
          Value& dereference() const {
            return const_cast<Value&>(current_);
          }
          
          Iterator();

          const Container* container_;
          Value current_;
      };

      /// A DIM-dimensional tile
      class Tile : boost::equality_comparable1<Tile> {
          typedef typename Range::element_index element_index;
          typedef typename Range::tile_index tile_index;
          /// first index
          element_index start_;
          /// past-last index, i.e. last + 1
          element_index finish_;

        public:
          Tile() {
          }
          
          Tile(element_index start, element_index finish) :
            start_(start), finish_(finish) {
          }
          
          Tile(const Tile& src) :
            start_(src.start_), finish_(src.finish_) {
          }
          
          const Tile& operator=(const Tile& src) {
            start_ = src.start_;
            finish_ = src.finish_;
            return *this;
          }
          
          bool operator==(const Tile& A) const {
            return start_ == A.start_ && finish_ == A.finish_;
          }
          
          element_index start() const {
            return start_;
          }
          
          element_index finish() const {
            return finish_;
          }
          
          typename element_index::volume size() const {
            element_index d = finish_ - start_;
            --d;
            typedef typename element_index::volume volume_type;
            volume_type vol = volume(d);
            return vol;
          }
      };

      // ready to declare iterators
      typedef typename Ranges::const_iterator range_iterator;
      typedef Iterator< tile_index> tile_iterator;
      typedef Iterator< element_index> iterator;

      // Default constructor
      Range() {
      }
      
      // Constructed with an array of ranges
      Range(const Range1* ranges) {
        std::copy(ranges, ranges+DIM, ranges_.begin());
      }
      
#if 0
      // Construct from a set arrays
      Range(const size_t** ranges, const size_t tiles[DIM]) :
      ranges_(DIM, Range1())
      {
        for(unsigned int dim = 0; dim < DIM; ++dim)
        {
          assert(ranges[dim]);
          ranges_[dim] = Range1(ranges[dim], tiles[dim]);
        }
      }
#endif
      
      /// Constructor from an iterator range of Range1
      template <typename RangeIterator> Range(
                                              const RangeIterator& ranges_begin,
                                              const RangeIterator& ranges_end) {
        std::copy(ranges_begin, ranges_end, ranges_.begin());
      }
      
      /// Returns an iterator pointing to the first range.
      iterator begin() const {
        return element_iterator(tile_index(), this);
      }
      
      /// Return an iterator pointing one past the last dimension.
      iterator end() const {
        tile_index _end;
        _end[DIM-1] = ranges_[DIM-1].ntiles();
        return tile_iterator(_end, this);
      }
      
      /// Returns an iterator pointing to the first range.
      tile_iterator begin_tile() const {
        return tile_iterator(tile_index(), this);
      }
      
      /// Return an iterator pointing one past the last dimension.
      tile_iterator end_tile() const {
        tile_index _end;
        _end[DIM-1] = ranges_[DIM-1].ntiles();
        return tile_iterator(_end, this);
      }
      
      /// Returns an iterator pointing to the first range.
      range_iterator begin_range() const {
        return ranges_.begin();
      }
      
      /// Return an iterator pointing one past the last dimension.
      range_iterator end_range() const {
        return ranges_.end();
      }
      
      /// return element with the smallest indices in each dimension
      element_index start() const {
        
      }
      /// return element past the one with the largest indices in each dimension
      element_index finish() const {
        
      }
      
      /// return tile with the smallest indices in each dimension
      tile_index start_tile() const {
        
      }
      /// return tile past the one with the largest indices in each dimension
      tile_index finish_tile() const {
        
      }
      
      /// number of elements
      element_index size() const {
        
      }
      /// number of tiles
      element_index ntiles() const {
        
      }
      
      /// Returns true if element_index is within the range
      bool includes(const element_index& e) const {
        return (e >= start()) && (e < finish());
      }
      
      /// Return the tile that contains an element index.
      tile_index find(const element_index e) const {
        tile_index tmp;
        
        for (unsigned int dim = 0; dim < DIM; ++dim)
          tmp[dim] = ranges_[dim].find(e[dim]);
        
        return tmp;
      }
      
      Range<DIM>& permute(const Permutation<DIM>& perm) {
        operator^(perm, ranges_);
        return *this;
      }
      
      // Equality operator
      bool operator ==(const Range<DIM>& ortho) const {
        if (&ortho == this)
          return true;
        else
          return std::equal(begin_range(), end_range(), ortho.begin_range());
      }
      
      /// computes an ordinal index for a given element_index
      static ordinal_index ordinal(const element_index& t) {
        
      }
      /// computes an ordinal index for a given tile_index
      static ordinal_index ordinal(const tile_index& t) {
        
      }
      
  };
  
  template<unsigned int DIM> std::ostream& operator <<(std::ostream& out,
                                                       const Range<DIM>& rng) {
    out << "Range<" << DIM << ">(" << " @= " << &rng << " low= " << rng.low()
        << " high= " << rng.high() << " size= " << rng.size() << " range= [ "
        << rng.range().first << "," << rng.range().second << " )" << " tile_size= " << rng.tile_size() << " nelements= "
        << rng.nelements() << " ntiles= " << rng.ntiles() << " )";
    return out;
  }

}
; // end of namespace TiledArray


#endif // RANGE_H__INCLUDED
