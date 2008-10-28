#ifndef RANGE_H__INCLUDED
#define RANGE_H__INCLUDED

#include <iostream>

#include <boost/array.hpp>
#include <boost/operators.hpp>
#include <boost/iterator/filter_iterator.hpp>

#include <range1.h>
#include <coordinates.h>
#include <iterator.h>

namespace TiledArray {

  // need these forward declarations
  template<unsigned int DIM, typename CS> class Range;
  template<unsigned int DIM, typename CS> std::ostream& operator<<(std::ostream& out,
                                                                   const Range<DIM,CS>& rng);

  
  /// Range is a tiled DIM-dimensional range. It is immutable, to simplify API.
  template<unsigned int DIM, typename CS = CoordinateSystem<DIM> > class Range :
      boost::equality_comparable1< Range<DIM,CS> > {
      
      typedef Range<DIM,CS> my_type;
      typedef boost::array<Range1,DIM> Ranges;
      typedef Range1::element_index index_t;
      struct ElementTag {
      };
      struct TileTag {
      };
      
    public:
      // typedefs
      typedef CS CoordinateSystem;
      typedef ArrayCoordinate<index_t,DIM,TileTag> tile_index;
      typedef ArrayCoordinate<index_t,DIM,ElementTag> element_index;
      typedef size_t ordinal_index;

#if 0
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
#endif
      
      /// A DIM-dimensional tile
      class Tile : boost::equality_comparable1<Tile> {
          typedef Tile my_type;
          typedef typename Range::element_index element_index;
          typedef typename Range::tile_index tile_index;
          /// first index
          element_index start_;
          /// past-last index, i.e. last + 1
          element_index finish_;

          element_index start() const {
            return start_;
          }
          
          element_index finish() const {
            return finish_;
          }
          
        public:
          Tile() {
          }
          
          Tile(element_index start, element_index finish) :
            start_(start), finish_(finish) {
          }
          
          template <typename Tile1ContainerIterator>
          Tile(const Tile1ContainerIterator& tbegin,
               const Tile1ContainerIterator& tend)
          {
            typename element_index::iterator start_iterator = start_.begin();
            typename element_index::iterator finish_iterator = finish_.begin();
            for(Tile1ContainerIterator d=tbegin; d!=tend; ++d, ++start_iterator, ++finish_iterator) {
              *start_iterator = * d->begin();
              *finish_iterator = * d->end();
            }
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
          
          typename element_index::volume size() const {
            element_index d = finish() - start();
            --d;
            typedef typename element_index::volume volume_type;
            volume_type vol = volume(d);
            return vol;
          }
          
          void print(std::ostream& out) const {
            out << "Range<" << DIM << ">::Tile(@= " << this;
            out << " start=" << start() << " finish=" << finish()
            << " size=" << size() << " )";
          }
          
          typedef detail::IndexIterator<element_index,my_type> element_iterator;
          element_iterator begin() const {
            element_iterator result(start(),*this);
            return result;
          }
          element_iterator end() const {
            element_iterator result(finish(),*this);
            return result;
          }
          void increment(element_index& i) const {
            // TODO implemented iterator over Range<D>::Tile
            abort();
          }
      };

      // ready to declare iterators
      /// iterates over Range1
      typedef typename Ranges::const_iterator range_iterator;
      /// iterates over tile indices
      typedef detail::IndexIterator< tile_index, my_type > tile_iterator;
      /// iterates over tile indices
      typedef detail::IndexIterator< element_index, my_type > element_iterator;

      // Default constructor
      Range() {
        init_();
      }
      
      // Constructed with an array of ranges
      Range(const Range1* ranges) {
        std::copy(ranges, ranges+DIM, ranges_.begin());
        init_();
      }
      
      /// Constructor from an iterator range of Range1
      template <typename RangeIterator> Range(const RangeIterator& ranges_begin,
                                              const RangeIterator& ranges_end) {
        std::copy(ranges_begin, ranges_end, ranges_.begin());
        init_();
      }
      
      /// Returns an iterator pointing to the first element
      element_iterator begin() const {
        element_iterator result(start_, *this);
        return result;
      }
      /// Return an iterator pointing to the one past the last element
      element_iterator end() const {
        element_iterator result(finish_, *this);
        return result;
      }
      
      /// Returns an iterator pointing to the first range.
      tile_iterator begin_tile() const {
        tile_iterator result(tile_index(), *this);
        return result;
      }
      
      /// Return an iterator pointing one past the last dimension.
      tile_iterator end_tile() const {
        tile_index _end;
        _end[DIM-1] = ranges_[DIM-1].ntiles();
        tile_iterator result(_end, *this);
        return result;
      }
      
      void increment(tile_index& t) const {
        // increment least significant -- see ArrayCoordinate
        // check if still contained, carry over if necessary
        const typename detail::DimensionOrder<DIM>::const_iterator end_iter = CoordinateSystem::ordering().end_order();
        typename detail::DimensionOrder<DIM>::const_iterator order_iter = CoordinateSystem::ordering().begin_order();
        unsigned int lsdim = *order_iter;
        typename tile_index::index& least_significant = t[lsdim];
        ++least_significant;
        Range1::tile_index bound = * ranges_[lsdim].end();
        while (least_significant == bound && ++order_iter != end_iter) {
          least_significant = * ranges_[lsdim].begin();
          lsdim = *order_iter;
          least_significant = t[lsdim];
          ++least_significant;
          bound = * ranges_[lsdim].end();
        }
      }
      
      /// Returns an iterator pointing to the first range.
      range_iterator begin_range() const {
        return ranges_.begin();
      }
      
      /// Return an iterator pointing one past the last dimension.
      range_iterator end_range() const {
        return ranges_.end();
      }

      /// number of elements
      ordinal_index size() const {
        return nelems_;
      }
      /// number of tiles
      ordinal_index ntiles() const {
        return ntiles_;
      }
      
      /// Returns true if element_index is within the range
      bool includes(const element_index& e) const {
        for(unsigned int d=0; d<DIM; ++d)
          if ( !ranges_[d].includes(e[d]) )
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
      tile_iterator find(const element_index e) {
        tile_index tmp;
        
        for (unsigned int dim = 0; dim < DIM; ++dim)
          tmp[dim] = *(ranges_[dim].find(e[dim]));

        if (this->includes(tmp)) {
          tile_iterator result(tmp,*this);
          return result;
        }
        else
          return end_tile();
      }

      /// Returns Tile for the given tile_index t. Assumes that includes(t) is true.
      Tile tile(const tile_index& t) const {
        boost::array<Range1::Tile,DIM> tiles1;
        for(unsigned int d=0; d<DIM; ++d)
          tiles1[d] = ranges_[d].tile(t[d]);
        Tile result(tiles1.begin(),tiles1.end());
        return result;
      }

      Range<DIM>& permute(const Permutation<DIM>& perm) {
        operator^(perm, ranges_);
        init_();
        return *this;
      }
      
      // Equality operator
      bool operator ==(const Range<DIM>& ortho) const {
        if (&ortho == this)
          return true;
        else
          return std::equal(begin_range(), end_range(), ortho.begin_range());
      }
      
      /// computes an ordinal index for a given tile_index
      ordinal_index ordinal(const element_index& e) {
        ordinal_index result = dot_product(e.r(),element_ordinal_weights_);
        return result;
      }
      /// computes an ordinal index for a given tile_index
      ordinal_index ordinal(const tile_index& t) {
        ordinal_index result = dot_product(t.r(),tile_ordinal_weights_);
        return result;
      }
    
      private:
        Ranges ranges_; // Vector of range data for each dimension

        /// precomputes useful data listed below
        void init_() {
          // init ordinal weights for tiles
          const detail::DimensionOrder<DIM>& dimorder = CoordinateSystem::ordering();
          typedef typename detail::DimensionOrder<DIM>::const_iterator citer;
          {
            ordinal_index tile_weight = 1;
            for(citer d=dimorder.begin_order(); d!=dimorder.end_order(); ++d) {
              const unsigned int dim = *d;
              tile_ordinal_weights_[dim] = tile_weight;
              tile_weight *= ranges_[dim].ntiles();
            }
            ntiles_ = tile_weight;
          }
          {
            ordinal_index element_weight = 1;
            for(citer d=dimorder.begin_order(); d!=dimorder.end_order(); ++d) {
              const unsigned int dim = *d;
              element_ordinal_weights_[dim] = element_weight;
              element_weight *= ranges_[dim].size();
            }
            nelems_ = element_weight;
          }
          {
            // first element is easy ...
            for(unsigned int d=0; d<DIM; ++d)
              start_[d] = * ranges_[d].begin();
            // last element is tricky: 
            for(unsigned int d=0; d<DIM; ++d)
              finish_[d] = * ranges_[d].end();
            ++result;
          }
        }

        /// to compute ordinal dot tile_index with this.
        /// this automatically takes care of dimension ordering
        boost::array<ordinal_index,DIM> tile_ordinal_weights_;
        boost::array<ordinal_index,DIM> element_ordinal_weights_;
        ordinal_index ntiles_;
        ordinal_index nelems_;
        element_index start_;
        element_index finish_;
        tile_index tile_finish_;

        friend std::ostream& operator << <>(std::ostream&, const Range& rng);

#if 1
      /// return element with the smallest indices in each dimension
      element_index start() const {
        element_index result;
        for(unsigned int d=0; d<DIM; ++d)
          result[d] = ranges_[d].start();
        return result;
      }
      /// return element past the one with the largest indices in each dimension
      element_index finish() const {
        element_index result;
        for(unsigned int d=0; d<DIM; ++d)
          result[d] = ranges_[d].finish()-1;
        ++result;
        return result;        
      }
      
      /// return tile with the smallest indices in each dimension
      tile_index start_tile() const {
        tile_index result;
        for(unsigned int d=0; d<DIM; ++d)
          result[d] = ranges_[d].start_tile();
        return result;
      }
      /// return tile past the one with the largest indices in each dimension
      tile_index finish_tile() const {
        tile_index result;
        for(unsigned int d=0; d<DIM; ++d)
          result[d] = ranges_[d].finish_tile()-1;
        ++result;
        return result;
      }
#endif
      
  };
  
  template<unsigned int DIM, typename CS> std::ostream& operator<<(std::ostream& out,
                                                                   const Range<DIM,CS>& rng) {
    out << "Range<" << DIM << ">(" << " @= " << &rng
        << " *begin=" << (*rng.begin()) << " *end=" << (*rng.end())
        << " *begin_tile=" << (*rng.begin_tile()) << " *end_tile=" << (*rng.end_tile())
        << " size=" << rng.size() << " ntiles=" << rng.ntiles() << " )";
    return out;
  }

  template<unsigned int DIM, typename CS> std::ostream& print(std::ostream& out,
                                                              const typename Range<DIM>::Tile& tile) {
    tile.print(out);
    return out;
  }

}
; // end of namespace TiledArray


#endif // RANGE_H__INCLUDED
