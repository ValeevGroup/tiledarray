#ifndef RANGE1_H__INCLUDED
#define RANGE1_H__INCLUDED

#include <vector>
#include <algorithm>
#include <cstddef>
#include <cassert>
#include <iostream>

#include <boost/operators.hpp>
#include <boost/limits.hpp>

#include <iterator.h>

namespace TiledArray {
  
  /** Range1 class defines a nonuniformly-tiled continuous one-dimensional range.
   The tiling data is constructed with and stored in an array with
   the format {a, b, c, ...}, where 0 <= a < b < c < ... Each tile is
   defined by [a,b), [b,c), ... The number of tiles in the range is
   defined as one less than the number of elements in the array.
   */
  class Range1 : boost::equality_comparable1<Range1> {
      typedef Range1 my_type;
    public:
      typedef size_t element_index;
      typedef size_t tile_index;

      /// Tile is an interval [start,finish)
      class Tile :
      boost::equality_comparable1<Tile>
      {
          typedef Range1::element_index element_index;
          typedef Range1::tile_index tile_index;
          typedef Tile my_type;
          
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
          
          void throw_if_invalid() {
            if (finish() < start())
              abort();
          }
          
        public:
          Tile() :
            start_(std::numeric_limits<element_index>::min()), finish_(std::numeric_limits<element_index>::min()) {
          }

          /// Tile is an interval [start,finish)
          Tile(element_index start, element_index finish) :
            start_(start), finish_(finish) {
            throw_if_invalid();
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

          element_index size() const {
            return finish_ - start_;
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
            ++i;
          }
      };

    private:
      /////////////
      // Range1 data
      /////////////
      typedef std::vector<Tile> Tiles;
      /// set of tiles. This is the primary data, i.e. other data is always recomputed from this, not modified directly
      Tiles tiles_;
      /// maps element index to tile index (secondary data)
      std::vector<tile_index> elem2tile_;

      /// Validates tile_boundaries
      static bool valid_(const std::vector<element_index>& tile_boundaries) {
        // Verify the requirement that a < b < c < ...
        for (size_t index = 1; index < tile_boundaries.size(); ++index)
          if (tile_boundaries[index - 1] >= tile_boundaries[index])
            return false;
        return true;
      }

      /// Initialize tiles use a set of tile offsets
      void init_tiles_(const std::vector<element_index>& tile_boundaries) {
        assert(valid_(tile_boundaries));
        if (tile_boundaries.size() >=2) {
          for (size_t i=1; i<tile_boundaries.size(); ++i) {
            tiles_.push_back(Tile(tile_boundaries[i-1], tile_boundaries[i]));
          }
        }
      }
      
      /// Initialize secondary data
      void init_map_() {
        // init elem2tile map
        elem2tile_.resize(size());
        size_t ii = 0;
        for (tile_index t=0; t < tiles_.size(); ++t)
          for (element_index i=0; i < tiles_[t].size(); ++i, ++ii)
            elem2tile_[ii] = t;
      }

      /// Returns the low element index of the range.
      element_index start() const
      {
        if (!empty())
          return *(tiles_.begin()->begin());
        else
          return std::numeric_limits<element_index>::min();
      }

      /// Returns the high element index of the range.
      element_index finish() const
      {
        if (!empty())
          return *(tiles_.rbegin()->end());
        else
          return std::numeric_limits<element_index>::min();
      }

      /// Returns the low tile index of the range.
      tile_index start_tile() const
      {
        if (!empty())
          return 0;
        else
          return std::numeric_limits<tile_index>::min();
      }

      /// Returns the high tile index of the range.
      tile_index finish_tile() const
      {
        if (!empty())
          return tiles_.size();
        else
          return std::numeric_limits<tile_index>::min();
      }
      
      friend std::ostream& operator <<(std::ostream& out, const Range1& rng);

    public:
      /// Default constructor, range of 1 tile and element.
      Range1() : tiles_() {
        init_map_();
      }
      
      /// Constructs range from a vector
      Range1(std::vector<element_index> tile_boundaries) {
        init_tiles_(tile_boundaries);
        init_map_();
      }
      
      /**
       * Construct range from a C-style array, ranges must have at least
       * ntiles + 1 elements.
       */
      Range1(const element_index* tile_boundaries, tile_index ntiles) {
        init_tiles_(std::vector<element_index>(tile_boundaries, tile_boundaries+ntiles+1));
        init_map_();
      }
      
      /// Copy constructor
      Range1(const Range1& rng) :
        tiles_(rng.tiles_), elem2tile_(rng.elem2tile_) {
      }
      
      /// Assignment operator
      Range1& operator =(const Range1& rng) {
        tiles_ = rng.tiles_;
        elem2tile_ = rng.elem2tile_;
        return *this;
      }
      
      /// Equality operator
      bool operator ==(const Range1& rng) const {
        return tiles_ == rng.tiles_;
      }
      
      /// iterates over tile indices
      typedef detail::IndexIterator<tile_index,my_type> tile_iterator;
      /// Returns an iterator to the first tile in the range.
      tile_iterator begin_tile() const
      {
        tile_iterator result(0,*this);
        return result;
      }
      /// Returns an iterator to the end of the range.
      tile_iterator end_tile() const
      {
        tile_iterator result(ntiles(),*this);
        return result;
      }
      /// how tile_index is incremented
      void increment(tile_index& t) const {
        ++t;
      }
      
      /// Return tile index associated with element_index
      tile_iterator find(element_index element) {
        element_index relindex = element - start();
        if (relindex >= size())
          return end_tile();
#ifdef NDEBUG
        tile_iterator result(elem2tile_[relindex],*this);
#else
        tile_iterator result(elem2tile_.at(relindex),*this);
#endif
        return result;
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
      /// since element_index and tile_index are the same type, they are incremented by the same function
      //void increment(element_index& i) const { ++i; }

      
      /// Return Tile corresponding to tile index t
      const Tile& tile(tile_index t) const {
#ifdef NDEBUG
        return tiles_[t];
#else
        return tiles_.at(t);
#endif
      }

      /// Returns the number of tiles in the range.
      size_t ntiles() const {
        return tiles_.size();
      }

#if 0
      /// Returns a pair containing high and low tile range.
      std::pair<tile_index, tile_index> tile_range() const {
        return std::pair<tile_index,tile_index>(0, ntiles());
      }
#endif
      
      bool
      empty() const {
        return tiles_.empty();
      }

      /// Returns the number of elements in the range.
      size_t size() const
      {
        return finish() - start();
      }

      /// contains this element?
      bool
      includes(const element_index& a) const {
        return a < finish() && a >= start();
      }

      /// contains this tile?
      bool
      includes_tile(const tile_index& a) const {
        return a < finish_tile() && a >= start_tile();
      }

    }; // end of Range1

    inline std::ostream& operator <<(std::ostream& out, const Range1& rng) {
      out << "Range1(" << "[" << rng.start() << "," << rng.finish() << ") "
      << " size=" << rng.size() << " ntiles= " << rng.ntiles() << " )" << std::endl;
      return out;
    }

}; // end of namespace TiledArray

#endif // RANGE1_H__INCLUDED
