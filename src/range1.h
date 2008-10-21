#ifndef RANGE1_H__INCLUDED
#define RANGE1_H__INCLUDED

#include <vector>
#include <algorithm>
#include <cstddef>
#include <cassert>
#include <iostream>

#include <boost/operators.hpp>
#include <boost/limits.hpp>

namespace TiledArray {
  
  /** Range1 class defines a nonuniformly-tiled one-dimensional range.
   The tiling data is constructed with and stored in an array with
   the format {a, b, c, ...}, where 0 <= a < b < c < ... Each tile is
   defined by [a,b), [b,c), ... The number of tiles in the range is
   defined as one less than the number of elements in the array.
   */
  class Range1 : boost::equality_comparable1<Range1> {
    public:
      typedef size_t element_index;
      typedef size_t tile_index;

      /// A tile is an interval [start,finish)
      class Tile :
      boost::equality_comparable1<Tile>
      {
          typedef Range1::element_index element_index;
          typedef Range1::tile_index tile_index;
          /// first index
          element_index start_;
          /// past-last index, i.e. last + 1
          element_index finish_;

        public:
          Tile() :
            start_(std::numeric_limits<element_index>::min()), finish_(std::numeric_limits<element_index>::min()) {
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

          element_index size() const {
            return finish_ - start_;
          }
      };

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
            
    public:
      typedef Tiles::const_iterator const_iterator;

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
      
      /// Return tile index associated with element_index
      tile_index find(element_index element) const {
#ifdef NDEBUG
        return elem2tile_[element - start()];
#else
        return elem2tile_.at(element - start());
#endif
      }
      
      /// Return tile t
      const Tile& tile(tile_index t) const {
#ifdef NDEBUG
        return tiles_[t];
#else
        return tiles_.at(t);
#endif
      }

#if 0   // not sure if need this
      /// contains this tile?
      bool
      contains(const Tile& t) {
        return a < ntiles() && a >= 0;
      }
#endif

      /// Returns an iterator to the first tile in the range.
      const_iterator
      begin_tile() const
      {
        return tiles_.begin();
      }

      /// Returns an iterator to the end of the range.
      const_iterator
      end_tile() const
      {
        return tiles_.end();
      }
      
      /// Returns the number of tiles in the range.
      tile_index ntiles() const {
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

      /// Returns the low element index of the range.
      element_index start() const
      {
        if (!empty())
          return tiles_.begin()->start();
        else
          return std::numeric_limits<element_index>::min();
      }

      /// Returns the high element index of the range.
      element_index finish() const
      {
        if (!empty())
          return tiles_.rbegin()->finish();
        else
          return std::numeric_limits<element_index>::min();
      }

      /// Returns the number of elements in the range.
      element_index size() const
      {
        return finish() - start();
      }

      /// contains this element?
      bool
      contains(const element_index& a) {
        return a < finish() && a >= start();
      }
      
    }; // end of Range1

    std::ostream& operator <<(std::ostream& out, const Range1& rng) {
      out << "Range1(" << "[" << rng.start() << "," << rng.finish() << ") "
      << " size=" << rng.size() << " ntiles= " << rng.ntiles() << " )" << std::endl;
      return out;
    }

}; // end of namespace TiledArray

#endif // RANGE1_H__INCLUDED
