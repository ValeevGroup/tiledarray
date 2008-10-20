#ifndef RANGE_H__INCLUDED
#define RANGE_H__INCLUDED

#include <vector>
#include <algorithm>
#include <cstddef>
#include <cassert>
#include <iostream>

namespace TiledArray {
  
  /** Range class defines a nonuniformly-tiled one-dimensional range.
   The tiling data is constructed with and stored in an array with
   the format {a, b, c, ...}, where 0 <= a < b < c < ... Each tile is
   defined by [a,b), [b,c), ... The number of tiles in the range is
   defined as one less than the number of elements in the array.
   */
  class Range {
    public:
      typedef size_t element_index;
      typedef size_t tile_index;

      /// Represents a tile which is an interval [low,high)
      class RangeTile {
          typedef Range::element_index element_index;
          typedef Range::tile_index tile_index;
          /// first index
          element_index m_low;
          /// plast index, i.e. last + 1
          element_index m_high;
          /// size
          element_index m_size;

        public:
          RangeTile() :
            m_low(0), m_high(0), m_size(0) {
          }

          RangeTile(element_index first, element_index plast) :
            m_low(first), m_high(plast), m_size(plast-first) {
          }

          RangeTile(const RangeTile& src) :
            m_low(src.m_low), m_high(src.m_high), m_size(src.m_size) {
          }

          const RangeTile& operator=(const RangeTile& src) {
            m_low = src.m_low;
            m_high = src.m_high;
            m_size = src.m_size;
            return *this;
          }

          bool operator==(const RangeTile& A) const {
            return m_low == A.m_low && m_high == A.m_high;
          }

          element_index low() const {
            return m_low;
          }

          element_index high() const {
            return m_high;
          }

          element_index size() const {
            return m_size;
          }
      };

      typedef std::vector<RangeTile> RangeTiles;

      /////////////
      // Range data
      /////////////
      /// set of tiles. This is the primary data, i.e. other data is always recomputed from this, not modified directly
      RangeTiles m_tiles;
      /// maps element index to tile index (secondary data)
      std::vector<tile_index> m_elem2tile_map;

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
            m_tiles.push_back(RangeTile(tile_boundaries[i-1], tile_boundaries[i]));
          }
        }
      }
      
      /// Initialize secondary data
      void init_map_() {
        // init elem2tile map
        m_elem2tile_map.resize(size());
        size_t ii = 0;
        for (tile_index t=0; t < m_tiles.size(); ++t)
          for (element_index i=0; i < m_tiles[t].size(); ++i, ++ii)
            m_elem2tile_map[ii] = t;
      }
            
    public:
      typedef RangeTiles::const_iterator const_iterator;

      /// Default constructor, range of 1 tile and element.
      Range() {
    	std::vector<element_index> tile_boundaries(2, 0);
    	tile_boundaries[1] = 1;
    	init_tiles_(tile_boundaries);
        init_map_();
      }
      
      /// Constructs range from a vector
      Range(std::vector<element_index> tile_boundaries) {
        init_tiles_(tile_boundaries);
        init_map_();
      }
      
      /**
       * Construct range from a C-style array, ranges must have at least
       * ntiles + 1 elements.
       */
      Range(const element_index* tile_boundaries, tile_index ntiles) {
        init_tiles_(std::vector<element_index>(tile_boundaries, tile_boundaries+ntiles+1));
        init_map_();
      }
      
      /// Copy constructor
      Range(const Range& rng) :
        m_tiles(rng.m_tiles), m_elem2tile_map(rng.m_elem2tile_map) {
      }
      
      /// Assignment operator
      Range& operator =(const Range& rng) {
        m_tiles = rng.m_tiles;
        m_elem2tile_map = rng.m_elem2tile_map;
        return *this;
      }
      
      /// Equality operator
      bool operator ==(const Range& rng) const {
        return m_tiles == rng.m_tiles;
      }
      
      /// Inequality operator
      bool operator !=(const Range& rng) const {
        return !(*this == rng);
      }
      
      /// Return tile index associated with element_index
      tile_index tile(element_index element) const {
#ifdef NDEBUG
        return m_elem2tile_map[element - low()];
#else
        return m_elem2tile_map.at(element - low());
#endif
      }
      
      /// Returns the number of tiles in the range.
      tile_index ntiles() const {
        return m_tiles.size();
      }

      /// Returns a pair containing high and low tile range.
      std::pair<tile_index, tile_index> tile_range() const {
        return std::pair<tile_index,tile_index>(0, ntiles());
      }

      /// Returns the first index of the tile
      element_index
      low(tile_index tile_index) const
      {
#ifdef NDEBUG
        return m_tiles[tile_index].low();
#else
        return m_tiles.at(tile_index).low();
#endif
      }

      /// Returns the plast index of the tile
      element_index
      high(tile_index tile_index) const
      {
#ifdef NDEBUG
        return m_tiles[tile_index].high();
#else
        return m_tiles.at(tile_index).high();
#endif
      }

      /// Returns the number of elements in a tile.
      element_index
      size(tile_index tile_index) const
      {
#ifdef NDEBUG
        return m_tiles[tile_index].size();
#else
        return m_tiles.at(tile_index).size();
#endif
      }

      /// Returns a pair that contains low and high of the tile.
      std::pair<element_index, element_index>
      range(tile_index tile_index) const
      {
        return std::make_pair(low(tile_index),high(tile_index));
      }

      bool
      empty() const {
        return m_tiles.empty();
      }

      /// Returns the low element index of the range.
      element_index low() const
      {
        if (!empty())
        return m_tiles.begin()->low();
        else
        return 0;
      }

      /// Returns the high element index of the range.
      element_index
      high() const
      {
        if (!empty())
        return m_tiles.rbegin()->high();
        else
        return 0;
      }

      /// Returns the number of elements in the range.
      element_index size() const
      {
        return high() - low();
      }

      std::pair<element_index, element_index>
      range() const
      {
        return std::make_pair(low(),high());
      }

      /// Returns an iterator to the first tile in the range.
      const_iterator
      begin() const
      {
        return m_tiles.begin();
      }

      /// Returns an iterator to the end of the range.
      const_iterator
      end() const
      {
        return m_tiles.end();
      }
      
      /// contains this element?
      bool
      contains(const element_index& a) {
        return a < high() && a >= low();
      }
      /// contains this tile?
      bool
      contains_tile(const tile_index& a) {
        return a < ntiles() && a >= 0;
      }

    }; // end of Range

    inline std::ostream& operator <<(std::ostream& out, const Range& rng) {
      out << "Range(" << "low=" << rng.low() << ", high=" << rng.high()
      << ", size=" << rng.size() << ", range= [" << rng.range().first << "," << rng.range().second << ")" << ", ntiles= " << rng.ntiles() << ", tile_range= ["
      << rng.tile_range().first << "," << rng.tile_range().second << ") )";
      return out;
    }

}; // end of namespace TiledArray

#endif // RANGE_H__INCLUDED
