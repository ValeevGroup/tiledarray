#ifndef RANGE_H__INCLUDED
#define RANGE_H__INCLUDED

#include <iterator.h>
#include <vector>

/** Range class defines a nonuniformly-tiled one-dimensional range.
 The tiling data is constructed with and stored in an array with
 the format {a, b, c, ...}, where 0 <= a < b < c < ... Each tile is
 defined by [a,b), [b,c), ... The number of tiles in the range is
 defined as one less than the number of elements in the array.
 */
class Range {
  public:
    typedef size_t index_t;
    typedef size_t indexdiff_t;

    /// Represents a tile which is an interval [low,high)
    class RangeTile {
        typedef Range::index_t index_t;
        typedef Range::indexdiff_t indexdiff_t;
        /// first index
        index_t m_low;
        /// plast index, i.e. last + 1
        index_t m_high;
        /// size
        indexdiff_t m_size;

      public:
        RangeTile() :
          m_low(0), m_high(0), m_size(0) {
        }
        
        RangeTile(index_t first, index_t plast) :
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
        
        index_t low() const {
          return m_low;
        }
        index_t high() const {
          return m_high;
        }
        indexdiff_t size() const {
          return m_size;
        }
        
    };

    typedef std::vector<RangeTile> RangeTiles;

    /////////////
    // Range data
    /////////////
    /// set of tiles. This is the primary data, i.e. other data is always recomputed from this, not modified directly
    std::vector<RangeTile> m_tiles;
    /// maps element index to tile index (secondary data)
    std::vector<index_t> m_elem2tile_map;

    /// Initialize tiles use a set of tile offsets
    void init_tiles_(const std::vector<index_t>& tile_boundaries) {
      assert(valid(tile_boundaries));
      if (tile_boundaries.size() >=2) {
        for (index_t i=1; i<tile_boundaries.size(); ++i) {
          m_tiles.push_back(RangeTile(tile_boundaries[i-1], tile_boundaries[i]));
        }
      }
    }
    
    /// Initialize secondary data
    void init_() {
      // init elem2tile map
      m_elem2tile_map.resize(size());
      index_t ii = 0;
      for (index_t t=0; t < m_tiles.size(); ++t)
        for (index_t i=0; i < m_tiles[t].size(); ++i, ++ii)
          m_elem2tile_map[ii] = t;
    }
    
    // Validates tile_boundaries
    static bool valid(const std::vector<index_t>& tile_boundaries) {
      // Verify the requirement that a < b < c < ...
      for (index_t index = 1; index < tile_boundaries.size(); ++index)
        if (tile_boundaries[index - 1] >= tile_boundaries[index])
          return false;
      return true;
    }
    
  public:
    typedef RangeTiles::iterator iterator;
    typedef RangeTiles::const_iterator const_iterator;

    // Default constructor, empty range
    Range() {
      init_();
    }
    
    // Constructs range from a vector
    Range(std::vector<index_t> tile_boundaries) {
      init_tiles_(tile_boundaries);
      init_();
    }
    
    /**
     Construct range from a C-style array, ranges must have at least
     ntiles + 1 elements.
     */
    Range(index_t* tile_boundaries, index_t ntiles) {
      std::vector<index_t> tbounds(ntiles+1);
      std::copy(tile_boundaries, tile_boundaries+ntiles+1, tbounds.begin());
      
      init_tiles_(tbounds);
      init_();
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
    index_t tile(index_t element) const {
#ifdef NDEBUG
      return m_elem2tile_map[element - low()];
#else
      return m_elem2tile_map.at(element - low());
#endif
    }
    
    // Returns the number of tiles in the range.
    index_t ntiles() const {
      return m_tiles.size();
    }
    
    std::pair<index_t, index_t> tile_range() const {
      return std::pair<index_t,index_t>(0, ntiles());
    }

    /// Returns the first index of the tile
    index_t
    low(index_t tile_index) const
    {
#ifdef NDEBUG
      return m_tiles[tile_index].low();
#else
      return m_tiles.at(tile_index).low();
#endif
    }

    /// Returns the plast index of the tile
    index_t
    high(index_t tile_index) const
    {
#ifdef NDEBUG
      return m_tiles[tile_index].high();
#else
      return m_tiles.at(tile_index).high();
#endif
    }

    /// Returns the number of elements in a tile.
    indexdiff_t
    size(index_t tile_index) const
    {
#ifdef NDEBUG
      return m_tiles[tile_index].size();
#else
      return m_tiles.at(tile_index).size();
#endif
    }

    // Returns a pair that contains low and high of the tile.
    std::pair<index_t, index_t>
    range(index_t tile_index) const
    {
      return std::make_pair(low(tile_index),high(tile_index));
    }

    bool
    empty() const {
      return m_tiles.empty();
    }

    // Returns the low element index of the range.
    index_t low() const
    {
      if (!empty())
      return m_tiles.begin()->low();
      else
      return 0;
    }

    // Returns the high element index of the range.
    index_t
    high() const
    {
      if (!empty())
      return m_tiles.rbegin()->high();
      else
      return 0;
    }

    // Returns the number of elements in the range.
    indexdiff_t size() const
    {
      return high() - low();
    }

    std::pair<index_t, index_t>
    range() const
    {
      return std::make_pair(low(),high());
    }

    // Returns an iterator to the first tile in the range.
    const_iterator
    begin() const
    {
      return m_tiles.begin();
    }

    // Returns an iterator to the end of the range.
    const_iterator
    end() const
    {
      return m_tiles.end();
    }

  }; // Range

inline std::ostream& operator <<(std::ostream& out, const Range& rng) {
  out << "Range(" << "low=" << rng.low() << ", high=" << rng.high()
      << ", size=" << rng.size() << ", range= [" << rng.range().first << "," << rng.range().second << ")" << ", ntiles= " << rng.ntiles() << ", tile_range= ["
      << rng.tile_range().first << "," << rng.tile_range().second << ") )";
  return out;
}

#endif // RANGE_H__INCLUDED
