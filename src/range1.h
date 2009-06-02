#ifndef RANGE1_H__INCLUDED
#define RANGE1_H__INCLUDED

#include <debug.h>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <cassert>
#include <iosfwd>
#include <boost/operators.hpp>
#include <boost/limits.hpp>

namespace TiledArray {

  /// Tile is used to store an interval [start,finish)
  struct Tile1 {
    std::size_t index;
    std::size_t start;
    std::size_t finish;
    std::size_t size;

    Tile1() : index(0), start(0), finish(0), size(0) { }
    Tile1(std::size_t s, std::size_t f, std::size_t i) :
        index(i), start(s), finish(f), size(f - s)
    { TA_ASSERT(start <= finish);}
  };

  inline bool operator ==(const Tile1& t1, const Tile1& t2) {
    return (t1.index == t2.index) && (t1.start == t2.start) && (t1.finish == t2.finish);
  }

  inline bool operator !=(const Tile1& t1, const Tile1& t2) { return ! operator ==(t1, t2); }

  /** Range1 class defines a nonuniformly-tiled continuous one-dimensional range.
   The tiling data is constructed with and stored in an array with
   the format {a, b, c, ...}, where 0 <= a < b < c < ... Each tile is
   defined by [a,b), [b,c), ... The number of tiles in the range is
   defined as one less than the number of elements in the array.
   */
  class Range1 : boost::equality_comparable1<Range1> {
  public:

    typedef size_t element_index;
    typedef size_t tile_index;
    typedef std::vector<Tile1> Tiles;
    typedef Tiles::const_iterator const_iterator;

  private:
    /////////////
    // Range1 data
    /////////////

    /// set of tiles. This is the primary data, i.e. other data is always recomputed from this, not modified directly
    Tiles tiles_;
    /// maps element index to tile index (secondary data)
    std::vector<tile_index> elem2tile_;

    /// Validates tile_boundaries
    static bool valid_(const std::vector<element_index>& tile_boundaries) {
   	  // Verify at least 2 elements are present if the vector is not empty.
      if(tile_boundaries.size() == 1)
        return false;
      // Verify the requirement that 0 <= a < b < c < ...
      if(tile_boundaries[0] < 0)
        return false;
      for (size_t index = 1; index < tile_boundaries.size(); ++index)
        if (tile_boundaries[index - 1] >= tile_boundaries[index])
          return false;
      return true;
    }

    /// Initialize tiles use a set of tile offsets
    void init_tiles_(const std::vector<element_index>& tile_boundaries, const tile_index start_tile) {
      assert(valid_(tile_boundaries));
      for (size_t i = 1; i < tile_boundaries.size(); ++i)
        tiles_.push_back(Tile1(tile_boundaries[i - 1], tile_boundaries[i], start_tile + i - 1));
    }

    /// Initialize secondary data
    void init_map_() {
      if(tiles_.empty())
        return;

      // initialize elem2tile map
      elem2tile_.resize(size());
      size_t ii = 0;
      for(tile_index t = start_tile(); t < finish_tile(); ++t)
        for(element_index i = 0; i < size(t); ++i, ++ii)
          elem2tile_[ii] = t;
    }

    friend std::ostream& operator <<(std::ostream& out, const Range1& rng);

  public:
    /// Default constructor, range of 1 tile and element.
    Range1() :
      tiles_(), elem2tile_()
    {
      init_map_();
    }

    /// Constructs range from a vector
    Range1(std::vector<element_index> tile_boundaries, tile_index start_t = 0) {
      init_tiles_(tile_boundaries, start_t);
      init_map_();
    }

    /**
     * Construct range from a C-style array, ranges must have at least
     * ntiles + 1 elements.
     */
    Range1(const element_index* tile_boundaries, tile_index ntiles, tile_index start_t = 0) {
      init_tiles_(std::vector<element_index>(tile_boundaries, tile_boundaries+ntiles+1), start_t);
      init_map_();
    }

    /// Copy constructor
    Range1(const Range1& rng) :
      tiles_(rng.tiles_), elem2tile_(rng.elem2tile_) {
    }

    /// Returns the low tile index of the range.
    tile_index start_tile() const {
      if(!empty())
        return tiles_.begin()->index;
      else
        return std::numeric_limits<tile_index>::min();
    }

    /// Returns the high tile index of the range.
    tile_index finish_tile() const {
      if(!empty())
        return tiles_.rbegin()->index + 1;
      else
        return std::numeric_limits<tile_index>::min();
    }

    /// Returns the low element index of the range.
    element_index start_element() const {
      if (!empty())
        return tiles_.begin()->start;
      else
        return std::numeric_limits<element_index>::min();
    }

    /// Returns the high element index of the range.
    element_index finish_element() const {
      if (!empty())
        return tiles_.rbegin()->finish;
      else
        return std::numeric_limits<element_index>::min();
    }

    /// Returns the low element index of the given tile.
    element_index start_element(tile_index t) const {
#ifdef NDEBUG
      return tiles_.at(t - start_tile()).start;
#else
      return tiles_[t - start_tile()].start;
#endif
    }

    /// Returns the low element index of the given tile.
    element_index finish_element(tile_index t) const {
#ifdef NDEBUG
      return tiles_.at(t - start_tile()).finish;
#else
      return tiles_[t - start_tile()].finish;
#endif
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

    /// Returns an iterator to the end of the range.
    const_iterator end() const {
      return tiles_.end();
    }

    /// Returns an iterator to the first tile in the range.
    const_iterator begin() const {
        return tiles_.begin();
    }

    /// Return tile iterator associated with element_index
    const Tile1& find(element_index element) const {
      TA_ASSERT(includes_element(element));
      element_index relindex = element - start_element();
#ifdef NDEBUG
      return tiles_[ elem2tile_[relindex] ];
#else
      return tiles_.at(elem2tile_.at(relindex));
#endif
    }

/*
    /// increment element index
    void increment(element_index& i) const {
      ++i;
    }
*/
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

    bool empty() const {
      return tiles_.empty();
    }

    /// Returns the number of elements in the range.
    size_t size() const {
      return finish_element() - start_element();
    }

    /// Returns the number of elements in a tile.
    size_t size(const tile_index t) const {
#ifdef NDEBUG
      return tiles_[t].size();
#else
      return tiles_.at(t).size;
#endif
    }

    /// contains this element?
    bool
    includes_element(const element_index& a) const {
      return a < finish_element() && a >= start_element();
    }

    /// contains this tile?
    bool
    includes_tile(const tile_index& a) const {
      return a < finish_tile() && a >= start_tile();
    }

  }; // Range1

  inline std::ostream& operator <<(std::ostream& out, const Range1& rng) {
    out << "Range1( tiles=[" << rng.start_tile() << "," << rng.finish_tile()
      << ") element=[" << rng.start_element() << "," << rng.finish_element()
      << ") " << " size=" << rng.size() << " ntiles= " << rng.ntiles() << " )";
    return out;
  }

  inline std::ostream& operator <<(std::ostream& out, const Tile1& t) {
	  out << "Tile( range[" << t.index << "] = [" << t.start << "," << t.finish << "), size= " << t.size << " )";
	  return out;
  }

}; // end of namespace TiledArray

#endif // RANGE1_H__INCLUDED
