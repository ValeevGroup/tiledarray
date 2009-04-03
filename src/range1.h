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
    typedef Range1 Range1_;
  public:
    typedef size_t element_index;
    typedef size_t tile_index;
    /// Iterates over element indices
    typedef detail::IndexIterator<element_index,Range1_> element_iterator;
    /// iterates over tile indices
    typedef detail::IndexIterator<tile_index,Range1_> tile_iterator;

    /// Tile is an interval [start,finish)
    class Tile : boost::equality_comparable1<Tile> {
      typedef Range1::element_index element_index;
      typedef Range1::tile_index tile_index;
    public:
      /// Element iterator for current tile
      typedef detail::IndexIterator<element_index,Tile> iterator;

    private:
      /// first index
      element_index start_;
      /// past-last index, i.e. last + 1
      element_index finish_;
      /// index of tile with range1
      tile_index index_;

      void throw_if_invalid() {
        if (finish() < start() )
          abort();
      }

    public:
      Tile() :
        start_(std::numeric_limits<element_index>::min()),
        finish_(std::numeric_limits<element_index>::min()),
        index_(std::numeric_limits<tile_index>::min())
      { }

      /// Tile is an interval [start,finish)
      Tile(element_index start, element_index finish, tile_index idx) :
        start_(start), finish_(finish), index_(idx) {
        throw_if_invalid();
      }

      Tile(const Tile& src) :
        start_(src.start_), finish_(src.finish_), index_(src.index_) {
      }

      const Tile& operator=(const Tile& src) {
        start_ = src.start_;
        finish_ = src.finish_;
        index_ = src.index_;
        return *this;
      }

      bool operator==(const Tile& A) const {
        return start_ == A.start_ && finish_ == A.finish_ && index_ == A.index_;
      }

      element_index start() const {
        return start_;
      }

      element_index finish() const {
        return finish_;
      }

      tile_index index() const {
        return index_;
      }

      element_index size() const {
        return finish_ - start_;
      }

      iterator begin() const {
        return iterator(start(),*this);
      }

      iterator end() const {
        return iterator(finish(),*this);
      }

      void increment(element_index& i) const {
        ++i;
      }
    }; // Tile

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
        tiles_.push_back(Tile(tile_boundaries[i - 1], tile_boundaries[i], start_tile + i - 1));
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
        return tiles_.begin()->index();
      else
        return std::numeric_limits<tile_index>::min();
    }

    /// Returns the high tile index of the range.
    tile_index finish_tile() const {
      if(!empty())
        return tiles_.rbegin()->index() + 1;
      else
        return std::numeric_limits<tile_index>::min();
    }

    /// Returns the low element index of the range.
    element_index start_element() const {
      if (!empty())
        return tiles_.begin()->start();
      else
        return std::numeric_limits<element_index>::min();
    }

    /// Returns the high element index of the range.
    element_index finish_element() const {
      if (!empty())
        return tiles_.rbegin()->finish();
      else
        return std::numeric_limits<element_index>::min();
    }

    /// Returns the low element index of the given tile.
    element_index start_element(tile_index t) const {
#ifdef NDEBUG
      return tiles_.at(t - start_tile()).start();
#else
      return tiles_[t - start_tile()].start();
#endif
    }

    /// Returns the low element index of the given tile.
    element_index finish_element(tile_index t) const {
#ifdef NDEBUG
      return tiles_.at(t - start_tile()).finish();
#else
      return tiles_[t - start_tile()].finish();
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
    tile_iterator end_tile() const {
      return tile_iterator(finish_tile(),*this);
    }

    /// Returns an iterator to the first tile in the range.
    tile_iterator begin_tile() const {
      if(!tiles_.empty())
        return tile_iterator(start_tile(),*this);
      else
        return end_tile();
    }

    /// how tile_index is incremented
    void increment(tile_index& t) const {
      ++t;
    }

    /// Return tile iterator associated with element_index
    tile_iterator find(element_index element) const {
      element_index relindex = element - start_element();
      if(relindex >= size())
        return end_tile();
#ifdef NDEBUG
      return tile_iterator(tiles_[elem2tile_[relindex]].index(),*this);
#else
      return tile_iterator(tiles_.at(elem2tile_.at(relindex)).index(),*this);
#endif
    }

    element_iterator begin_element() const {
      element_iterator result(start_element(),*this);
      return result;
    }

    element_iterator end_element() const {
      element_iterator result(finish_element(),*this);
      return result;
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
      return tiles_.at(t).size();
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

  inline std::ostream& operator <<(std::ostream& out, const Range1::Tile& t) {
	  out << "Tile( range= [" << t.start() << "," << t.finish() << "), size= " << t.size() << " )";
	  return out;
  }

}; // end of namespace TiledArray

#endif // RANGE1_H__INCLUDED
