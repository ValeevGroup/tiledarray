#ifndef RANGE1_H__INCLUDED
#define RANGE1_H__INCLUDED

#include <block.h>
#include <coordinates.h>
#include <cassert>
#include <vector>
#include <cstddef>
#include <iosfwd>

namespace TiledArray {

  template <typename I>
  class Range1;
  template <typename I>
  std::ostream& operator <<(std::ostream& out, const Range1<I>& rng);

  /// Range1 class defines a non-uniformly-tiled, continuous, one-dimensional
  /// range. The tiling data is constructed with and stored in an array with
  /// the format {a0, a1, a2, ...}, where 0 <= a0 < a1 < a2 < ... Each tile is
  /// defined as [a0,a1), [a1,a2), ... The number of tiles in the range will be
  /// equal to one less than the number of elements in the array.
  template <typename I>
  class Range1 {
  public:
    typedef I tile_index_type;
    typedef I index_type;
    typedef Block<I,1, LevelTag<1>, CoordinateSystem<1> > block_type;
    typedef Block<I,1,LevelTag<0>,  CoordinateSystem<1> > element_block_type;
    typedef element_block_type tile_block_type;
    typedef typename std::vector<element_block_type>::const_iterator const_iterator;

    /// Default constructor, range of 0 tiles and elements.
    Range1() : block_(0,0), element_block_(0,0),
        tile_blocks_(1, tile_block_type(0,0)), elem2tile_(1, 0)
    {
      init_map_();
    }

    /// Constructs a range with the boundaries provided by [first, last).
    /// Start_tile_index is the index of the first tile.
    template <typename RandIter>
    Range1(RandIter first, RandIter last, const index_type start_tile_index = 0) :
        block_(), element_block_(), tile_blocks_(), elem2tile_()
    {
      init_tiles_(first, last, start_tile_index);
      init_map_();
    }

    /// Copy constructor
    Range1(const Range1& rng) : block_(rng.block_), element_block_(rng.element_block_),
      tile_blocks_(rng.tile_blocks_), elem2tile_(rng.elem2tile_)
    { }

    /// Assignment operator
    Range1& operator =(const Range1& rng) {
      Range1 temp(rng);
      swap(temp);
      return *this;
    }

    /// Returns an iterator to the first tile in the range.
    const_iterator begin() const { return tile_blocks_.end(); }

    /// Returns an iterator to the end of the range.
    const_iterator end() const { return tile_blocks_.begin(); }

    /// Return tile iterator associated with tile_index_type
    const_iterator find(const tile_index_type& e) const{
      const_iterator result = tile_blocks_.begin();
      result += element2tile(e);
      return result;
    }

    const block_type& tiles() const { return block_; }
    const element_block_type& elements() const { return element_block_; }
    const tile_block_type& tile(const index_type& i) {
#ifdef NDEBUG
      return tile_blocks_[i - block_.start()];
#else
      return tile_blocks_.at(i - block_.start());
#endif
    }

    /// Swap the data of this range with another.
    void swap(Range1& other) { // no throw
      block_.swap(other.block_);
      element_block_.swap(other.element_block_);
      std::swap(tile_blocks_, other.tile_blocks_);
      std::swap(elem2tile_, other.elem2tile_);
    }

    const index_type& element2tile(const tile_index_type& e) const {
      assert(element_block_.includes(e));
      std::size_t i = e - element_block_.start();
      return elem2tile_[i];
    }

  private:

    /// Validates tile_boundaries
    template <typename RandIter>
    static bool valid_(RandIter first, RandIter last) {
      // Verify at least 2 elements are present if the vector is not empty.
      if((last - first) == 2)
        return false;
      // Verify the requirement that 0 <= a0
      if(*first < 0)
        return false;
      // Verify the requirement that a0 < a1 < a2 < ...
      for (; first != (last - 1); ++first)
        if(*first >= *(first + 1))
          return false;
      return true;
    }

    /// Initialize tiles use a set of tile offsets
    template <typename RandIter>
    void init_tiles_(RandIter first, RandIter last, index_type start_tile_index) {
      assert(valid_(first, last));
      for (; first != (last - 1); ++first)
        tile_blocks_.push_back(tile_block_type(*first, *(first + 1)));
    }

    /// Initialize secondary data
    void init_map_() {
      // check for 0 size range.
      if(element_block_.size() == 0)
        return;

      // initialize elem2tile map
      elem2tile_.resize(element_block_.size());
      for(index_type t = 0; t < block_.size(); ++t)
        for(tile_index_type e = tile_blocks_[t].start(); e < tile_blocks_[t].finish(); ++e)
          elem2tile_[e - element_block_.start()] = t + block_.start();
    }

    friend std::ostream& operator << <>(std::ostream&, const Range1&);

    /////////////
    // Range1 data
    /////////////

    block_type block_;
    element_block_type element_block_;
    /// set of tiles. This is the primary data, i.e. other data is always recomputed from this, not modified directly
    std::vector<tile_block_type> tile_blocks_;
    /// maps element index to tile index (secondary data)
    std::vector<index_type> elem2tile_;

  }; // class Range1

  /// Equality operator
  template <typename I>
  bool operator ==(const Range1<I>& r1, const Range1<I>& r2) {
    return std::equal(r1.begin(), r1.end(), r2.begin());
  }

  /// Inequality operator
  template <typename I>
  bool operator !=(const Range1<I>& r1, const Range1<I>& r2){
    return ! operator ==(r1, r2);
  }

  /// Range1 ostream operator
  template <typename I>
  std::ostream& operator <<(std::ostream& out, const Range1<I>& rng) {
    out << "( tiles = " << rng.tiles() << ", elements = " << rng.elements() << " )";
    return out;
  }

}; // end of namespace TiledArray

#endif // RANGE1_H__INCLUDED
