#ifndef RANGE1_H__INCLUDED
#define RANGE1_H__INCLUDED

#include <cassert>
#include <vector>
#include <cstddef>
#include <iosfwd>

namespace TiledArray {

  namespace detail {
    /// Tile is used to store an interval [start,finish)
    struct Tile1 {
      typedef std::size_t index_type;
      index_type index;
      index_type start;
      index_type finish;
      index_type size;

      Tile1() : index(0), start(0), finish(0), size(0) { }
      Tile1(index_type s, index_type f, index_type i) :
          index(i), start(s), finish(f), size(f - s)
      { assert(start <= finish);}

    };

    bool operator ==(const Tile1& t1, const Tile1& t2);
    bool operator !=(const Tile1& t1, const Tile1& t2);
  } // namespace detail



  std::ostream& operator <<(std::ostream&, const detail::Tile1&);

  /// Range1 class defines a non-uniformly-tiled, continuous, one-dimensional
  /// range. The tiling data is constructed with and stored in an array with
  /// the format {a0, a1, a2, ...}, where 0 <= a0 < a1 < a2 < ... Each tile is
  /// defined as [a0,a1), [a1,a2), ... The number of tiles in the range will be
  /// equal to one less than the number of elements in the array.
  class Range1 {
  public:
    typedef size_t tile_index_type;
    typedef size_t index_type;
    typedef std::vector<detail::Tile1>::const_iterator const_iterator;

    /// Default constructor, range of 0 tiles and elements.
    Range1();

    /// Constructs a range with the boundaries provided by [first, last).
    /// Start_tile_index is the index of the first tile.
    template <typename RandIter>
    Range1(RandIter first, RandIter last, const index_type start_tile_index = 0);

    /// Copy constructor
    Range1(const Range1& rng);

    /// Returns the low tile index of the range.
    index_type start() const;

    /// Returns the high tile index of the range.
    index_type finish() const;

    /// Returns the low element index of the range.
    tile_index_type start_element() const;

    /// Returns the high element index of the range.
    tile_index_type finish_element() const;

    /// Returns the low element index of the given tile.
    tile_index_type start_element(index_type t) const;

    /// Returns the low element index of the given tile.
    tile_index_type finish_element(index_type t) const;

    /// Assignment operator
    Range1& operator =(const Range1& rng);

    /// Returns an iterator to the end of the range.
    const_iterator end() const;

    /// Returns an iterator to the first tile in the range.
    const_iterator begin() const;

    /// Return tile iterator associated with tile_index_type
    const_iterator find(tile_index_type e) const;

    /// Returns the number of tiles in the range.
    size_t ntiles() const;

    /// Returns the number of tiles in the range.
    size_t size() const;

    /// Returns the number of elements in the range.
    size_t size_element() const;

    /// Returns the number of elements in a tile.
    size_t size_element(const index_type t) const;

    /// contains this element?
    bool includes_element(const tile_index_type& e) const;

    /// contains this tile?
    bool includes(const index_type& t) const;

    /// Swap the data of this range with another.
    void swap(Range1& other); // no throw

  private:

    /// Validates tile_boundaries
    template <typename RandIter>
    static bool valid_(RandIter first, RandIter last);

    /// Initialize tiles use a set of tile offsets
    template <typename RandIter>
    void init_tiles_(RandIter first, RandIter last, index_type start_tile_index);

    /// Initialize secondary data
    void init_map_();

    friend std::ostream& operator <<(std::ostream&, const Range1&);

    /////////////
    // Range1 data
    /////////////

    /// set of tiles. This is the primary data, i.e. other data is always recomputed from this, not modified directly
    std::vector<detail::Tile1> tiles_;
    /// maps element index to tile index (secondary data)
    std::vector<index_type> elem2tile_;

  }; // class Range1

  /// Comparison operators
  bool operator ==(const Range1&, const Range1&);
  bool operator !=(const Range1&, const Range1&);
  std::ostream& operator <<(std::ostream&, const Range1&);

}; // end of namespace TiledArray

#endif // RANGE1_H__INCLUDED
