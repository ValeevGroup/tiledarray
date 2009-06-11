#include <range1.h>
#include <algorithm>
#include <boost/limits.hpp>
#include <ostream>

namespace TiledArray {

  namespace detail {
    bool operator ==(const Tile1& t1, const Tile1& t2) {
      return (t1.index == t2.index) && (t1.start == t2.start) && (t1.finish == t2.finish);
    }

    bool operator !=(const Tile1& t1, const Tile1& t2) {
      return (t1.index != t2.index) || (t1.start != t2.start) || (t1.finish != t2.finish);
    }

  } // namespace detail

  /// Default constructor, range of 1 tile and element.
  Range1::Range1() :  tiles_(1, detail::Tile1(0,0,0)), elem2tile_() {
    init_map_();
  }

  /// Copy constructor
  Range1::Range1(const Range1& rng) :
    tiles_(rng.tiles_), elem2tile_(rng.elem2tile_) {
  }

  /// Returns the low tile index of the range.
  Range1::index_type Range1::start() const {
    return tiles_.front().index;
  }

  /// Returns the high tile index of the range.
  Range1::index_type Range1::finish() const {
    return tiles_.back().index;
  }

  /// Returns the low element index of the range.
  Range1::tile_index_type Range1::start_element() const {
    return tiles_.front().start;
  }

  /// Returns the high element index of the range.
  Range1::tile_index_type Range1::finish_element() const {
    return tiles_.back().finish;
  }

  /// Returns the low element index of the given tile.
  Range1::tile_index_type Range1::start_element(index_type t) const {
#ifdef NDEBUG
    return tiles_[t - start()].start;
#else
    return tiles_.at(t - start()).start;
#endif
  }

  /// Returns the low element index of the given tile.
  Range1::tile_index_type Range1::finish_element(index_type t) const {
#ifdef NDEBUG
    return tiles_[t - start()].finish;
#else
    return tiles_.at(t - start()).finish;
#endif
  }

  /// Assignment operator
  Range1& Range1::operator =(const Range1& rng) {
    Range1 temp(rng);
    swap(temp);
    return *this;
  }

  /// Returns an iterator to the end of the range.
  Range1::const_iterator Range1::end() const {
    return tiles_.end();
  }

  /// Returns an iterator to the first tile in the range.
  Range1::const_iterator Range1::begin() const {
    return tiles_.begin();
  }

  /// Return tile iterator associated with Range1::tile_index_type
  Range1::const_iterator Range1::find(tile_index_type element) const {
    assert(includes_element(element));
    std::size_t relindex = element - start_element();
    const_iterator result = tiles_.begin() + relindex;
    return result;
  }

  /// Returns the number of tiles in the range.
  size_t Range1::size() const {
    return tiles_.size();
  }

  /// Returns the number of elements in the range.
  size_t Range1::size_element() const {
    return finish_element() - start_element();
  }

  /// Returns the number of elements in a tile.
  size_t Range1::size_element(const index_type t) const {
#ifdef NDEBUG
    return tiles_[t].size();
#else
    return tiles_.at(t).size;
#endif
  }

  /// contains this element?
  bool Range1::includes_element(const tile_index_type& a) const {
    return a < finish_element() && a >= start_element();
  }

  /// contains this tile?
  bool Range1::includes(const index_type& a) const {
    return a < finish() && a >= start();
  }

  /// Swap the data of this Range1 with another.
  void Range1::swap(Range1& other) { // no throw
    std::swap(tiles_, other.tiles_);
    std::swap(elem2tile_, other.elem2tile_);
  }

  /// Initialize secondary data
  void Range1::init_map_() {
    // check for 0 size range.
    if(tiles_.front().size == 0)
      return;

    // initialize elem2tile map
    elem2tile_.resize(tiles_.back().finish - tiles_.front().start);
    size_t ii = 0;
    for(const_iterator it = begin(); it != end(); ++it)
      for(tile_index_type i = 0; i < it->size; ++i, ++ii)
        elem2tile_[ii] = it->index;
  }

  /// Equality operator
  bool operator ==(const Range1& r1, const Range1& r2) {
    return std::equal(r1.begin(), r1.end(), r2.begin());
  }

  /// Inequality operator
  bool operator !=(const Range1& r1, const Range1& r2){
    return ! operator ==(r1, r2);
  }

  std::ostream& operator <<(std::ostream& out, const Range1& rng) {
    out << "Range1( tiles=[" << rng.start() << "," << rng.finish()
      << ") element=[" << rng.start_element() << "," << rng.finish_element()
      << ") " << " element size=" << rng.size_element() << " size= " << rng.size() << " )";
    return out;
  }

  std::ostream& operator <<(std::ostream& out, const detail::Tile1& t) {
	  out << "Tile( range[" << t.index << "] = [" << t.start << "," << t.finish << "), size= " << t.size << " )";
	  return out;
  }

} // namesapce TiledArray
