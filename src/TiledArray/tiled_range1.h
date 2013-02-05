#ifndef TILEDARRAY_TILED_RANGE1_H__INCLUDED
#define TILEDARRAY_TILED_RANGE1_H__INCLUDED

#include <TiledArray/range.h>

namespace TiledArray {

  /// TiledRange1 class defines a non-uniformly-tiled, continuous, one-dimensional
  /// range. The tiling data is constructed with and stored in an array with
  /// the format {a0, a1, a2, ...}, where 0 <= a0 < a1 < a2 < ... Each tile is
  /// defined as [a0,a1), [a1,a2), ... The number of tiles in the range will be
  /// equal to one less than the number of elements in the array.
  class TiledRange1 {
  private:
    struct Enabler { };
  public:
    typedef std::size_t size_type;
    typedef std::pair<size_type, size_type> range_type;
    typedef std::vector<range_type>::const_iterator const_iterator;

    /// Default constructor, range of 0 tiles and elements.
    TiledRange1() :
        range_(0,0), element_range_(0,0),
        tile_ranges_(1, range_type(0,0)), elem2tile_(1, 0)
    {
      init_map_();
    }

    /// Constructs a range with the boundaries provided by [first, last).
    /// Start_tile_index is the index of the first tile.
    template <typename RandIter>
    TiledRange1(RandIter first, RandIter last, const size_type start_tile_index = 0,
        typename madness::enable_if<detail::is_random_iterator<RandIter>, Enabler >::type = Enabler()) :
        range_(), element_range_(), tile_ranges_(), elem2tile_()
    {
      TA_STATIC_ASSERT(detail::is_random_iterator<RandIter>::value);
      init_tiles_(first, last, start_tile_index);
      init_map_();
    }

    /// Copy constructor
    TiledRange1(const TiledRange1& rng) :
        range_(rng.range_), element_range_(rng.element_range_),
        tile_ranges_(rng.tile_ranges_), elem2tile_(rng.elem2tile_)
    { }

    /// Construct a 1D tiled range.

    /// This will construct a 1D tiled range with the given tile boundaries. The
    /// first argument is the number of tiles. The number of tile boundaries
    /// must be n + 1. Tiles are defined as [t0, t1), [t1, t2), [t2, t3), ...
    /// \param start_tile_index
    /// \param n is the number of tiles.
    /// \param t0 The first lower bound
    /// \param t1 ... are the tile boundaries.
    explicit TiledRange1(const size_type start_tile_index, const std::size_t n, const size_type t0, const size_type t1, ...) {
      TA_ASSERT(n >= 1);
      va_list ap;
      va_start(ap, t1);

      std::vector<size_type> r;
      r.push_back(t0);
      r.push_back(t1);
      size_type ti; // ci is used as an intermediate
      for(unsigned int i = 1; i < n; ++i) {
        ti = 0ul;
        ti = va_arg(ap, size_type);
        r.push_back(ti);
      }

      va_end(ap);

      init_tiles_(r.begin(), r.end(), start_tile_index);
      init_map_();
    }

    /// Assignment operator
    TiledRange1& operator =(const TiledRange1& rng) {
      TiledRange1(rng).swap(*this);
      return *this;
    }

    /// Returns an iterator to the first tile in the range.
    const_iterator begin() const { return tile_ranges_.begin(); }

    /// Returns an iterator to the end of the range.
    const_iterator end() const { return tile_ranges_.end(); }

    /// Return tile iterator associated with ordinal_index
    const_iterator find(const size_type& e) const{
      if(! includes(element_range_, e))
        return tile_ranges_.end();
      const_iterator result = tile_ranges_.begin();
      result += element2tile(e);
      return result;
    }

    /// Tiles range accessor
    const range_type& tiles() const { return range_; }

    /// Elements range accessor
    const range_type& elements() const { return element_range_; }

    /// Tile range accessor

    /// \param i The coordinate index for the tile range to be returned
    /// \return A const reference to a the tile range for tile \c i
    /// \throw std::out_of_range When \c i \c >= \c tiles().size()
    const range_type& tile(const size_type i) const {
      TA_ASSERT(includes(range_, i));
      return tile_ranges_[i - range_.first];
    }

    const size_type& element2tile(const size_type& i) const {
      TA_ASSERT( includes(element_range_, i) );
      return elem2tile_[i - element_range_.first];
    }

    void swap(TiledRange1& other) { // no throw
      std::swap(range_, other.range_);
      std::swap(element_range_, other.element_range_);
      std::swap(tile_ranges_, other.tile_ranges_);
      std::swap(elem2tile_, other.elem2tile_);
    }

  private:

    static bool includes(const range_type& r, size_type i) { return (i >= r.first) && (i < r.second); }

    /// Validates tile_boundaries
    template <typename RandIter>
    static bool valid_(RandIter first, RandIter last) {
      // Verify at least 2 elements are present if the vector is not empty.
      if((last - first) == 2)
        return false;
      // Verify the requirement that a0 < a1 < a2 < ...
      for (; first != (last - 1); ++first)
        if(*first >= *(first + 1))
          return false;
      return true;
    }

    /// Initialize tiles use a set of tile offsets
    template <typename RandIter>
    void init_tiles_(RandIter first, RandIter last, size_type start_tile_index) {
      TA_ASSERT( valid_(first, last) );
      range_.first = start_tile_index;
      range_.second = start_tile_index + last - first - 1;
      element_range_.first = *first;
      element_range_.second = *(last - 1);
      for (; first != (last - 1); ++first)
        tile_ranges_.push_back(range_type(*first, *(first + 1)));
    }

    /// Initialize secondary data
    void init_map_() {
      // check for 0 size range.
      if((element_range_.second - element_range_.first) == 0)
        return;

      // initialize elem2tile map
      elem2tile_.resize(element_range_.second - element_range_.first);
      const size_type end = range_.second - range_.first;
      for(size_type t = 0; t < end; ++t)
        for(size_type e = tile_ranges_[t].first; e < tile_ranges_[t].second; ++e)
          elem2tile_[e - element_range_.first] = t + range_.first;
    }

    friend std::ostream& operator <<(std::ostream&, const TiledRange1&);

    // TiledRange1 data
    range_type range_; ///< stores the overall dimensions of the tiles.
    range_type element_range_; ///< stores overall element dimensions.
    std::vector<range_type> tile_ranges_; ///< stores the dimensions of each tile.
    std::vector<size_type> elem2tile_; ///< maps element index to tile index (secondary data).

  }; // class TiledRange1

  /// Exchange the data of the two given ranges.
  inline void swap(TiledRange1& r0, TiledRange1& r1) { // no throw
    r0.swap(r1);
  }

  /// Equality operator
  inline bool operator ==(const TiledRange1& r1, const TiledRange1& r2) {
    return std::equal(r1.begin(), r1.end(), r2.begin()) &&
        (r1.tiles() == r2.tiles()) && (r1.elements() == r2.elements());
  }

  /// Inequality operator
  inline bool operator !=(const TiledRange1& r1, const TiledRange1& r2){
    return ! operator ==(r1, r2);
  }

  /// TiledRange1 ostream operator
  inline std::ostream& operator <<(std::ostream& out, const TiledRange1& rng) {
    out << "( tiles = [ " << rng.tiles().first << ", " << rng.tiles().second
        << " ), elements = [ " << rng.elements().first << ", " << rng.elements().second << " ) )";
    return out;
  }

} // namespace TiledArray

#endif // TILEDARRAY_TILED_RANGE1_H__INCLUDED
