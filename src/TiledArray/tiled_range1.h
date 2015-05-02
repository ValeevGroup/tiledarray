/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef TILEDARRAY_TILED_RANGE1_H__INCLUDED
#define TILEDARRAY_TILED_RANGE1_H__INCLUDED

#include <TiledArray/range.h>
#include <initializer_list>

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
        typename std::enable_if<detail::is_random_iterator<RandIter>::value, Enabler >::type = Enabler()) :
        range_(), element_range_(), tile_ranges_(), elem2tile_()
    {
      static_assert(detail::is_random_iterator<RandIter>::value,
          "TiledRange1 constructor requires a random access iterator");
      init_tiles_(first, last, start_tile_index);
      init_map_();
    }

    /// Copy constructor
    TiledRange1(const TiledRange1& rng) :
        range_(rng.range_), element_range_(rng.element_range_),
        tile_ranges_(rng.tile_ranges_), elem2tile_(rng.elem2tile_)
    { }

    /// Construct a 1D tiled range.

    /// This will construct a 1D tiled range with tile boundaries {t0, t_rest}
    /// The number of tile boundaries is n + 1, where n is the number of tiles.
    /// Tiles are defined as [t0, t1), [t1, t2), [t2, t3), ...
    /// \param t0 The starting index of the first tile
    /// \param t_rest The rest of tile boundaries
    template<typename... _sizes>
    explicit TiledRange1(const size_type& t0, const _sizes&... t_rest)
    {
      const size_type n = sizeof...(_sizes) + 1;
      size_type tile_boundaries[n] = {t0, static_cast<size_type>(t_rest)...};
      init_tiles_(tile_boundaries, tile_boundaries+n, 0);
      init_map_();
    }

    /// Construct a 1D tiled range.

    /// This will construct a 1D tiled range with tile boundaries {t0, t_rest}
    /// The number of tile boundaries is n + 1, where n is the number of tiles.
    /// Tiles are defined as [t0, t1), [t1, t2), [t2, t3), ...
    /// \tparam T The element type of the initializer list.
    /// \param list The list of tile boundaries in order from smallest to largest
    template<typename T>
    explicit TiledRange1(const std::initializer_list<T>& list)
    {
      init_tiles_(list.begin(), list.end(), 0);
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
    static void valid_(RandIter first, RandIter last) {
      // Verify at least 2 elements are present if the vector is not empty.
      TA_USER_ASSERT((std::distance(first, last) >= 2),
          "TiledRange1 construction failed: You need at least 2 elements in the tile boundary list.");
      // Verify the requirement that a0 < a1 < a2 < ...
      for (; first != (last - 1); ++first)
        TA_USER_ASSERT(*first < *(first + 1),
            "TiledRange1 construction failed: Invalid tile boundary, tile boundary i must be greater than tile boundary i+1 for all i. ");
    }

    /// Initialize tiles use a set of tile offsets
    template <typename RandIter>
    void init_tiles_(RandIter first, RandIter last, size_type start_tile_index) {
#ifndef NDEBUG
      valid_(first, last);
#endif // NDEBUG
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
