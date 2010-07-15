#ifndef RANGE1_H__INCLUDED
#define RANGE1_H__INCLUDED

#include <TiledArray/range.h>
//#include <TiledArray/coordinates.h>
#include <vector>
//#include <cstddef>
//#include <iosfwd>

namespace TiledArray {

  template <typename I>
  class TiledRange1;
  template <typename I>
  void swap(TiledRange1<I>&, TiledRange1<I>&);
  template<typename CS>
  Range<CoordinateSystem<1, CS::level, CS::order, typename CS::ordinal_index> >
  make_range1(const typename CS::ordinal_index&, const typename CS::ordinal_index&);
  template <typename I>
  bool operator ==(const TiledRange1<I>&, const TiledRange1<I>&);
  template <typename I>
  bool operator !=(const TiledRange1<I>&, const TiledRange1<I>&);
  template <typename I>
  std::ostream& operator <<(std::ostream&, const TiledRange1<I>&);

  /// TiledRange1 class defines a non-uniformly-tiled, continuous, one-dimensional
  /// range. The tiling data is constructed with and stored in an array with
  /// the format {a0, a1, a2, ...}, where 0 <= a0 < a1 < a2 < ... Each tile is
  /// defined as [a0,a1), [a1,a2), ... The number of tiles in the range will be
  /// equal to one less than the number of elements in the array.
  template <typename CS>
  class TiledRange1 {
  public:
    typedef CS coordinate_system;
    typedef CoordinateSystem<CS::dim, CS::level - 1, CS::order, typename CS::ordinal_index> tile_coordinate_system;

  private:
    typedef CoordinateSystem<1, CS::level, CS::order, typename CS::ordinal_index> internal_coordinate_system;
    typedef CoordinateSystem<1, CS::level - 1, CS::order, typename CS::ordinal_index> internal_element_coordinate_system;

  public:
    typedef Range<internal_coordinate_system> range_type;
    typedef Range<internal_element_coordinate_system> tile_range_type;
    typedef typename CS::ordinal_index ordinal_index;
    typedef typename std::vector<tile_range_type>::const_iterator const_iterator;

  private:
    typedef typename range_type::index range_index;
    typedef typename tile_range_type::index tile_range_index;

  public:

    /// Default constructor, range of 0 tiles and elements.
    TiledRange1() : range_(make_range1<coordinate_system>(0,0)), element_range_(make_range1<tile_coordinate_system>(0,0)),
        tile_ranges_(1, make_range1<tile_coordinate_system>(0,0)), elem2tile_(1, 0)
    {
      init_map_();
    }

    /// Constructs a range with the boundaries provided by [first, last).
    /// Start_tile_index is the index of the first tile.
    template <typename RandIter>
    TiledRange1(RandIter first, RandIter last, const ordinal_index start_tile_index = 0) :
        range_(), element_range_(), tile_ranges_(), elem2tile_()
    {
      BOOST_STATIC_ASSERT(detail::is_random_iterator<RandIter>::value);
      init_tiles_(first, last, start_tile_index);
      init_map_();
    }

    /// Copy constructor
    TiledRange1(const TiledRange1& rng) : range_(rng.range_), element_range_(rng.element_range_),
      tile_ranges_(rng.tile_ranges_), elem2tile_(rng.elem2tile_)
    { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// Construct a 1D tiled range.

    /// This will construct a 1D tiled range with the given tile boundaries. The
    /// first argument is the number of tiles. The number of tile boundaries
    /// must be n + 1. Tiles are defined as [t1, t2), [t2, t3), ...
    /// \var \c n is the number of tiles.
    /// \var \c t1, t2, ... are the tile boundaries.
    template <typename... Params>
    explicit TiledRange1(const ordinal_index start_tile_index, const std::size_t n, Params... params) {
      BOOST_STATIC_ASSERT(detail::Count<Params...>::value == coordinate_system::dim);
      BOOST_STATIC_ASSERT(detail::is_integral_list<Params...>::value);
      std::vector<ordinal_index> r(detail::Count<Params...>::value, ordinal_index(0));
      detail::fill(r.begin(), params...);

      init_tiles_(r.begin(), r.end(), start_tile_index);
      init_map_();
    }

#else
    /// Construct a 1D tiled range.

    /// This will construct a 1D tiled range with the given tile boundaries. The
    /// first argument is the number of tiles. The number of tile boundaries
    /// must be n + 1. Tiles are defined as [t1, t2), [t2, t3), ...
    /// \var \c n is the number of tiles.
    /// \var \c t1, t2, ... are the tile boundaries.
    explicit TiledRange1(const ordinal_index start_tile_index, const std::size_t n, const ordinal_index t0, const ordinal_index t1, ...) {
      TA_ASSERT(n >= 1, std::runtime_error, "There must be at least one tile.");
      va_list ap;
      va_start(ap, t1);

      std::vector<ordinal_index> r;
      r.push_back(t0);
      r.push_back(t1);
      ordinal_index ti; // ci is used as an intermediate
      for(unsigned int i = 1; i < n; ++i) {
        ci = 0ul;
        ci = va_arg(ap, ordinal_index);
        r.push_back(ti)
      }

      va_end(ap);

      init_tiles_(r.begin(), r.end(), start_tile_index);
      init_map_();
    }

#endif // __GXX_EXPERIMENTAL_CXX0X__

    /// Assignment operator
    TiledRange1& operator =(const TiledRange1& rng) {
      TiledRange1 temp(rng);
      swap(*this, temp);
      return *this;
    }

    template <typename RandIter>
    TiledRange1& resize(RandIter first, RandIter last, const ordinal_index start_tile_index = 0) {
      BOOST_STATIC_ASSERT(detail::is_random_iterator<RandIter>::value);
      TiledRange1 temp(first, last, start_tile_index);
      swap(*this, temp);
      return *this;
    }

    /// Returns an iterator to the first tile in the range.
    const_iterator begin() const { return tile_ranges_.begin(); }

    /// Returns an iterator to the end of the range.
    const_iterator end() const { return tile_ranges_.end(); }

    /// Return tile iterator associated with ordinal_index
    const_iterator find(const ordinal_index& e) const{
      if(! element_range_.includes(tile_range_index(e)))
        return tile_ranges_.end();
      const_iterator result = tile_ranges_.begin();
      result += element2tile(e);
      return result;
    }

    const range_type& tiles() const { return range_; }
    const tile_range_type& elements() const { return element_range_; }
    const tile_range_type& tile(const ordinal_index i) const {
      return tile_ranges_.at(i - range_.start()[0]);
    }

    const ordinal_index& element2tile(const ordinal_index& e) const {
      TA_ASSERT( element_range_.includes(tile_range_index(e)) ,
          std::out_of_range, "Element index is out of range.");
      std::size_t i = e - element_range_.start()[0];
      return elem2tile_[i];
    }

  private:

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
    void init_tiles_(RandIter first, RandIter last, ordinal_index start_tile_index) {
      TA_ASSERT( valid_(first, last) , std::runtime_error,
          "Tile boundaries do not have the expected structure.");
      range_.resize(range_index(start_tile_index), range_index(start_tile_index + last - first - 1));
      element_range_.resize(tile_range_index(*first), tile_range_index(*(last - 1)));
      for (; first != (last - 1); ++first)
        tile_ranges_.push_back(make_range1<tile_coordinate_system>(*first, *(first + 1)));
    }

    /// Initialize secondary data
    void init_map_() {
      // check for 0 size range.
      if(element_range_.size()[0] == 0)
        return;

      // initialize elem2tile map
      elem2tile_.resize(element_range_.size()[0]);
      for(ordinal_index t = 0; t < range_.size()[0]; ++t)
        for(ordinal_index e = tile_ranges_[t].start()[0]; e < tile_ranges_[t].finish()[0]; ++e)
          elem2tile_[e - element_range_.start()[0]] = t + range_.start()[0];
    }

    friend void swap<>(TiledRange1<CS>&, TiledRange1<CS>&);
    friend std::ostream& operator << <>(std::ostream&, const TiledRange1&);

    // TiledRange1 data
    range_type range_; ///< stores the overall dimensions of the tiles.
    tile_range_type element_range_; ///< stores overall element dimensions.
    std::vector<tile_range_type> tile_ranges_; ///< stores the dimensions of each tile.
    std::vector<ordinal_index> elem2tile_; ///< maps element index to tile index (secondary data).

  }; // class TiledRange1

  /// Exchange the data of the two given ranges.
  template <typename CS>
  void swap(TiledRange1<CS>& r0, TiledRange1<CS>& r1) { // no throw
    TiledArray::swap(r0.range_, r1.range_);
    TiledArray::swap(r0.element_range_, r1.element_range_);
    std::swap(r0.tile_ranges_, r1.tile_ranges_);
    std::swap(r0.elem2tile_, r1.elem2tile_);
  }

  /// Equality operator
  template <typename I>
  bool operator ==(const TiledRange1<I>& r1, const TiledRange1<I>& r2) {
    return std::equal(r1.begin(), r1.end(), r2.begin()) &&
        (r1.tiles() == r2.tiles()) && (r1.elements() == r2.elements());
  }

  /// Inequality operator
  template <typename I>
  bool operator !=(const TiledRange1<I>& r1, const TiledRange1<I>& r2){
    return ! operator ==(r1, r2);
  }

  template<typename CS>
  Range<CoordinateSystem<1, CS::level, CS::order, typename CS::ordinal_index> >
  make_range1(const typename CS::ordinal_index& s, const typename CS::ordinal_index& f) {
    typedef CoordinateSystem<1, CS::level, CS::order, typename CS::ordinal_index> cs1;
    typedef Range<cs1> range_type;
    typedef typename range_type::ordinal_index ordinal_index;

    return range_type(typename cs1::index(s), typename cs1::index(f));
  }

  /// TiledRange1 ostream operator
  template <typename I>
  std::ostream& operator <<(std::ostream& out, const TiledRange1<I>& rng) {
    out << "( tiles = [ " << rng.tiles().start()[0] << ", " << rng.tiles().finish()[0]
        << " ), elements = [ " << rng.elements().start()[0] << ", " << rng.elements().finish()[0] << " ) )";
    return out;
  }

} // namespace TiledArray

#endif // RANGE1_H__INCLUDED
