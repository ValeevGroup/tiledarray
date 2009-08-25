#ifndef TILEDARRAY_TILE_SLICE_H__INCLUDED
#define TILEDARRAY_TILE_SLICE_H__INCLUDED

#include <iterator.h>
#include <range.h>
#include <type_traits.h>
#include <boost/type_traits.hpp>

namespace TiledArray {

  /// \c TileSlice represents an arbitrary sub-range of a tile. \c TileSlice
  /// does not contain any element data. The primary use of \c TileSlice is to
  /// provide the ability to iterate over a sub section of the referenced tile.
  /// All element access is done via index translation between the slice and the
  /// tile.
  /// Note: Ordinal indexes for the slice are not equivalent to the tile ordinal
  /// indexes.
  /// Note: The memory of the slice may or may not be contiguous, depending on
  /// the slice selected.
  template<class T>
  class TileSlice
  {
  public:
    typedef TileSlice<T> TileSlice_;
    typedef typename boost::remove_const<T>::type tile_type;
    typedef typename tile_type::value_type value_type;
    typedef typename detail::mirror_const<T, value_type>::reference reference_type;
    typedef const value_type& const_reference_type;
    typedef typename tile_type::coordinate_system coordinate_system;
    typedef typename tile_type::range_type range_type;
    typedef typename tile_type::index_type index_type;
    typedef typename tile_type::size_array size_array;
    typedef typename tile_type::volume_type volume_type;
    typedef typename tile_type::index_iterator index_iterator;
    typedef detail::ElementIterator<value_type, typename range_type::const_iterator, TileSlice_> iterator;
    typedef detail::ElementIterator<const value_type, typename range_type::const_iterator, TileSlice_> const_iterator;

    static unsigned int dim() { return tile_type::dim(); }

    /// Slice constructor.

    /// Constructs a slice of tile \c t given a sub range. The range \c r must
    /// be completely contained by the tile range. You may easily construct a
    /// range that is contained by the original tile range by using the &
    /// operator on the tile range and an arbitrary range (i.e. slice_range =
    /// t.range() & other_range;).
    /// \arg \c t is the tile which the slice will reference.
    /// \arg \c r is the range which defines the slice.
    ///
    /// Warning: Iteration and element access for a slice are more expensive
    /// operations than the equivalent tile operations. If you need to iterate
    /// over a slice in a time critical loop, you may want to copy the slice
    /// into a new tile object.
    TileSlice(T& t, const range_type& r) : r_(r), t_(t)
    {
      TA_ASSERT( ( valid_range_(r, t) ) ,
          std::runtime_error("TileSlice<...>::TileSlice(...): Range slice is not contained by the range of the original tile."));
    }

    /// Copy constructor
    TileSlice(const TileSlice_& other) : r_(other.r_), t_(other.t_) { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// Move constructor
    TileSlice(TileSlice&& other) : r_(std::move(other.r_)), t_(other.t_) { }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    ~TileSlice() { }

    /// Assignment operator
    TileSlice_& operator =(const TileSlice_& other) {
      r_ = other.r_;
      t_ = other.t_;

      return *this;
    }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// Move assignment operator
    TileSlice_& operator =(TileSlice_&& other) {
      r_ = std::move(other.r_);
      t_ = other.t_;

      return *this;
    }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    /// Returns an iterator to the first element of the tile.
    iterator begin() { return iterator(r_.begin(), this); } // no throw

    /// Returns an iterator that points to the end of the tile.
    iterator end() { return iterator(r_.end(), this); } // no throw

    /// Returns a const_iterator to the first element of the tile.
    const_iterator begin() const { return const_iterator(r_.begin(), const_cast<TileSlice_*>(this)); } // no throw

    /// Returns a const_iterator that points to the end of the tile.
    const_iterator end() const { return const_iterator(r_.end(), const_cast<TileSlice_*>(this)); } // no throw

    /// return a constant reference to the tile \c Range<> object.
    const range_type& range() const { return r_; }
    /// Returns the tile range start.
    const index_type& start() const { return r_.start(); }
    /// Returns the tile range finish.
    const index_type& finish() const { return r_.finish(); }
    /// Returns the tile range size.
    const size_array& size() const { return r_.size(); }
    /// Returns the number of elements in the volume.
    const volume_type volume() const { return r_.volume(); }

    /// Returns true when index \c i is included in the tile.
    /// \arg \c i Element index.
    bool includes(const index_type& i) const { return r_.includes(i); }
    /// Returns true if index \c i is included in the tile.

    // The at() functions do error checking, but we do not need to implement it
    // here because the data container already does that. There is no need to do
    // it twice.
    /// Element access with range checking
    reference_type at(const index_type& i) { return t_.at(i); }

    /// Element access with range checking
    const_reference_type at(const index_type& i) const { return t_.at(i); }

    /// Element access without error checking
    reference_type operator [](const index_type& i) { return t_[i]; }

    /// Element access without error checking
    const_reference_type operator [](const index_type& i) const { return t_[i]; }

    /// Exchange calling tile's data with that of \c other.
    void swap(TileSlice_& other) {
      r_.swap(other.r_);
      std::swap(t_, other.t_);
    }

  private:

    TileSlice(); ///< No default construction allowed.

    bool valid_range_(const range_type& r, const tile_type& t) const {
      if(detail::greater_eq(r.start().data(), t.start().data()) &&
          detail::less(r.start().data(), t.finish().data()) &&
          detail::greater(r.finish().data(), t.start().data()) &&
          detail::less_eq(r.finish().data(), t.finish().data()))
        return true;

      return false;
    }

    range_type r_;  ///< tile slice dimension information
    T& t_;          ///< element data

  }; // class TileSlice

} // namespace TiledArray

#endif // TILEDARRAY_TILE_SLICE_H__INCLUDED
