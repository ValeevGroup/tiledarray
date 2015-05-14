/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
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

#ifndef RANGE_H__INCLUDED
#define RANGE_H__INCLUDED

#include <TiledArray/tiled_range1.h>
#include <TiledArray/range.h>

namespace TiledArray {

  /// Range data of an N-dimensional, tiled tensor.
  class TiledRange {
  private:

    /// Constructed with a set of ranges pointed to by [ first, last ).
    void init() {
      const std::size_t dim = ranges_.size();

      // Indices used to store range start and finish.
      std::vector<size_type> start;
      std::vector<size_type> finish;
      std::vector<size_type> start_element;
      std::vector<size_type> finish_element;

      start.reserve(dim);
      finish.reserve(dim);
      start_element.reserve(dim);
      finish_element.reserve(dim);

      // Find the start and finish of the over all tiles and element ranges.
      for(unsigned int d=0; d < dim; ++d) {
        start.push_back(ranges_[d].tiles().first);
        finish.push_back(ranges_[d].tiles().second);

        start_element.push_back(ranges_[d].elements().first);
        finish_element.push_back(ranges_[d].elements().second);
      }
      range_type(start, finish).swap(range_);
      tile_range_type(start_element, finish_element).swap(element_range_);
    }

  public:
    // typedefs
    typedef TiledRange TiledRange_;
    typedef Range range_type;
    typedef Range tile_range_type;
    typedef std::size_t size_type;
    typedef range_type::index index;
    typedef range_type::size_array size_array;
    typedef std::vector<TiledRange1> Ranges;

    /// Default constructor
    TiledRange() : range_(), element_range_(), ranges_() { }

    /// Constructed with a set of ranges pointed to by [ first, last ).
    template <typename InIter>
    TiledRange(InIter first, InIter last) :
      range_(), element_range_(), ranges_(first, last)
    {
      init();
    }

    /// Constructed with a set of ranges pointed to by [ first, last ).
    template <typename T>
    TiledRange(const std::initializer_list<std::initializer_list<T> >& list) :
      range_(), element_range_(), ranges_(list.begin(), list.end())
    {
      init();
    }

    /// Constructed with an initializer_list of TiledRange1s
    TiledRange(const std::initializer_list<TiledRange1>& list) :
      range_(), element_range_(), ranges_(list.begin(), list.end())
    {
      init();
    }

    /// Copy constructor
    TiledRange(const TiledRange_& other) :
        range_(other.range_), element_range_(other.element_range_), ranges_(other.ranges_)
    { }

    /// TiledRange assignment operator

    /// \return A reference to this object
    TiledRange_& operator =(const TiledRange_& other) {
      if(this != &other)
        TiledRange_(other).swap(*this);
      return *this;
    }

    /// In place permutation of this range.

    /// \return A reference to this object
    TiledRange_& operator *=(const Permutation& p) {
      TA_ASSERT(p.dim() == range_.dim());
      Ranges temp = p * ranges_;
      TiledRange(temp.begin(), temp.end()).swap(*this);
      return *this;
    }

    /// Access the range information on the tiles

    /// \return A const reference to the tile range object
    const range_type& tiles() const {
      return range_;
    }

    /// Access the range information on the elements

    /// \return A const reference to the element range object
    const tile_range_type& elements() const {
      return element_range_;
    }


    /// Construct a range for the given index.

    /// \param i The ordinal index of the tile range to be constructed
    /// \throw std::runtime_error Throws if i is not included in the range
    /// \return The constructed range object
    tile_range_type make_tile_range(const size_type& i) const {
      TA_ASSERT(range_.includes(i));
      return make_tile_range(tiles().idx(i));
    }

    /// Construct a range for the given tile.

    /// \param i The index of the tile range to be constructed
    /// \throw std::runtime_error Throws if i is not included in the range
    /// \return The constructed range object
    template <typename Index>
    typename std::enable_if<! std::is_integral<Index>::value, tile_range_type>::type
    make_tile_range(const Index& i) const {
      const unsigned int dim = range_.dim();
      TA_ASSERT(i.size() == dim);
      TA_ASSERT(range_.includes(i));
      typename tile_range_type::index start;
      typename tile_range_type::index finish;
      start.reserve(dim);
      finish.reserve(dim);
      for(unsigned int d = 0; d < dim; ++d) {
        start.push_back(data()[d].tile(i[d]).first);
        finish.push_back(data()[d].tile(i[d]).second);
      }

      return tile_range_type(start, finish);
    }

    /// Convert an element index to a tile index

    /// \tparam Index the index type
    /// \param index The element index to convert
    /// \return The tile index that corresponds to the given element index
    template <typename Index>
    typename std::enable_if<! std::is_integral<Index>::value, typename range_type::index>::type
    element_to_tile(const Index& index) const {
      const unsigned int dim = range_.dim();
      typename range_type::index result;
      result.reserve(dim);
      for(size_type i = 0; i < dim; ++i)
        result.push_back(ranges_[i].element2tile(index[i]));

      return result;
    }

    /// Tile dimension boundary array accessor

    /// \return A reference to the array of Range1 objects.
    /// \throw nothing
    const Ranges& data() const { return ranges_; }


    void swap(TiledRange_& other) {
      range_.swap(other.range_);
      element_range_.swap(other.element_range_);
      std::swap(ranges_, other.ranges_);
    }

  private:
    range_type range_; ///< Stores information on tile indexing for the range.
    tile_range_type element_range_; ///< Stores information on element indexing for the range.
    Ranges ranges_; ///< Stores tile boundaries for each dimension.
  };

  /// TiledRange permutation operator.

  /// This function will permute the range. Note: only tiles that are not
  /// being used by other objects will be permuted. The owner of those
  /// objects are
  inline TiledRange operator *(const Permutation& p, const TiledRange& r) {
    TA_ASSERT(r.tiles().dim() == p.dim());
    TiledRange::Ranges data = p * r.data();

    return TiledRange(data.begin(), data.end());
  }

  /// Exchange the content of the two given TiledRange's.
  inline void swap(TiledRange& r0, TiledRange& r1) { r0.swap(r1); }

  /// Returns true when all tile and element ranges are the same.
  inline bool operator ==(const TiledRange& r1, const TiledRange& r2) {
    return (r1.tiles().dim() == r2.tiles().dim()) &&
        (r1.tiles() == r2.tiles()) && (r1.elements() == r2.elements()) &&
        std::equal(r1.data().begin(), r1.data().end(), r2.data().begin());
  }

  inline bool operator !=(const TiledRange& r1, const TiledRange& r2) {
    return ! operator ==(r1, r2);
  }

  inline std::ostream& operator<<(std::ostream& out, const TiledRange& rng) {
    out << "(" << " tiles = " << rng.tiles()
        << ", elements = " << rng.elements() << " )";
    return out;
  }

} // namespace TiledArray


#endif // RANGE_H__INCLUDED
