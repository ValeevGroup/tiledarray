#ifndef RANGE_H__INCLUDED
#define RANGE_H__INCLUDED

#include <TiledArray/tiled_range1.h>
#include <iterator>
#include <iosfwd>
#include <world/sharedptr.h>
#include <boost/utility/enable_if.hpp>

namespace TiledArray {

  template <unsigned int>
  class Permutation;
  template <typename>
  class TiledRange;
  template <typename CS>
  void swap(TiledRange<CS>&, TiledRange<CS>&);
  template <typename CS>
  TiledRange<CS> operator ^(const Permutation<CS::dim>&, const TiledRange<CS>&);
  template <typename CS>
  bool operator ==(const TiledRange<CS>&, const TiledRange<CS>&);

  /// TiledRange is a tiled DIM-dimensional range. It is immutable, to simplify API.
  template <typename CS>
  class TiledRange : public boost::equality_comparable1< TiledRange<CS> > {
	public:
    // typedefs
    typedef TiledRange<CS> TiledRange_;
    typedef CS coordinate_system;
    typedef CoordinateSystem<CS::dim, CS::level - 1, CS::order, typename CS::ordinal_index> tile_coordinate_system;

    typedef TiledRange1<coordinate_system> tiled_range1_type;
    typedef Range<coordinate_system> range_type;
    typedef Range<tile_coordinate_system> tile_range_type;

    typedef typename coordinate_system::volume_type volume_type;
    typedef typename coordinate_system::index index;
    typedef typename tile_coordinate_system::index tile_index;
    typedef typename coordinate_system::ordinal_index ordinal_index;
    typedef typename coordinate_system::size_array size_array;

	private:
    typedef std::array<tiled_range1_type,coordinate_system::dim> Ranges;

	public:
    /// Default constructor
    TiledRange() : range_(), element_range_(), ranges_() { }

    /// Constructed with a set of ranges pointed to by [ first, last ).
    template <typename InIter>
    TiledRange(InIter first, InIter last) {
      TA_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      TA_ASSERT(std::distance(first, last) == coordinate_system::dim, std::runtime_error,
          "Tiling for all dimensions must be specified.");
      std::copy(first, last, ranges_.begin());
      init_();
    }

    /// Copy constructor
    TiledRange(const TiledRange& other) :
        range_(other.range_), element_range_(other.element_range_), ranges_(other.ranges_)
    { }

    /// In place permutation of this range.

    /// \return A reference to this object
    TiledRange& operator ^=(const Permutation<coordinate_system::dim>& p) {
      TiledRange temp = p ^ *this;
      TiledArray::swap(*this, temp);
      return *this;
    }

    /// TiledRange assignment operator

    /// \return A reference to this object
    TiledRange& operator =(const TiledRange& other) {
      TiledRange temp(other);
      swap(*this, temp);
      return *this;
    }

    /// Resize the tiled range

    /// \tparam InIter Input iterator type (InIter::value_type == tiled_range1_type).
    /// \param first An input iterator to the beginning of a tile boundary array set.
    /// \param last An input iterator to one past the end of a tile boundary array set.
    /// \return A reference to the newly resized Tiled Range
    template <typename InIter>
    TiledRange& resize(InIter first, InIter last) {
      TA_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      TiledRange temp(first, last);
      swap(*this, temp);
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
    std::shared_ptr<tile_range_type> make_tile_range(const ordinal_index& i) const {
      TA_ASSERT(range_.includes(i), std::runtime_error, "Index i is not included in the range.");
      return make_tile_range(coordinate_system::calc_index(i, range_.weight()));
    }

    /// Construct a range for the given tile.

    /// \param i The index of the tile range to be constructed
    /// \throw std::runtime_error Throws if i is not included in the range
    /// \return The constructed range object
    std::shared_ptr<tile_range_type> make_tile_range(const index& i) const {
      TA_ASSERT(range_.includes(i), std::runtime_error, "Index i is not included in the range.");
      tile_index start;
      tile_index finish;
      for(unsigned int d = 0; d < coordinate_system::dim; ++d) {
        start[d] = ranges_[d].tile(i[d]).start();
        finish[d] = ranges_[d].tile(i[d]).finish();
      }

      return std::make_shared<tile_range_type>(start, finish);
    }

  private:
    /// precomputes useful data listed below
    void init_() {
      // Indices used to store range start and finish.
      index start;
      index finish;
      tile_index start_element;
      tile_index finish_element;

      // Find the start and finish of the over all tiles and element ranges.
      for(unsigned int d=0; d < coordinate_system::dim; ++d) {
        start[d] = ranges_[d].tiles().start();
        finish[d] = ranges_[d].tiles().finish();

        start_element[d] = ranges_[d].elements().start();
        finish_element[d] = ranges_[d].elements().finish();
      }
      range_.resize(start, finish);
      element_range_.resize(start_element, finish_element);
    }

    /// Returns the tile index of the tile that contains the given element index.
    index element2tile_(const tile_index& e) const {
      index result;
      if(elements().includes(e))
        for(unsigned int d = 0; d < coordinate_system::dim; ++d)
          result[d] = ranges_[d].element2tile(e[d]);
      else
        result = range_.finish().data();

      return result;
    }

    friend void swap<>(TiledRange_&, TiledRange_&);
    friend TiledRange operator ^ <>(const Permutation<coordinate_system::dim>&, const TiledRange<CS>&);
    friend bool operator == <>(const TiledRange_&, const TiledRange_&);

    range_type range_; ///< Stores information on tile indexing for the range.
    tile_range_type element_range_; ///< Stores information on element indexing for the range.
    Ranges ranges_; ///< Stores tile boundaries for each dimension.
  };

  /// TiledRange permutation operator.

  /// This function will permute the range. Note: only tiles that are not
  /// being used by other objects will be permuted. The owner of those
  /// objects are
  template <typename CS>
  TiledRange<CS> operator ^(const Permutation<CS::dim>& p, const TiledRange<CS>& r) {
    TiledRange<CS> result(r);
    result.ranges_ ^= p;
    result.range_ ^= p;
    result.element_range_ ^= p;

    return result;
  }

  /// Exchange the content of the two given TiledRange's.
  template <typename CS>
  void swap(TiledRange<CS>& r0, TiledRange<CS>& r1) {
    TiledArray::swap(r0.range_, r1.range_);
    TiledArray::swap(r0.element_range_, r1.element_range_);
    std::swap(r0.ranges_, r1.ranges_);
  }

  /// Returns true when all tile and element ranges are the same.
  template <typename CS>
  bool operator ==(const TiledRange<CS>& r1, const TiledRange<CS>& r2) {
    return
#ifndef NDEBUG
        // Do some extra equality checking while debugging. If everything is
        // working properly, the range data will be consistent with the data in
        // ranges.
        (r1.range_ == r2.range_) && (r1.element_range_ == r2.element_range_) &&
#endif
        std::equal(r1.ranges_.begin(), r1.ranges_.end(), r2.ranges_.begin());
  }

  template <typename CS>
  bool operator !=(const TiledRange<CS>& r1, const TiledRange<CS>& r2) {
    return ! operator ==(r1, r2);
  }

  template <typename CS>
  std::ostream& operator<<(std::ostream& out, const TiledRange<CS>& rng) {
    out << "(" << " tiles = " << rng.tiles()
        << ", elements = " << rng.elements() << " )";
    return out;
  }

} // namespace TiledArray


#endif // RANGE_H__INCLUDED
