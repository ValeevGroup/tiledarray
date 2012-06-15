#ifndef RANGE_H__INCLUDED
#define RANGE_H__INCLUDED

#include <TiledArray/tiled_range1.h>
#include <TiledArray/permutation.h>
#include <iterator>
#include <algorithm>
#include <vector>
#include <iosfwd>

namespace TiledArray {

  template <typename> class TiledRange;
  template <typename> class StaticTiledRange;
  class DynamicTiledRange;
  template <typename Derived>
  Derived operator ^(const Permutation&, const TiledRange<Derived>&);


  template <typename> struct RangeTraits;

  template <typename CS>
  struct RangeTraits<StaticTiledRange<CS> > {
    typedef CS coordinate_system;
    typedef CoordinateSystem<CS::dim, CS::level - 1> tile_coordinate_system;
    typedef StaticRange<coordinate_system> range_type;
    typedef StaticRange<tile_coordinate_system> tile_range_type;
    typedef typename range_type::size_array size_array;
    typedef std::array<TiledRange1,coordinate_system::dim> Ranges;
  };

  template <>
  struct RangeTraits<DynamicTiledRange> {
    typedef void coordinate_system;
    typedef void tile_coordinate_system;
    typedef DynamicRange range_type;
    typedef DynamicRange tile_range_type;
    typedef range_type::size_type size_array;
    typedef std::vector<TiledRange1> Ranges;
  };

  /// Range data of an N-dimensional, tiled tensor.

  /// \tparam Derived The type of the class that uses this class as a base class
  /// (i.e. curiously recurring template pattern).
  /// TiledRange is an interface class for tiled range objects. It handles
  /// operations that are common to both fixed size and non-fixed size tiled
  /// ranges.
  /// \note This object should not be used directly. Instead use \c StaticTiledRange
  /// for fixed dimension or \c DynamicTiledRange for non-fixed dimension tiled
  /// ranges.
  template <typename Derived>
  class TiledRange {
	public:
    // typedefs
    typedef TiledRange<Derived> TiledRange_;
    typedef typename RangeTraits<Derived>::coordinate_system coordinate_system;
    typedef typename RangeTraits<Derived>::tile_coordinate_system tile_coordinate_system;

    typedef typename RangeTraits<Derived>::range_type range_type;
    typedef typename RangeTraits<Derived>::tile_range_type tile_range_type;

    typedef std::size_t size_type;
    typedef typename RangeTraits<Derived>::size_array size_array;
    typedef typename RangeTraits<Derived>::Ranges Ranges;

    Derived& derived() { return static_cast<Derived&>(*this); }
    const Derived& derived() const { return static_cast<const Derived&>(*this); }

    // Compiler generated constructor and destructor are OK

    TiledRange_& operator=(const TiledRange_& other) {
      derived() = other.derived();
      return *this;
    }

    template <typename D>
    TiledRange_& operator=(const TiledRange<D>& other) {
      derived() = other.derived();
      return *this;
    }

    /// Access the range information on the tiles

    /// \return A const reference to the tile range object
    const range_type& tiles() const {
      return derived().tiles();
    }

    /// Access the range information on the elements

    /// \return A const reference to the element range object
    const tile_range_type& elements() const {
      return derived().elements();
    }

    /// Construct a range for the given index.

    /// \param i The ordinal index of the tile range to be constructed
    /// \throw std::runtime_error Throws if i is not included in the range
    /// \return The constructed range object
    tile_range_type make_tile_range(const size_type& i) const {
      TA_ASSERT(tiles().includes(i));
      return make_tile_range(tiles().idx(i));
    }

    /// Construct a range for the given tile.

    /// \param i The index of the tile range to be constructed
    /// \throw std::runtime_error Throws if i is not included in the range
    /// \return The constructed range object
    template <typename Index>
    tile_range_type make_tile_range(const Index& i) const {
      TA_ASSERT(tiles().includes(i));
      typename range_type::index  idx = tiles().idx(i);
      typename tile_range_type::index start;
      typename tile_range_type::index finish;
      size_index_(start);
      size_index_(finish);
      const unsigned int dim = derived().tiles().dim();
      for(unsigned int d = 0; d < dim; ++d) {
        start[d] = data()[d].tile(idx[d]).first;
        finish[d] = data()[d].tile(idx[d]).second;
      }

      return tile_range_type(start, finish);
    }

    /// Convert an element index to a tile index

    /// \tparam Index the index type
    /// \param index The element index to convert
    /// \return The tile index that corresponds to the given element index
    template <typename Index>
    typename madness::disable_if<std::is_integral<Index>, typename range_type::index>::type
    element_to_tile(const Index& index) const {
      typename range_type::index result = tiles().start();
      for(size_type i = 0; i < tiles().dim(); ++i) {
        result[i] = data()[i].element2tile(index[i]);
      }

      return result;
    }

    /// Tile dimension boundary array accessor

    /// \return A reference to the array of Range1 objects.
    /// \throw nothing
    const Ranges& data() const { return derived().data(); }

  private:

    template <typename T>
    std::vector<T>& size_index_(std::vector<T>& i) const {
      if(i.size() != tiles().dim())
        i.resize(tiles().dim());

      return i;
    }

    template <typename T>
    T& size_index_(T& i) const {
      TA_ASSERT(tiles().dim() == i.size());
      return i;
    }

  };


  template <typename CS>
  class StaticTiledRange : public TiledRange< StaticTiledRange<CS> > {
  public:
    // typedefs
    typedef StaticTiledRange<CS> StaticTiledRange_;
    typedef TiledRange< StaticTiledRange_ > base;
    typedef typename base::coordinate_system coordinate_system;
    typedef typename base::tile_coordinate_system tile_coordinate_system;

    typedef typename base::range_type range_type;
    typedef typename base::tile_range_type tile_range_type;

    typedef typename base::size_type size_type;
    typedef typename base::size_array size_array;
    typedef typename base::Ranges Ranges;

    /// Default constructor
    StaticTiledRange() : range_(), element_range_(), ranges_() { }

    /// Constructed with a set of ranges pointed to by [ first, last ).
    template <typename InIter>
    StaticTiledRange(InIter first, InIter last) {
      TA_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      TA_ASSERT(std::distance(first, last) == coordinate_system::dim);
      std::copy(first, last, ranges_.begin());
      init_();
    }

    /// Copy constructor
    StaticTiledRange(const StaticTiledRange_& other) :
        range_(other.range_), element_range_(other.element_range_), ranges_(other.ranges_)
    { }

    /// Copy constructor
    template <typename D>
    StaticTiledRange(const TiledRange<D>& other) :
        range_(other.tiles()), element_range_(other.elements()), ranges_()
    {
      std::copy(other.data().begin(), other.data().end(), ranges_.begin());
    }

    /// TiledRange assignment operator

    /// \return A reference to this object
    StaticTiledRange_& operator =(const StaticTiledRange_& other) {
      if(this != &other)
        StaticTiledRange_(other).swap(*this);
      return *this;
    }

    template <typename D>
    StaticTiledRange_& operator =(const TiledRange<D>& other) {
      if(this != &(other.derived()))
        StaticTiledRange_(other).swap(*this);
      return *this;
    }

    /// In place permutation of this range.

    /// \return A reference to this object
    StaticTiledRange_& operator ^=(const Permutation& p) {
      TA_ASSERT(p.dim() == tiles().dim());
      StaticTiledRange_(p ^ *this).swap(*this);
      return *this;
    }

    /// Access the range information on the tiles

    /// \return A const reference to the tile range object
    const range_type& tiles() const { return range_; }

    /// Access the range information on the elements

    /// \return A const reference to the element range object
    const tile_range_type& elements() const { return element_range_; }

    /// Tile dimension boundary array accessor

    /// \return A reference to the array of Range1 objects.
    /// \throw nothing
    const Ranges& data() const { return ranges_; }

    void swap(StaticTiledRange_& other) {
      range_.swap(other.range_);
      element_range_.swap(other.element_range_);
      std::swap(ranges_, other.ranges_);
    }

  private:
    /// precomputes useful data listed below
    void init_() {
      // Indices used to store range start and finish.
      typename coordinate_system::index start;
      typename coordinate_system::index finish;
      typename tile_coordinate_system::index start_element;
      typename tile_coordinate_system::index finish_element;

      // Find the start and finish of the over all tiles and element ranges.
      for(unsigned int d=0; d < coordinate_system::dim; ++d) {
        start[d] = ranges_[d].tiles().first;
        finish[d] = ranges_[d].tiles().second;

        start_element[d] = ranges_[d].elements().first;
        finish_element[d] = ranges_[d].elements().second;
      }
      range_type(start, finish).swap(range_);
      tile_range_type(start_element, finish_element).swap(element_range_);
    }

//    /// Returns the tile index of the tile that contains the given element index.
//    index element2tile_(const tile_index& e) const {
//      index result;
//      if(elements().includes(e))
//        for(unsigned int d = 0; d < coordinate_system::dim; ++d)
//          result[d] = ranges_[d].element2tile(e[d]);
//      else
//        result = range_.finish().data();
//
//      return result;
//    }

    range_type range_; ///< Stores information on tile indexing for the range.
    tile_range_type element_range_; ///< Stores information on element indexing for the range.
    Ranges ranges_; ///< Stores tile boundaries for each dimension.
  };


  class DynamicTiledRange : public TiledRange<DynamicTiledRange> {
  public:
    // typedefs
    typedef DynamicTiledRange DynamicTiledRange_;
    typedef TiledRange< DynamicTiledRange_ > base;
    typedef base::coordinate_system coordinate_system;
    typedef base::tile_coordinate_system tile_coordinate_system;

    typedef base::range_type range_type;
    typedef base::tile_range_type tile_range_type;

    typedef base::size_type size_type;
    typedef base::size_array size_array;
    typedef base::Ranges Ranges;

    /// Default constructor
    DynamicTiledRange() : range_(), element_range_(), ranges_() { }

    /// Constructed with a set of ranges pointed to by [ first, last ).
    template <typename InIter>
    DynamicTiledRange(InIter first, InIter last) :
      ranges_(first, last)
    {
      TA_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      init_();
    }

    /// Copy constructor
    DynamicTiledRange(const DynamicTiledRange_& other) :
        range_(other.range_), element_range_(other.element_range_), ranges_(other.ranges_)
    { }

    /// Copy constructor
    template <typename D>
    DynamicTiledRange(const TiledRange<D>& other) :
        range_(other.tiles()), element_range_(other.elements()),
        ranges_(other.data().begin(), other.data().end())
    { }

    /// TiledRange assignment operator

    /// \return A reference to this object
    DynamicTiledRange_& operator =(const DynamicTiledRange_& other) {
      if(this != &other)
        DynamicTiledRange_(other).swap(*this);
      return *this;
    }


    template <typename D>
    DynamicTiledRange_& operator =(const TiledRange<D>& other) {
      if(this != &(other.derived()))
        DynamicTiledRange_(other).swap(*this);
      return *this;
    }

    /// In place permutation of this range.

    /// \return A reference to this object
    DynamicTiledRange_& operator ^=(const Permutation& p) {
      TA_ASSERT(p.dim() == tiles().dim());
      DynamicTiledRange_(p ^ *this).swap(*this);
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

    /// Tile dimension boundary array accessor

    /// \return A reference to the array of Range1 objects.
    /// \throw nothing
    const Ranges& data() const { return ranges_; }


    void swap(DynamicTiledRange_& other) {
      range_.swap(other.range_);
      element_range_.swap(other.element_range_);
      std::swap(ranges_, other.ranges_);
    }

  private:
    /// precomputes useful data listed below
    void init_() {
      const std::size_t dim = ranges_.size();

      // Indices used to store range start and finish.
      std::vector<size_type> start(dim, 0);
      std::vector<size_type> finish(dim, 0);
      std::vector<size_type> start_element(dim, 0);
      std::vector<size_type> finish_element(dim, 0);

      // Find the start and finish of the over all tiles and element ranges.
      for(unsigned int d=0; d < dim; ++d) {
        start[d] = ranges_[d].tiles().first;
        finish[d] = ranges_[d].tiles().second;

        start_element[d] = ranges_[d].elements().first;
        finish_element[d] = ranges_[d].elements().second;
      }
      range_type(start, finish).swap(range_);
      tile_range_type(start_element, finish_element).swap(element_range_);
    }

    /// Returns the tile index of the tile that contains the given element index.
//    index element2tile_(const tile_index& e) const {
//      index result;
//      if(elements().includes(e))
//        for(unsigned int d = 0; d < coordinate_system::dim; ++d)
//          result[d] = ranges_[d].element2tile(e[d]);
//      else
//        result = range_.finish().data();
//
//      return result;
//    }

    range_type range_; ///< Stores information on tile indexing for the range.
    tile_range_type element_range_; ///< Stores information on element indexing for the range.
    Ranges ranges_; ///< Stores tile boundaries for each dimension.
  };

  /// TiledRange permutation operator.

  /// This function will permute the range. Note: only tiles that are not
  /// being used by other objects will be permuted. The owner of those
  /// objects are
  template <typename Derived>
  Derived operator ^(const Permutation& p, const TiledRange<Derived>& r) {
    TA_ASSERT(r.tiles().dim() == p.dim());
    typename Derived::Ranges data = p ^ r.data();

    return Derived(data.begin(), data.end());
  }

  /// Exchange the content of the two given TiledRange's.
  template <typename CS>
  void swap(TiledRange<CS>& r0, TiledRange<CS>& r1) {
    TiledArray::swap(r0.range_, r1.range_);
    TiledArray::swap(r0.element_range_, r1.element_range_);
    std::swap(r0.ranges_, r1.ranges_);
  }

  /// Returns true when all tile and element ranges are the same.
  template <typename D1, typename D2>
  bool operator ==(const TiledRange<D1>& r1, const TiledRange<D2>& r2) {
    return (r1.tiles().dim() == r2.tiles().dim()) &&
        (r1.tiles() == r2.tiles()) && (r1.elements() == r2.elements()) &&
        std::equal(r1.data().begin(), r1.data().end(), r2.data().begin());
  }

  template <typename D1, typename D2>
  bool operator !=(const TiledRange<D1>& r1, const TiledRange<D2>& r2) {
    return ! operator ==(r1, r2);
  }

  template <typename D>
  std::ostream& operator<<(std::ostream& out, const TiledRange<D>& rng) {
    out << "(" << " tiles = " << rng.tiles()
        << ", elements = " << rng.elements() << " )";
    return out;
  }

} // namespace TiledArray


#endif // RANGE_H__INCLUDED
