#ifndef RANGE_H__INCLUDED
#define RANGE_H__INCLUDED

//#include <coordinates.h>
#include <tiled_range1.h>
#include <array_storage.h>
//#include <iosfwd>
//#include <boost/array.hpp>
//#include <boost/operators.hpp>

namespace TiledArray {

  // Forward declaration of TiledArray Permutation.
  template <unsigned int DIM>
  class Permutation;
  template <typename I>
  class TiledRange1;

  // need these forward declarations
  template<typename I, unsigned int DIM, typename CS>
  class TiledRange;
  template<typename I, unsigned int DIM, typename CS>
  void swap(TiledRange<I, DIM, CS>&, TiledRange<I, DIM, CS>&);
  template<typename I, unsigned int DIM, typename CS>
  TiledRange<I,DIM,CS> operator ^(const Permutation<DIM>&, const TiledRange<I,DIM,CS>&);
  template<typename I, unsigned int DIM, typename CS>
  bool operator ==(const TiledRange<I,DIM,CS>&, const TiledRange<I,DIM,CS>&);
  template<typename I, unsigned int DIM, typename CS>
  bool operator !=(const TiledRange<I, DIM, CS>&, const TiledRange<I, DIM, CS>&);
  template<typename I, unsigned int DIM, typename CS>
  std::ostream& operator<<(std::ostream& out, const TiledRange<I,DIM,CS>& rng);

  /// TiledRange is a tiled DIM-dimensional range. It is immutable, to simplify API.
  template<typename I, unsigned int DIM, typename CS = CoordinateSystem<DIM> >
  class TiledRange : public boost::equality_comparable1< TiledRange<I,DIM,CS> > {
	public:
      // typedefs
      typedef TiledRange<I,DIM,CS> TiledRange_;
      typedef CS coordinate_system;
      typedef TiledRange1<I> tiled_range1_type;
      typedef Range<I,DIM,LevelTag<1>,coordinate_system> range_type;
      typedef Range<I,DIM,LevelTag<0>,coordinate_system> element_range_type;
      typedef element_range_type tile_range_type;
      typedef typename range_type::size_array size_array;
      typedef typename range_type::index_type index_type;
      typedef typename range_type::volume_type volume_type;
      typedef typename tile_range_type::index_type element_index_type;
      typedef typename tile_range_type::index_type tile_index_type;
      typedef DenseArrayStorage<tile_range_type, DIM, LevelTag<1>, coordinate_system > tile_container;
      typedef typename tile_container::iterator iterator;
      typedef typename tile_container::const_iterator const_iterator;

      /// Returns the number of dimensions
      static unsigned int dim() { return DIM; }

      // Default constructor
      TiledRange() : range_(), element_range_(), tile_ranges_(), ranges_() { }

      // Constructed with a set of ranges pointed to by [ first, last ).
      template <typename InIter>
      TiledRange(InIter first, InIter last) {
        for(typename Ranges::iterator it = ranges_.begin(); it != ranges_.end(); ++it, ++first) {
          TA_ASSERT( (first != last),
              std::runtime_error("TiledRange<...>::TiledRange(...): iterator unexpectedly reached the end of the range.") );
          *it = *first;
        }
        init_();
      }

      /// Copy constructor
      TiledRange(const TiledRange& other) :
          range_(other.range_), element_range_(other.element_range_),
          tile_ranges_(other.tile_ranges_), ranges_(other.ranges_)
      { }

      /// Returns a const_iterator to the first tile range in TiledRange.
      const_iterator begin() const { return tile_ranges_.begin(); }
      /// Return a const_iterator to the end of the tile range.
      const_iterator end() const { return tile_ranges_.end(); }

      /// Return iterator to the tile range that contains the element index.
      const_iterator find(const tile_index_type e) const {
        const_iterator result;
        if(element_range_.includes(e)) {
          index_type t = element2tile_(e);
          result = tile_ranges_.begin() + tile_ranges_.ordinal(t);
        } else {
          result = tile_ranges_.end();
        }
        return result;
      }

      /// In place permutation of range.

      /// This function will permute the range. Note: only tiles that are not
      /// being used by other objects will be permuted. The owner of those
      /// objects are
      TiledRange& operator ^=(const Permutation<DIM>& p) {
        TiledRange temp = p ^ *this;
        TiledArray::swap(*this, temp);
        return *this;
      }

      /// TiledRange assignment operator
      TiledRange& operator =(const TiledRange& other) {
        TiledRange temp(other);
        swap(*this, temp);
        return *this;
      }

      /// Resize the range to the set of dimensions in [first, last) input
      /// iterators. The iterators must dereference to a tiled_range1_type.
      template <typename InIter>
      TiledRange& resize(InIter first, InIter last) {
        TiledRange temp(first, last);
        swap(*this, temp);
        return *this;
      }

      /// Access the range information on the tiles contained by the range.
      const range_type& tiles() const {
        return range_;
      }

      /// Access the range information on the elements contained by the range.
      const element_range_type& elements() const {
        return element_range_;
      }

      /// Access the range information on the elements contained by tile t.
      const tile_range_type& tile(const index_type& t) const {
        TA_ASSERT( tile_ranges_.includes(t),
            std::out_of_range("TiledRange<...>::tile(...) const: Tile index is out of range."));
        return tile_ranges_[t - range_.start()];
      }

    private:
      /// precomputes useful data listed below
      void init_() {
        // Indices used to store range start and finish.
        index_type start;
        index_type finish;
        tile_index_type start_element;
        tile_index_type finish_element;

        // Find the start and finish of the over all tiles and element ranges.
        for(unsigned int d=0; d < DIM; ++d) {
          start[d] = ranges_[d].tiles().start()[0];
          finish[d] = ranges_[d].tiles().finish()[0];

          start_element[d] = ranges_[d].elements().start()[0];
          finish_element[d] = ranges_[d].elements().finish()[0];
        }
        range_.resize(start, finish);
        element_range_.resize(start_element, finish_element);

        // Find the dimensions of each tile and store its range information.
        tile_index_type start_tile;
        tile_index_type finish_tile;
        tile_ranges_.resize(range_.size());
        for(typename range_type::const_iterator it = range_.begin(); it != range_.end(); ++it) {
          // Determine the start and finish of each tile.
          for(unsigned int d = 0; d < DIM; ++d) {
            start_tile[d] = ranges_[d].tile( (*it)[d] ).start()[0];
            finish_tile[d] = ranges_[d].tile( (*it)[d] ).finish()[0];
          }

          // Create and store the tile range.
          tile_ranges_[ *it - range_.start()] = tile_range_type(start_tile, finish_tile);
        }
      }

      /// Returns the tile index of the tile that contains the given element index.
      index_type element2tile_(const tile_index_type& e) const {
        index_type result;
        if(elements().includes(e)) {
          for(unsigned int d = 0; d < DIM; ++d) {
            result[d] = ranges_[d].element2tile(e[d]);
          }
        } else {
          result = range_.finish().data();
        }

        return result;
      }

      friend void swap<>(TiledRange_&, TiledRange_&);
      friend TiledRange operator ^ <>(const Permutation<DIM>&, const TiledRange<I,DIM,CS>&);
      friend bool operator == <>(const TiledRange&, const TiledRange&);

      range_type range_; ///< Stores information on tile indexing for the range.
      element_range_type element_range_; ///< Stores information on element indexing for the range.
      tile_container tile_ranges_; ///< Stores a indexing information for each tile in the range.
      typedef boost::array<tiled_range1_type,DIM> Ranges;
      Ranges ranges_; ///< Stores tile boundaries for each dimension.

  };

  /// TiledRange permutation operator.

  /// This function will permute the range. Note: only tiles that are not
  /// being used by other objects will be permuted. The owner of those
  /// objects are
  template<typename I, unsigned int DIM, typename CS>
  TiledRange<I,DIM,CS> operator ^(const Permutation<DIM>& p, const TiledRange<I,DIM,CS>& r) {
    TiledRange<I,DIM,CS> result(r);
    result.ranges_ ^= p;
    result.range_ ^= p;
    result.element_range_ ^= p;
    result.tile_ranges_ ^= p;
    for(typename TiledRange<I,DIM,CS>::range_type::const_iterator it = r.tiles().begin(); it != r.tiles().end(); ++it)
      result.tile_ranges_[ p ^ *it ] ^= p;

    return result;
  }

  /// Exchange the content of the two given TiledRange's.
  template<typename I, unsigned int DIM, typename CS>
  void swap(TiledRange<I, DIM, CS>& r0, TiledRange<I, DIM, CS>& r1) {
    TiledArray::swap(r0.range_, r1.range_);
    TiledArray::swap(r0.element_range_, r1.element_range_);
    TiledArray::swap(r0.tile_ranges_, r1.tile_ranges_);
    boost::swap(r0.ranges_, r1.ranges_);
  }

  /// Returns true when all tile and element ranges are the same.
  template<typename I, unsigned int DIM, typename CS>
  bool operator ==(const TiledRange<I, DIM, CS>& r1, const TiledRange<I, DIM, CS>& r2) {
    return
#ifndef NDEBUG
        // Do some extra equality checking while debugging. If everything is
        // working properly, the range data will be consistent with the data in
        // ranges.
        (r1.range_ == r2.range_) && (r1.element_range_ == r2.element_range_) &&
        std::equal(r1.tile_ranges_.begin(), r1.tile_ranges_.end(), r2.tile_ranges_.begin()) &&
#endif
        std::equal(r1.ranges_.begin(), r1.ranges_.end(), r2.ranges_.begin());
  }

  template<typename I, unsigned int DIM, typename CS>
  bool operator !=(const TiledRange<I, DIM, CS>& r1, const TiledRange<I, DIM, CS>& r2) {
    return ! operator ==(r1, r2);
  }

  template<typename I, unsigned int DIM, typename CS>
  std::ostream& operator<<(std::ostream& out, const TiledRange<I,DIM,CS>& rng) {
    out << "(" << " tiles = " << rng.tiles()
        << " elements = " << rng.elements() << " )";
    return out;
  }


} // namespace TiledArray


#endif // RANGE_H__INCLUDED
