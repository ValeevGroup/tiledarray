#ifndef TILEDARRAY_TILE_H__INCLUDED
#define TILEDARRAY_TILE_H__INCLUDED

#include <TiledArray/range.h>
#include <TiledArray/type_traits.h>
#include <TiledArray/math.h>
#include <TiledArray/tile_proxy.h>
#include <TiledArray/perm_algorithm.h>
#include <world/archive.h>
#include <algorithm>
#include <functional>
#include <iosfwd>

namespace TiledArray {

  // Forward declarations
  template <typename, typename, typename>
  class Tile;
  template <typename T, typename CS, typename A>
  void swap(Tile<T, CS, A>&, Tile<T, CS, A>&);
  template <typename T, typename CS, typename A>
  Tile<T, CS, A> operator ^(const Permutation<CS::dim>&, const Tile<T, CS, A>&);

  /// Tile is an N-dimensional, dense array.

  /// \tparam T Tile element type.
  /// \tparam CS A \c CoordinateSystem type
  /// \tparam A A C++ standard library compliant allocator (Default:
  /// \c Eigen::aligned_allocator<T>)
  template <typename T, typename CS, typename A = Eigen::aligned_allocator<T> >
  class Tile : public TileBase<T,A> {
  private:
    typedef TileBase<T,A> base;

  public:
    typedef Tile<T,CS,A> Tile_;                             ///< This object's type
    typedef CS coordinate_system;                           ///< The array coordinate system

    typedef typename CS::volume_type volume_type;           ///< Array volume type
    typedef typename CS::index index;                       ///< Array coordinate index type
    typedef typename CS::ordinal_index ordinal_index;       ///< Array ordinal index type
    typedef typename CS::size_array size_array;             ///< Size array type

    typedef typename base::allocator_type allocator_type;   ///< Allocator type
    typedef typename base::size_type size_type;             ///< Size type
    typedef typename base::difference_type difference_type; ///< difference type
    typedef typename base::value_type value_type;           ///< Array element type
    typedef typename base::reference reference;             ///< Element reference type
    typedef typename base::const_reference const_reference; ///< Element reference type
    typedef typename base::pointer pointer;                 ///< Element pointer type
    typedef typename base::const_pointer const_pointer;     ///< Element const pointer type
    typedef typename base::iterator iterator;               ///< Element iterator type
    typedef typename base::const_iterator const_iterator;   ///< Element const iterator type

    typedef Range<coordinate_system> range_type;            ///< Tile range type


    /// Default constructor

    /// Constructs a tile with zero size.
    /// \note You must call resize() before attempting to access any elements.
    Tile() :
        base(), range_()
    { }

    /// Copy constructor

    /// \param other The tile to be copied.
    Tile(const Tile_& other) :
        base(other),
        range_(other.range_)
    { }

    /// Copy constructor

    /// \param other The tile to be copied.
//    Tile(const TileProxy<T,A>& other) :
//        base(other),
//        range_(index(0), index(other.size().begin()))
//    {
//      TA_ASSERT(order() == other.order(), std::runtime_error, "Array order does not match.");
//    }

    /// Constructs a new tile

    /// The tile will have the dimensions specified by the range object \c r and
    /// the elements of the new tile will be equal to \c v. The provided
    /// allocator \c a will allocate space for only for the tile data.
    /// \param r A shared pointer to the range object that will define the tile
    /// dimensions
    /// \param val The fill value for the new tile elements ( default: value_type() )
    /// \param a The allocator object for the tile data ( default: alloc_type() )
    /// \throw std::bad_alloc There is not enough memory available for the target tile
    /// \throw anything Any exception that can be thrown by \c T type default or
    /// copy constructors
    Tile(const range_type& r, const value_type& val = value_type(), const allocator_type& a = allocator_type()) :
        base(r.volume(), val, a),
        range_(r)
    { }


    /// Constructs a new tile

    /// The tile will have the dimensions specified by the range object \c r and
    /// the elements of the new tile will be equal to \c v. The provided
    /// allocator \c a will allocate space for only for the tile data.
    /// \tparam InIter An input iterator type.
    /// \param r A shared pointer to the range object that will define the tile
    /// dimensions
    /// \param first An input iterator to the beginning of the data to copy.
    /// \param last An input iterator to one past the end of the data to copy.
    /// \param a The allocator object for the tile data ( default: alloc_type() )
    /// \throw std::bad_alloc There is not enough memory available for the
    /// target tile
    /// \throw anything Any exceptions that can be thrown by \c T type default
    /// or copy constructors
    template <typename InIter>
    Tile(const range_type& r, InIter first, const allocator_type& a = allocator_type()) :
        base(r.volume(), first, a),
        range_(r)
    { }

    /// Assignment operator

    /// \param other The tile object to be moved
    /// \return A reference to this object
    /// \throw std::bad_alloc There is not enough memory available for the target tile
    Tile_& operator =(const Tile_& other) {
      Tile_(other).swap(*this);
      return *this;
    }

    /// Assignment operator

    /// \tparam U The proxy tile element type
    /// \tparam B The proxy tile allocator type
    /// \param other The tile object to be moved
    /// \return A reference to this object
    /// \throw std::bad_alloc There is not enough memory available for the target tile
    Tile_& operator =(const TileProxy<T, A>& other) {
      TA_ASSERT(std::equal(begin(), end(), other.begin(), other.end()), std::runtime_error,
          "Tile sizes are not equal.");
      std::copy(other.begin(), other.end(), begin());
      Tile_(other).swap(*this);
      return *this;
    }

    /// Assignment operator

    /// \param other The tile object to be moved
    /// \return A reference to this object
    /// \throw std::bad_alloc There is not enough memory available for the target tile
    Tile_& operator =(Tile_& other) {
      TA_ASSERT(std::equal(begin(), end(), other.begin(), other.end()), std::runtime_error,
          "Tile sizes are not equal.");
      std::copy(other.begin(), other.end(), begin());
      Tile_(other).swap(*this);
      return *this;
    }

    /// destructor
    ~Tile() { }

    /// In-place permutation of tile elements.

    /// \param p A permutation object.
    /// \return A reference to this object
    /// \warning This function modifies the shared range object.
    Tile_& operator ^=(const Permutation<coordinate_system::dim>& p) {

      // create a permuted copy of the tile data
      base temp(range_.volume());
      detail::permute_tensor<coordinate_system>(p, range_.size(), *this, temp);

      // Write changes
      range_ ^= p;
      temp.swap(*this);

      return *this;
    }

    Tile_& operator+=(const Tile_& other) {
      if(base::size() == 0ul) {
        *this = other;
      } else if(other.size() != 0ul) {
        TA_ASSERT(range() == other.range(), std::runtime_error, "The ranges must be equal.");
        const_iterator other_it = other.begin();
        for(iterator it = begin(); it != end(); ++it)
          *it += *other_it++;
      }

      return *this;
    }

    Tile_& operator+=(const value_type& value) {
      if(range().volume() != 0)
        for(iterator it = begin(); it != end(); ++it)
          *it += value;

      return *this;
    }

    Tile_& operator-=(const Tile_& other) {
      if(range().volume() == 0)
        *this = -other;
      else if(other.range().volume() != 0ul) {
        TA_ASSERT(range() == other.range(), std::runtime_error, "The ranges must be equal.");
        const_iterator other_it = other.begin();
        for(iterator it = begin(); it != end(); ++it)
          *it -= *other_it++;
      }

      return *this;
    }

    Tile_& operator-=(const value_type& value) {
      if(range().volume() != 0)
        for(iterator it = begin(); it != end(); ++it)
          *it -= value;

      return *this;
    }

    Tile_& operator*=(const value_type& value) {
      if(range().volume() != 0)
        for(iterator it = begin(); it != end(); ++it)
          *it *= value;

      return *this;
    }

    /// Resize the array to the specified dimensions.

    /// \param r The range object that specifies the new size.
    /// \param val The value that will fill any new elements in the array
    /// ( default: value_type() ).
    /// \return A reference to this object.
    /// \note The current data common to both arrays is maintained.
    /// \note This function cannot change the number of tile dimensions.
    Tile_& resize(const range_type& r, value_type val = value_type()) {
      Tile_ temp(r, val);
      if(base::data()) {
        // replace Range with ArrayDim?
        range_type range_common = r & (range_);

        for(typename range_type::const_iterator it = range_common.begin(); it != range_common.end(); ++it)
          temp[ *it ] = operator[]( *it ); // copy common data.
      }
      swap(temp);
      return *this;
    }

    using base::data;
    using base::begin;
    using base::end;

    /// Returns a reference to element i (range checking is performed).

    /// This function provides element access to the element located at index i.
    /// If i is not included in the range of elements, std::out_of_range will be
    /// thrown. Valid types for Index are ordinal_type and index_type.
    template <typename Index>
    reference at(const Index& i) { return base::at(range_.ord(i)); }

    /// Returns a constant reference to element i (range checking is performed).

    /// This function provides element access to the element located at index i.
    /// If i is not included in the range of elements, std::out_of_range will be
    /// thrown. Valid types for Index are ordinal_type and index_type.
    template <typename Index>
    const_reference at(const Index& i) const { return base::at(range_.ord(i)); }

    /// Returns a reference to the element at i.

    /// This No error checking is performed.
    template <typename Index>
    reference operator[](const Index& i) { return base::operator[](range_.ord(i)); }

    /// Returns a constant reference to element i. No error checking is performed.
    template <typename Index>
    const_reference operator[](const Index& i) const { return base::operator[](range_.ord(i)); }

    /// Tile range accessor

    /// \return A const reference to the tile range object.
    /// \throw nothing
    const range_type& range() const { return range_; }

    static unsigned int dim() { return coordinate_system::dim; }
    static detail::DimensionOrderType order() { return coordinate_system::order; }

    /// Exchange the content of this object with other.

    /// \param other The other Tile to swap with this object
    /// \throw nothing
    void swap(Tile_& other) {
      base::swap(other);
      std::swap(range_, other.range_);
    }

  protected:

    template <typename Archive>
    void load(const Archive& ar) {
      base::load(ar);
      ar & range_;
    }

    template <typename Archive>
    void store(const Archive& ar) const {
      base::store(ar);
      ar & range_;
    }

  private:

    friend Tile_ operator ^ <>(const Permutation<coordinate_system::dim>&, const Tile_&);
    template <class, class>
    friend struct madness::archive::ArchiveStoreImpl;
    template <class, class>
    friend struct madness::archive::ArchiveLoadImpl;

    range_type range_;  ///< Range data for this tile
  }; // class Tile


  /// Swap the data of the two arrays.

  /// \tparam T Tile element type
  /// \tparam CS Tile coordinate system type
  /// \tparam A Tile allocator
  /// \param first The first tile to swap
  /// \param second The second tile to swap
  template <typename T, typename CS, typename A>
  void swap(Tile<T, CS, A>& first, Tile<T, CS, A>& second) { // no throw
    first.swap(second);
  }

  /// Permutes the content of the n-dimensional array.

  /// \tparam T Tile element type
  /// \tparam CS Tile coordinate system type
  /// \tparam A Tile allocator
  template <typename T, typename CS, typename A>
  Tile<T,CS,A> operator ^(const Permutation<CS::dim>& p, const Tile<T,CS,A>& t) {

    typename Tile<T,CS,A>::range_type r(p ^ t.range());
    Tile<T,CS,A> result(r);

    // create a permuted copy of the tile data
    detail::permute_tensor<CS>(p, t.range().size(), t, result);

    return result;
  }

  /// Tile addition operator

  /// Add the elements of two tiles together.
  /// \tparam T Tile element type
  /// \tparam CS Tile coordinate system type
  /// \tparam A Tile allocator
  /// \param left The left-hand, tile argument
  /// \param right The right-hand, tile argument
  /// \return A new tile where: \c result[i] \c == \c left[i] \c + \c right[i]
  /// \note The range of the two tiles must be equivalent
  template <typename T, typename CS, typename A>
  inline Tile<T, CS, A> operator+(const Tile<T, CS, A>& left, const Tile<T, CS, A>& right) {
    if(left.range().volume() && right.range().volume()) {
      TA_ASSERT(left.range() == right.range(), std::range_error, "Tile range must be equal.");
      return Tile<T, CS, A>(left.range(), detail::make_tran_it(left.begin(), right.begin(),
          std::plus<typename Tile<T, CS, A>::value_type>()));
    }

    return (left.range().volume() ? left : right);
  }

  /// Tile subtraction operator

  /// Subtract the elements of two tiles together.
  /// \tparam T Tile element type
  /// \tparam CS Tile coordinate system type
  /// \tparam A Tile allocator
  /// \param left The left-hand, tile argument
  /// \param right The right-hand, tile argument
  /// \return A new tile where: \c result[i] \c == \c left[i] \c - \c right[i]
  /// \note The range of the two tiles must be equivalent
  template <typename T, typename CS, typename A>
  inline Tile<T, CS, A> operator-(const Tile<T, CS, A>& left, const Tile<T, CS, A>& right) {
    if(left.range().volume() && right.range().volume()) {
      TA_ASSERT(left.range() == right.range(), std::range_error, "Tile range must be equal.");
      return Tile<T, CS, A>(left.range(), detail::make_tran_it(left.begin(), right.begin(),
          std::minus<typename Tile<T, CS, A>::value_type>()));
    }

    return (left.range().volume() ? left : -right);
  }

  /// Tile negation operator

  /// Negate each element of the tile.
  /// \tparam T Tile element type
  /// \tparam CS Tile coordinate system type
  /// \tparam A Tile allocator
  /// \param arg The tile argument
  /// \return A new tile where: \c result[i] \c == \c -arg[i]
  template <typename T, typename CS, typename A>
  inline Tile<T, CS, A> operator-(Tile<T, CS, A> arg) {
    return Tile<T, CS, A>(arg.range(), detail::make_tran_it(arg.begin(),
        std::negate<typename Tile<T, CS, A>::value_type>()));
  }


  /// Tile scalar addition operator

  /// Add a scalar value to each element of a tile.
  /// \tparam T Tile element type
  /// \tparam CS Tile coordinate system type
  /// \tparam A Tile allocator
  /// \param left The left-hand, scalar argument
  /// \param right The right-hand, tile argument
  /// \return A new tile where: \c result[i] \c == \c left \c + \c right[i]
  template <typename T, typename CS, typename A>
  inline Tile<T, CS, A> operator+(const typename Tile<T, CS, A>::value_type& left, const Tile<T, CS, A>& right) {
    return Tile<T, CS, A>(right.range(), detail::make_tran_it(right.begin(),
        std::bind1st(std::plus<typename Tile<T, CS, A>::value_type>(), left)));
  }

  /// Tile scalar addition operator

  /// Add a scalar value to each element of a tile.
  /// \tparam T Tile element type
  /// \tparam CS Tile coordinate system type
  /// \tparam A Tile allocator
  /// \param left The left-hand, tile argument
  /// \param right The right-hand, scalar argument
  /// \return A new tile where: \c result[i] \c == \c left[i] \c + \c right
  template <typename T, typename CS, typename A>
  inline Tile<T, CS, A> operator+(const Tile<T, CS, A>& left, const typename Tile<T, CS, A>::value_type& right) {
    return Tile<T, CS, A>(left.range(), detail::make_tran_it(left.begin(),
        std::bind2nd(std::plus<typename Tile<T, CS, A>::value_type>(), right)));
  }

  /// Tile scalar subtraction operator

  /// Subtract a scalar value to each element of a tile.
  /// \tparam T Tile element type
  /// \tparam CS Tile coordinate system type
  /// \tparam A Tile allocator
  /// \param left The left-hand, scalar argument
  /// \param right The right-hand, tile argument
  /// \return A new tile where: \c result[i] \c == \c left \c - \c right[i]
  template <typename T, typename CS, typename A>
  inline Tile<T, CS, A> operator-(const typename Tile<T, CS, A>::value_type& left, const Tile<T, CS, A>& right) {
    return Tile<T, CS, A>(right.range(), detail::make_tran_it(right.begin(),
        std::bind1st(std::minus<typename Tile<T, CS, A>::value_type>(), left)));
  }

  /// Tile scalar subtraction operator

  /// Subtract a scalar value to each element of a tile.
  /// \tparam T Tile element type
  /// \tparam CS Tile coordinate system type
  /// \tparam A Tile allocator
  /// \param left The left-hand, tile argument
  /// \param right The right-hand, scalar argument
  /// \return A new tile where: \c result[i] \c == \c left[i] \c - \c right
  template <typename T, typename CS, typename A>
  inline Tile<T, CS, A> operator-(const Tile<T, CS, A>& left, const typename Tile<T, CS, A>::value_type& right) {
    return Tile<T, CS, A>(left.range(), detail::make_tran_it(left.begin(),
        std::bind2nd(std::minus<typename Tile<T, CS, A>::value_type>(), right)));
  }

  /// Tile scale operator

  /// Scale the elements of the tile by the given scalar value.
  /// \tparam T Tile element type
  /// \tparam CS Tile coordinate system type
  /// \tparam A Tile allocator
  /// \param left The left-hand, scalar argument
  /// \param right The right-hand, tile argument
  /// \return A new tile where: \c result[i] \c == \c left \c - \c right[i]
  template <typename T, typename CS, typename A>
  inline Tile<T, CS, A> operator*(const typename Tile<T, CS, A>::value_type& left, const Tile<T, CS, A>& right) {
    return Tile<T, CS, A>(right.range(), detail::make_tran_it(right.begin(),
        std::bind1st(std::multiplies<typename Tile<T, CS, A>::value_type>(), left)));
  }

  /// Tile scale operator

  /// Scale the elements of the tile by the given scalar value.
  /// \tparam T Tile element type
  /// \tparam CS Tile coordinate system type
  /// \tparam A Tile allocator
  /// \param left The left-hand, tile argument
  /// \param right The right-hand, scalar argument
  /// \return A new tile where: \c result[i] \c == \c left[i] \c + \c right
  template <typename T, typename CS, typename A>
  inline Tile<T, CS, A> operator*(const Tile<T, CS, A>& left, const typename Tile<T, CS, A>::value_type& right) {
    return Tile<T, CS, A>(left.range(), detail::make_tran_it(left.begin(),
        std::bind2nd(std::multiplies<typename Tile<T, CS, A>::value_type>(), right)));
  }

  /// ostream output operator.

  /// \tparam T Tile element type
  /// \tparam CS Tile coordinate system type
  /// \tparam A Tile allocator
  /// \param out The output stream that will hold the tile output.
  /// \param t The tile to be place in the output stream.
  /// \return The modified \c out .
  template <typename T, typename CS, typename A>
  std::ostream& operator <<(std::ostream& out, const Tile<T, CS, A>& t) {
    typedef Tile<T, CS, A> tile_type;
    typedef typename detail::CoordIterator<const typename tile_type::size_array,
        tile_type::coordinate_system::order>::iterator weight_iterator;

    typename tile_type::ordinal_index i = 0;
    weight_iterator weight_begin_1 = tile_type::coordinate_system::begin(t.range().weight()) + 1;
    weight_iterator weight_end = tile_type::coordinate_system::end(t.range().weight());
    weight_iterator weight_it;

    out << "{";
    for(typename tile_type::const_iterator it = t.begin(); it != t.end(); ++it, ++i) {
      for(weight_it = weight_begin_1; weight_it != weight_end; ++weight_it) {
        if((i % *weight_it) == 0)
          out << "{";
      }

      out << *it << " ";

      for(weight_it = weight_begin_1; weight_it != weight_end; ++weight_it) {
        if(((i + 1) % *weight_it) == 0)
          out << "}";
      }
    }
    out << "}";
    return out;
  }

} // namespace TiledArray

namespace madness {
  namespace archive {

    template <class Archive, class T>
    struct ArchiveStoreImpl;
    template <class Archive, class T>
    struct ArchiveLoadImpl;

    template <class Archive, typename T, typename CS, typename A>
    struct ArchiveStoreImpl<Archive, TiledArray::Tile<T, CS, A> > {
      static void store(const Archive& ar, const TiledArray::Tile<T, CS, A>& t) {
        t.store(ar);
      }
    };

    template <class Archive, typename T, typename CS, typename A>
    struct ArchiveLoadImpl<Archive, TiledArray::Tile<T, CS, A> > {
      typedef TiledArray::Tile<T, CS, A> tile_type;

      static void load(const Archive& ar, tile_type& t) {
        t.load(ar);
      }
    };
  }
}
#endif // TILEDARRAY_TILE_H__INCLUDED
