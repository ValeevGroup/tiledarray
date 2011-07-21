/*
 * tile_proxy.h
 *
 *  Created on: Jul 18, 2011
 *      Author: justus
 */

#ifndef TILEDARRAY_TILE_PROXY_H__INCLUDED
#define TILEDARRAY_TILE_PROXY_H__INCULDED

#include <TiledArray/error.h>
#include <TiledArray/tile_base.h>
#include <TiledArray/transform_iterator.h>
#include <TiledArray/coordinate_system.h>
#include <TiledArray/perm_algorithm.h>
#include <vector>
#include <numeric>
#include <Eigen/Core>

namespace TiledArray {

  /// TileProxy is an N-dimensional, dense array.

  /// \tparam T TileProxy element type.
  /// \tparam CS A \c CoordinateSystem type
  /// \tparam A A C++ standard library compliant allocator (Default:
  /// \c Eigen::aligned_allocator<T>)
  template <typename T, typename A = Eigen::aligned_allocator<T> >
  class TileProxy : public TileBase<T,A> {
  private:
    typedef TileBase<T,A> base;

  public:
    typedef TileProxy<T,A> TileProxy_;                               ///< This object's type

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

    typedef size_type ordinal_index;       ///< Array ordinal index type
    typedef std::vector<size_type> size_array;             ///< Size array type

  public:

    /// Default constructor

    /// Constructs a tile with zero size.
    /// \note You must call resize() before attempting to access any elements.
    TileProxy() :
        base(), size_(), order_(detail::decreasing_dimension_order)
    { }

    /// Copy constructor

    /// \param other The tile to be copied.
    TileProxy(const TileProxy_& other) :
        base(other), size_(other.size_), order_(other.order_)
    { }

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
    template <typename Size>
    TileProxy(const Size& s, const value_type& val = value_type(),
      detail::DimensionOrderType o = detail::decreasing_dimension_order, const allocator_type& a = allocator_type()) :
        base(volume(s), val, a), size_(s.begin(), s.end()), order_(o)
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
    template <typename Size, typename InIter>
    TileProxy(const Size& s, InIter first, detail::DimensionOrderType o = detail::decreasing_dimension_order,
      const allocator_type& a = allocator_type()) :
        base(volume(s), first, a),
        size_(s.begin(), s.end()),
        order_(o)
    { }

    /// Assignment operator

    /// \param other The tile object to be moved
    /// \return A reference to this object
    /// \throw std::bad_alloc There is not enough memory available for the target tile
    TileProxy_& operator =(const TileProxy_& other) {
      TileProxy_(other).swap(*this);
      return *this;
    }

    /// Assignment operator

    /// \param other The tile object to be moved
    /// \return A reference to this object
    /// \throw std::bad_alloc There is not enough memory available for the target tile
//    TileProxy_& operator =(const ArrayMove& a) {
//      TileProxy_ temp;.swap(*this);
//      return *this;
//    }

    /// destructor
    ~TileProxy() { }

    /// In-place permutation of tile elements.

    /// \param p A permutation object.
    /// \return A reference to this object
    /// \warning This function modifies the shared range object.
    template <unsigned int DIM>
    TileProxy_& operator ^=(const Permutation<DIM>& p) {
      TA_ASSERT(dim() == DIM, std::runtime_error,
          "Permutation dimensions must match tile dimensions.");

      // create a permuted copy of the tile data
      base temp(size_.volume());
      detail::permute_tensor(order_, p, size_, *this, temp);

      // Take ownership of permuted data
      size_ ^= p;
      temp.swap(*this);

      return *this;
    }

    TileProxy_& operator+=(const TileProxy_& other) {
      if(base::size() == 0ul) {
        *this = other;
      } else if(static_cast<const base&>(other).size() != 0ul) {
        TA_ASSERT(size() == other.size(), std::runtime_error, "The ranges must be equal.");
        const_iterator other_it = other.begin();
        for(iterator it = begin(); it != end(); ++it)
          *it += *other_it++;
      }

      return *this;
    }

    TileProxy_& operator+=(const value_type& value) {
      if(base::size() != 0)
        for(iterator it = begin(); it != end(); ++it)
          *it += value;

      return *this;
    }

    TileProxy_& operator-=(const TileProxy_& other) {
      if(base::size() == 0)
        *this = -other;
      else if(static_cast<const base&>(other).size() != 0ul) {
        TA_ASSERT(size() == other.size(), std::runtime_error, "The ranges must be equal.");
        const_iterator other_it = other.begin();
        for(iterator it = begin(); it != end(); ++it)
          *it -= *other_it++;
      }

      return *this;
    }

    TileProxy_& operator-=(const value_type& value) {
      if(base::size() != 0)
        for(iterator it = begin(); it != end(); ++it)
          *it -= value;

      return *this;
    }

    TileProxy_& operator*=(const value_type& value) {
      if(base::size() != 0)
        for(iterator it = begin(); it != end(); ++it)
          *it *= value;

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
    reference at(const Index& i) { return base::at(size_.ord(i)); }

    /// Returns a constant reference to element i (range checking is performed).

    /// This function provides element access to the element located at index i.
    /// If i is not included in the range of elements, std::out_of_range will be
    /// thrown. Valid types for Index are ordinal_type and index_type.
    template <typename Index>
    const_reference at(const Index& i) const { return base::at(size_.ord(i)); }

    /// Returns a reference to the element at i.

    /// This No error checking is performed.
    template <typename Index>
    reference operator[](const Index& i) { return base::operator[](size_.ord(i)); }

    /// Returns a constant reference to element i. No error checking is performed.
    template <typename Index>
    const_reference operator[](const Index& i) const { return base::operator[](size_.ord(i)); }

    unsigned int dim() const { return size_.size(); }
    detail::DimensionOrderType order() { return order_; }
    const size_array& size() const { return size_; }
    size_type volume() const { return base::size(); }

    /// Exchange the content of this object with other.

    /// \param other The other TileProxy to swap with this object
    /// \throw nothing
    void swap(TileProxy_& other) {
      base::swap(other);
      std::swap(size_, other.range_);
    }

  protected:

    template <typename Archive>
    void load(const Archive& ar) {
      base::load(ar);
      ar & size_;
    }

    template <typename Archive>
    void store(const Archive& ar) const {
      base::store(ar);
      ar & size_;
    }

  private:

    template <typename Size>
    static size_type volume(const Size& s) {
      return std::accumulate(s.begin(), s.end(), 1ul, std::multiplies<unsigned long>());
    }

    template <class, class>
    friend struct madness::archive::ArchiveStoreImpl;
    template <class, class>
    friend struct madness::archive::ArchiveLoadImpl;

    size_array size_;  ///< Shared pointer to the range data for this tile
    detail::DimensionOrderType order_;
  }; // class TileProxy


  /// Swap the data of the two arrays.

  /// \tparam T TileProxy element type
  /// \tparam CS TileProxy coordinate system type
  /// \tparam A TileProxy allocator
  /// \param first The first tile to swap
  /// \param second The second tile to swap
  template <typename T, typename A>
  void swap(TileProxy<T, A>& first, TileProxy<T, A>& second) { // no throw
    first.swap(second);
  }

  /// Permutes the content of the n-dimensional array.

  /// \tparam T TileProxy element type
  /// \tparam CS TileProxy coordinate system type
  /// \tparam A TileProxy allocator
//  template <unsigned int DIM, typename T, typename A>
//  TileProxy<T,A> operator ^(const Permutation<DIM>& p, const TileProxy<T,A>& t) {
//    typedef detail::AssignmentOp<typename TileProxy<T,A>::iterator, typename TileProxy<T,A>::const_iterator> assign_op;
//
//    typename TileProxy<T,A>::range_type r(p ^ t.range());
//    TileProxy<T,A> result(r);
//
//    // create a permuted copy of the tile data
//    detail::Permute<CS, assign_op> f_perm(t.range(), assign_op(t.begin(), t.end()));
//    f_perm(p, result.begin(), result.end());
//
//    return result;
//  }

  /// TileProxy addition operator

  /// Add the elements of two tiles together.
  /// \tparam T TileProxy element type
  /// \tparam CS TileProxy coordinate system type
  /// \tparam A TileProxy allocator
  /// \param left The left-hand, tile argument
  /// \param right The right-hand, tile argument
  /// \return A new tile where: \c result[i] \c == \c left[i] \c + \c right[i]
  /// \note The range of the two tiles must be equivalent
  template <typename T, typename A>
  inline TileProxy<T, A> operator+(const TileProxy<T, A>& left, const TileProxy<T, A>& right) {
    if(left.volume() && right.volume()) {
      TA_ASSERT(left.size() == right.size(), std::range_error, "TileProxy range must be equal.");
      return TileProxy<T, A>(left.size(), detail::make_tran_it(left.begin(), right.begin(),
          std::plus<typename TileProxy<T, A>::value_type>()));
    }

    return (left.volume() ? left : right);
  }

  /// TileProxy subtraction operator

  /// Subtract the elements of two tiles together.
  /// \tparam T TileProxy element type
  /// \tparam CS TileProxy coordinate system type
  /// \tparam A TileProxy allocator
  /// \param left The left-hand, tile argument
  /// \param right The right-hand, tile argument
  /// \return A new tile where: \c result[i] \c == \c left[i] \c - \c right[i]
  /// \note The range of the two tiles must be equivalent
  template <typename T, typename A>
  inline TileProxy<T, A> operator-(const TileProxy<T, A>& left, const TileProxy<T, A>& right) {
    if(left.volume() && right.volume()) {
      TA_ASSERT(left.size() == right.size(), std::range_error, "TileProxy range must be equal.");
      return TileProxy<T, A>(left.size(), detail::make_tran_it(left.begin(), right.begin(),
          std::minus<typename TileProxy<T, A>::value_type>()));
    }

    return (left.volume() ? left : -right);
  }

  /// TileProxy negation operator

  /// Negate each element of the tile.
  /// \tparam T TileProxy element type
  /// \tparam CS TileProxy coordinate system type
  /// \tparam A TileProxy allocator
  /// \param arg The tile argument
  /// \return A new tile where: \c result[i] \c == \c -arg[i]
  template <typename T, typename A>
  inline TileProxy<T, A> operator-(TileProxy<T, A> arg) {
    return TileProxy<T, A>(arg.size(), detail::make_tran_it(arg.begin(),
        std::negate<typename TileProxy<T, A>::value_type>()));
  }


  /// TileProxy scalar addition operator

  /// Add a scalar value to each element of a tile.
  /// \tparam T TileProxy element type
  /// \tparam CS TileProxy coordinate system type
  /// \tparam A TileProxy allocator
  /// \param left The left-hand, scalar argument
  /// \param right The right-hand, tile argument
  /// \return A new tile where: \c result[i] \c == \c left \c + \c right[i]
  template <typename T, typename A>
  inline TileProxy<T, A> operator+(const typename TileProxy<T, A>::value_type& left, const TileProxy<T, A>& right) {
    return TileProxy<T, A>(right.size(), detail::make_tran_it(right.begin(),
        std::bind1st(std::plus<typename TileProxy<T, A>::value_type>(), left)));
  }

  /// TileProxy scalar addition operator

  /// Add a scalar value to each element of a tile.
  /// \tparam T TileProxy element type
  /// \tparam CS TileProxy coordinate system type
  /// \tparam A TileProxy allocator
  /// \param left The left-hand, tile argument
  /// \param right The right-hand, scalar argument
  /// \return A new tile where: \c result[i] \c == \c left[i] \c + \c right
  template <typename T, typename CS, typename A>
  inline TileProxy<T, A> operator+(const TileProxy<T, A>& left, const typename TileProxy<T, A>::value_type& right) {
    return TileProxy<T, A>(left.size(), detail::make_tran_it(left.begin(),
        std::bind2nd(std::plus<typename TileProxy<T, A>::value_type>(), right)));
  }

  /// TileProxy scalar subtraction operator

  /// Subtract a scalar value to each element of a tile.
  /// \tparam T TileProxy element type
  /// \tparam CS TileProxy coordinate system type
  /// \tparam A TileProxy allocator
  /// \param left The left-hand, scalar argument
  /// \param right The right-hand, tile argument
  /// \return A new tile where: \c result[i] \c == \c left \c - \c right[i]
  template <typename T, typename A>
  inline TileProxy<T, A> operator-(const typename TileProxy<T, A>::value_type& left, const TileProxy<T, A>& right) {
    return TileProxy<T, A>(right.size(), detail::make_tran_it(right.begin(),
        std::bind1st(std::minus<typename TileProxy<T, A>::value_type>(), left)));
  }

  /// TileProxy scalar subtraction operator

  /// Subtract a scalar value to each element of a tile.
  /// \tparam T TileProxy element type
  /// \tparam CS TileProxy coordinate system type
  /// \tparam A TileProxy allocator
  /// \param left The left-hand, tile argument
  /// \param right The right-hand, scalar argument
  /// \return A new tile where: \c result[i] \c == \c left[i] \c - \c right
  template <typename T, typename A>
  inline TileProxy<T, A> operator-(const TileProxy<T, A>& left, const typename TileProxy<T, A>::value_type& right) {
    return TileProxy<T, A>(left.size(), detail::make_tran_it(left.begin(),
        std::bind2nd(std::minus<typename TileProxy<T, A>::value_type>(), right)));
  }

  /// TileProxy scale operator

  /// Scale the elements of the tile by the given scalar value.
  /// \tparam T TileProxy element type
  /// \tparam CS TileProxy coordinate system type
  /// \tparam A TileProxy allocator
  /// \param left The left-hand, scalar argument
  /// \param right The right-hand, tile argument
  /// \return A new tile where: \c result[i] \c == \c left \c - \c right[i]
  template <typename T, typename A>
  inline TileProxy<T, A> operator*(const typename TileProxy<T, A>::value_type& left, const TileProxy<T, A>& right) {
    return TileProxy<T, A>(right.size(), detail::make_tran_it(right.begin(),
        std::bind1st(std::multiplies<typename TileProxy<T, A>::value_type>(), left)));
  }

  /// TileProxy scale operator

  /// Scale the elements of the tile by the given scalar value.
  /// \tparam T TileProxy element type
  /// \tparam CS TileProxy coordinate system type
  /// \tparam A TileProxy allocator
  /// \param left The left-hand, tile argument
  /// \param right The right-hand, scalar argument
  /// \return A new tile where: \c result[i] \c == \c left[i] \c + \c right
  template <typename T, typename A>
  inline TileProxy<T, A> operator*(const TileProxy<T, A>& left, const typename TileProxy<T, A>::value_type& right) {
    return TileProxy<T, A>(left.size(), detail::make_tran_it(left.begin(),
        std::bind2nd(std::multiplies<typename TileProxy<T, A>::value_type>(), right)));
  }


}  // namespace TiledArray


#endif // TILE_PROXY_H_
