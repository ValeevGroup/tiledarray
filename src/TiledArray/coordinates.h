#ifndef TILEDARRAY_COORDINATES_H__INCLUDED
#define TILEDARRAY_COORDINATES_H__INCLUDED

#include <TiledArray/array_util.h>
#include <TiledArray/utility.h>
#include <boost/operators.hpp>
#include <boost/array.hpp>
#include <boost/type_traits/is_integral.hpp>
#include <boost/utility/enable_if.hpp>
#include <boost/functional/hash.hpp>
#include <iosfwd>
#include <cstddef>
#ifndef __GXX_EXPERIMENTAL_CXX0X__
#include <stdarg.h>
#endif // __GXX_EXPERIMENTAL_CXX0X__

  // Forward declarations.
namespace boost {
  template <typename T, std::size_t D>
  std::ostream& operator<<(std::ostream&, const boost::array<T,D>&);
} // namespace boost

namespace TiledArray {
  // Forward declarations.
  template <unsigned int DIM>
  class Permutation;
  namespace detail {
    template <typename InIter0, typename InIter1, typename RandIter>
    void permute(InIter0, InIter0, InIter1, RandIter);
  }  // namespace detail


  /// ArrayCoordinate represents a coordinate index of an DIM-dimensional orthogonal tensor.

  /// \tparam I The index type of each coordinate element.
  /// \tparam DIM The number of dimensions in the coordinate
  /// \tparam Tag Type used to differentiate different coordinate systems.
  template <typename I, unsigned int DIM, typename Tag>
  class ArrayCoordinate : public
      boost::addable< ArrayCoordinate<I,DIM,Tag>,                // point + point
      boost::subtractable< ArrayCoordinate<I,DIM,Tag>,           // point - point
      boost::less_than_comparable1< ArrayCoordinate<I,DIM,Tag>,  // point < point
      boost::equality_comparable1< ArrayCoordinate<I,DIM,Tag>    // point == point
      > > > >
  {
  private:
    typedef boost::array<I,DIM> array_type; ///< array_type type used to store coordinates

    struct Enabler { };

  public:
    BOOST_STATIC_ASSERT(boost::is_integral<I>::value);

    typedef ArrayCoordinate<I,DIM,Tag> ArrayCoordinate_;                        ///< This type
    typedef I index;                                                            ///< Coordinate element type
    typedef typename array_type::iterator iterator;                             ///< Coordinate element iterator
    typedef typename array_type::const_iterator const_iterator;                 ///< Coordinate element const iterator
    typedef typename array_type::reverse_iterator reverse_iterator;             ///< Coordinate element reverse iterator
    typedef typename array_type::const_reverse_iterator const_reverse_iterator; ///< Coordinate element const reverse iterator

    /// Default constructor

    /// All coordinate elements are initialized to 0.
    ArrayCoordinate() { r_.assign(index(0)); }

    /// Initialize coordinate with an iterator

    /// \tparam InIter Input iterator type (InIter::value_type must be an
    /// integral type)
    /// \param first An iterator that points to the beginning of a list elements
    /// used to initialize the coordinate elements
    /// \throw nothing
    template <typename InIter>
    explicit ArrayCoordinate(InIter first, typename boost::disable_if<boost::is_integral<InIter>, Enabler >::type = Enabler()) {
      // should variadic constructor been chosen?
      // need to disambiguate the call if DIM==1
      // assume iterators if InIter is not an integral type
      // else assume wanted variadic constructor
      // this scheme follows what std::vector does
      detail::initialize_from_values(first, r_.begin(), DIM, boost::is_integral<InIter>());
    }

    /// Copy the content of the boost::array<I, DIM> object into the coordinate elements.

    /// \throw nothing
    ArrayCoordinate(const array_type& init_values) : r_(init_values) { }

    /// Copy constructor

    /// \throw nothing
    ArrayCoordinate(const ArrayCoordinate& a) : r_(a.r_) { }
#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// Move constructor

    /// \throw nothing
    ArrayCoordinate(ArrayCoordinate&& a) : r_(std::move(a.r_)) { }
#endif // __GXX_EXPERIMENTAL_CXX0X__

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// Constant list constructor constructor.

    /// This constructor takes a list of integral-type values and initializes
    /// the elements of the coordinate with them. The number of parameters
    /// passed to this constructor must be exactly equal to DIM.
    /// \tparam ...Params The variadic parameter list (All parameters types must
    /// be integral types)
    /// \param params The list of parameters to be used to initialize the
    /// coordinate
    /// \throw nothing
    template <typename... Params>
    explicit ArrayCoordinate(Params... params) {
      BOOST_STATIC_ASSERT(detail::Count<Params...>::value == DIM);
      BOOST_STATIC_ASSERT(detail::is_integral_list<Params...>::value);
      detail::fill(r_.begin(), params...);
    }
#else
    /// Constant index constructor.

    /// Constructs an ArrayCoordinate with the specified constants. For example,
    /// ArrayCoordinate<std::size_t, 4, ...> p(0, 1, 2, 3); would construct a
    /// point with the coordinates (0, 1, 2, 3).
    explicit ArrayCoordinate(const index c0, const index c1, ...) {
      r_.assign(0ul);
      va_list ap;
      va_start(ap, c1);

      r_[0] = c0;
      r_[1] = c1;
      unsigned int ci = 0; // ci is used as an intermediate
      for(unsigned int i = 2; i < DIM; ++i) {
        ci = va_arg(ap, index);
        r_[i] = ci;
      }

      va_end(ap);
    }

#endif // __GXX_EXPERIMENTAL_CXX0X__

    /// Begin iterator factory

    /// \return An iterator to the first coordinate element
    /// \throw nothing
    iterator begin() { return r_.begin(); }

    /// Begin const iterator factory

    /// \return A const iterator to the first coordinate element
    /// \throw nothing
    const_iterator begin() const { return r_.begin(); }

    /// End iterator factory

    /// \return An iterator pointing to one past the end of the coordinate
    /// elements.
    /// \throw nothing
    iterator end() { return r_.end(); }

    /// End const iterator factory

    /// \return A const iterator pointing to one past the end of the coordinate
    /// elements.
    /// \throw nothing
    const_iterator end() const { return r_.end(); }

    /// Reverse begin iterator factory

    /// \return An iterator pointing to one element before the first coordinate
    /// elements.
    reverse_iterator rbegin() { return r_.rbegin(); }

    /// Reverse begin const iterator factory

    /// \return A const iterator pointing to one element before the first
    /// coordinate elements.
    const_reverse_iterator rbegin() const { return r_.rbegin(); }

    /// Reverse end iterator factory

    /// \return An iterator to the last coordinate element
    reverse_iterator rend() { return r_.rend(); }

    /// Reverse end const iterator factory

    /// \return A const iterator to the last coordinate element
    const_reverse_iterator rend() const { return r_.rend(); }

    /// Assignment operator

    /// \param other The other coordinate to be copied
    /// \return A reference to this object
    /// \throw nothing
    ArrayCoordinate_&
    operator =(const ArrayCoordinate_& other) {
      r_ = other.r_;

      return (*this);
    }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// Move assignment operator

    /// \param other The other coordinate to be moved
    /// \return A reference to this object
    /// \throw nothing
    ArrayCoordinate_&
    operator =(ArrayCoordinate_&& other) {
      r_ = std::move(other.r_);

      return (*this);
    }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    /// Addition-assignment operator

    /// \param other The coordinate to be added to this coordinate
    /// \return A reference to this object
    /// \throw nothing
    ArrayCoordinate_& operator+=(const ArrayCoordinate_& other) {
      const_iterator other_it = other.begin();
      for(iterator it = r_.begin(); it != r_.end(); ++it, ++other_it)
        *it += *other_it;
      return *this;
    }

    /// Subtract-assignement operator

    /// \param other The coordinate to be subtracted from this coordinate
    /// \return A reference to this object
    /// \throw nothing
    ArrayCoordinate_ operator-=(const ArrayCoordinate_& other) {
      const_iterator other_it = other.begin();
      for(iterator it = r_.begin(); it != r_.end(); ++it, ++other_it)
        *it -= *other_it;
      return *this;
    }

    /// Coordinate element const accessor

    /// \param n The element to access
    /// \return A const reference to element \c n
    /// \throw std::out_of_range When NDEBUG is defined and \c n \c >= \c DIM
    const index& operator[](std::size_t n) const {
#ifdef NDEBUG
      return r_[n];
#else
      return r_.at(n);
#endif
    }

    /// Coordinate element accessor

    /// \param n The element to access
    /// \return A reference to element \c n
    /// \throw std::out_of_range When NDEBUG is defined and \c n \c >= \c DIM
    index& operator[](std::size_t d) {
#ifdef NDEBUG
      return r_[d];
#else
      return r_.at(d);
#endif
    }

    /// Coordinate element const accessor

    /// \param n The element to access
    /// \return A const reference to element \c n
    /// \throw std::out_of_range When \c n \c >= \c DIM
    const index& at(std::size_t d) const {  return r_.at(d);  }

    /// Coordinate element accessor

    /// \param n The element to access
    /// \return A reference to element \c n
    /// \throw std::out_of_range When \c n \c >= \c DIM
    index& at(std::size_t d) { return r_.at(d); }

    /// Coordinate array const accessor

    /// \return A const reference to the coordinate data array
    /// \throw nothing
    const array_type& data() const { return r_; }

    /// Coordinate array accessor

    /// \return A reference to the coordinate data array
    /// \throw nothing
    array_type& data() { return r_; }

    /// Coordinate permutation operator

    /// Permute this coordinate.
    /// \param p Permutation object
    /// \return A reference to this object
    ArrayCoordinate_& operator ^= (const Permutation<DIM>& p) {
      array_type temp;
      detail::permute(p.begin(), p.end(), r_.begin(), temp.begin());
      boost::swap(r_, temp);
      return *this;
    }

    /// Serialize coordinate data

    /// Serialized the coordinate data into archive \c ar.
    /// \tparam Archive The serialization archive type
    /// \tparam ar The serialization archive object
    template <typename Archive>
    void serialize(const Archive& ar) {
      ar & r_;
    }

  private:
    array_type r_; ///< The coordinate data array
  };

  /// Swap the data of \c c1 with \c c2.

  /// \tparam I The coordinate index type
  /// \tparam DIM the coordinate dimension
  /// \tparam Tag The coordinate system tag
  /// \param c1 The first coordinate to be swaped
  /// \param c2 The second coordinate to be swaped
  /// \throw nothing
  template <typename I, unsigned int DIM, typename Tag>
  void swap(ArrayCoordinate<I,DIM,Tag>& c1, ArrayCoordinate<I,DIM,Tag>& c2) { // no throw
    boost::swap(c1.data(), c2.data());
  }

  /// Add a constant to a coordinate

  /// \tparam I The coordinate index type
  /// \tparam DIM the coordinate dimension
  /// \tparam Tag The coordinate system tag
  /// \param c The coordinate to be modified
  /// \param s The constant to be added to each element of the coordinate
  /// \return A reference to the modified coordinate \c c
  /// \throw nothing
  template <typename I, unsigned int DIM, typename Tag>
  ArrayCoordinate<I,DIM,Tag>& operator +=(ArrayCoordinate<I,DIM,Tag>& c, const I& s) {
    for(typename ArrayCoordinate<I,DIM,Tag>::iterator it = c.begin(); it != c.end(); ++it)
      *it += s;
    return c;
  }

  /// Add a constant to a coordinate

  /// \tparam I The coordinate index type
  /// \tparam DIM the coordinate dimension
  /// \tparam Tag The coordinate system tag
  /// \param c The coordinate to be modified
  /// \param s The constant to be added to each element of the coordinate
  /// \return A copy of coordinate \c c with the constant \c s added to it
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  ArrayCoordinate<I,DIM,Tag> operator +(const ArrayCoordinate<I,DIM,Tag>& c, const I& s) {
    ArrayCoordinate<I,DIM,Tag> result(c);
    for(typename ArrayCoordinate<I,DIM,Tag>::iterator it = result.begin(); it != result.end(); ++it)
      *it += s;
    return result;
  }

  /// Subtract a constant from a coordinate

  /// \tparam I The coordinate index type
  /// \tparam DIM the coordinate dimension
  /// \tparam Tag The coordinate system tag
  /// \param c The original coordinate
  /// \param s The constant to be subtracted from each element of the coordinate
  /// \return A reference to the modified coordinate \c c
  /// \throw nothing
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  ArrayCoordinate<I,DIM,Tag>& operator -=(ArrayCoordinate<I,DIM,Tag>& c, const I& s) {
    for(typename ArrayCoordinate<I,DIM,Tag>::iterator it = c.begin(); it != c.end(); ++it)
      *it -= s;
    return c;
  }

  /// Subtract a constant from a coordinate

  /// \tparam I The coordinate index type
  /// \tparam DIM the coordinate dimension
  /// \tparam Tag The coordinate system tag
  /// \param c The original coordinate
  /// \param s The constant to be added to each element of the coordinate
  /// \return A copy of coordinate \c c with the constant \c s subtracted from it
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  ArrayCoordinate<I,DIM,Tag> operator -(const ArrayCoordinate<I,DIM,Tag>& c, const I& s) {
    ArrayCoordinate<I,DIM,Tag> result(c);
    for(typename ArrayCoordinate<I,DIM,Tag>::iterator it = result.begin(); it != result.end(); ++it)
      *it -= s;
    return result;
  }

  /// Lexicographic comparison of two ArrayCoordinates.

  /// \return \c true when all elements of
  template <typename I, unsigned int DIM, typename Tag>
  bool operator<(const ArrayCoordinate<I,DIM,Tag>& c1, const ArrayCoordinate<I,DIM,Tag>& c2) {
    return c1.data() < c2.data();
  }

  template <typename I, unsigned int DIM, typename Tag>
  bool operator==(const ArrayCoordinate<I,DIM,Tag>& c1, const ArrayCoordinate<I,DIM,Tag>& c2) {
    return c1.data() == c2.data();
  }

  /// Permute an ArrayCoordinate
  template <typename I, unsigned int DIM, typename Tag>
  ArrayCoordinate<I,DIM,Tag> operator ^(const Permutation<DIM>& p, const ArrayCoordinate<I,DIM,Tag>& c) {
    ArrayCoordinate<I,DIM,Tag> result;
    detail::permute(p.begin(), p.end(), c.begin(), result.begin());
    return result;
  }

  /// Hash function for array coordinates

  /// \tparam I The array coordinate element type
  /// \tparam DIM The array coordinate dimensions
  /// \tparam Tag The array coordinate tag type
  /// \param c The array coordinate to hash
  template <typename I, unsigned int DIM, typename Tag>
  std::size_t hash_value(const ArrayCoordinate<I,DIM,Tag>& c) {
      boost::hash<boost::array<I,DIM> > hasher;
      return hasher(c.data());
  }

  /// Append an ArrayCoordinate to an output stream.
  template <typename I, unsigned int DIM, typename Tag>
  std::ostream& operator<<(std::ostream& output, const ArrayCoordinate<I,DIM,Tag>& c) {
    output << "(";
    detail::print_array(output, c.begin(), c.end());
    output << ")";
    return output;
  }

} // namespace TiledArray

// We need this operator in the boost namespace so it works properly in other
// name spaces, specifically in boost::test namespace.
namespace boost {
  /// Append a boost::array<T,D> to an output stream.
  template <typename T, std::size_t D>
  std::ostream& operator<<(std::ostream& output, const array<T,D>& a) {
    output << "{{";
    TiledArray::detail::print_array(output, a.begin(), a.end());
    output << "}}";
    return output;
  }
} // namespace boost

#endif // TILEDARRAY_COORDINATES_H__INCLUDED
