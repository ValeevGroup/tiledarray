#ifndef TILEDARRAY_COORDINATES_H__INCLUDED
#define TILEDARRAY_COORDINATES_H__INCLUDED

#include <TiledArray/config.h>
#include <TiledArray/error.h>
#include <TiledArray/permutation.h>
#include <boost/operators.hpp>
#include <world/array.h>
#include <boost/utility/enable_if.hpp>
#include <iosfwd>
#include <cstddef>
#include <stdarg.h>

namespace TiledArray {

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

    struct Enabler { };

  public:
    TA_STATIC_ASSERT(std::is_integral<I>::value);

    typedef ArrayCoordinate<I,DIM,Tag> ArrayCoordinate_;                        ///< This type
    typedef std::array<I,DIM> array_type;                                       ///< array_type type used to store coordinates
    typedef typename array_type::value_type value_type;                         ///< Coordinate element type
    typedef typename array_type::reference reference;
    typedef typename array_type::const_reference const_reference;
    typedef typename array_type::size_type size_type;
    typedef typename array_type::iterator iterator;                             ///< Coordinate element iterator
    typedef typename array_type::const_iterator const_iterator;                 ///< Coordinate element const iterator
    typedef typename array_type::reverse_iterator reverse_iterator;             ///< Coordinate element reverse iterator
    typedef typename array_type::const_reverse_iterator const_reverse_iterator; ///< Coordinate element const reverse iterator

    /// Default constructor

    /// All coordinate elements are initialized to 0.
    ArrayCoordinate() { std::fill(r_.begin(), r_.end(), value_type(0)); }

    /// Initialize coordinate with an iterator

    /// \tparam InIter Input iterator type (InIter::value_type must be an
    /// integral type)
    /// \param first An iterator that points to the beginning of a list elements
    /// used to initialize the coordinate elements
    /// \throw nothing
    template <typename InIter>
    explicit ArrayCoordinate(InIter first, typename boost::enable_if<detail::is_input_iterator<InIter>, Enabler >::type = Enabler()) {
      for(std::size_t d = 0; d < DIM; ++d, ++first)
        r_[d] = *first;
    }

    /// Copy the content of the std::array<I, DIM> object into the coordinate elements.

    /// \throw nothing
    ArrayCoordinate(const array_type& init_values) : r_(init_values) { }

    /// Copy constructor

    /// \throw nothing
    ArrayCoordinate(const ArrayCoordinate_& a) : r_(a.r_) { }

    /// Initialize coordinate with a given value

    /// All elements of the coordinate are initialized to the value of \c i
    /// \param i The value that will be assigned to all coordinate elements
    /// \throw nothing
    explicit ArrayCoordinate(value_type i) {
      std::fill_n(r_.begin(), DIM, i);
    }

    /// Constant index constructor.

    /// Constructs an ArrayCoordinate with the specified constants. For example,
    /// ArrayCoordinate<std::size_t, 4, ...> p(0, 1, 2, 3); would construct a
    /// point with the coordinates (0, 1, 2, 3).
    explicit ArrayCoordinate(const value_type c0, const value_type c1, ...) {
      std::fill_n(r_.begin(), DIM, 0ul);
      va_list ap;
      va_start(ap, c1);

      r_[0] = c0;
      r_[1] = c1;
      unsigned int ci = 0; // ci is used as an intermediate
      for(unsigned int i = 2; i < DIM; ++i) {
        ci = va_arg(ap, value_type);
        r_[i] = ci;
      }

      va_end(ap);
    }

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
    ArrayCoordinate_& operator =(const ArrayCoordinate_& other) {
      r_ = other.r_;

      return (*this);
    }

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
    ArrayCoordinate_& operator-=(const ArrayCoordinate_& other) {
      const_iterator other_it = other.begin();
      for(iterator it = r_.begin(); it != r_.end(); ++it, ++other_it)
        *it -= *other_it;
      return *this;
    }

    /// Coordinate element const accessor

    /// \param n The element to access
    /// \return A const reference to element \c n
    /// \throw std::out_of_range When NDEBUG is defined and \c n \c >= \c DIM
    const_reference operator[](std::size_t n) const {
      return r_[n];
    }

    /// Coordinate element accessor

    /// \param d The dimension index
    /// \return A reference to element \c n
    /// \throw std::out_of_range When NDEBUG is defined and \c n \c >= \c DIM
    reference operator[](std::size_t d) {
      return r_[d];
    }

    /// Coordinate element const accessor

    /// \param d The dimension index
    /// \return A const reference to element \c n
    /// \throw std::out_of_range When \c n \c >= \c DIM
    const_reference at(std::size_t d) const {
      TA_CHECK(d < DIM);
      return r_[d];
    }

    /// Coordinate element accessor

    /// \param d The dimension index
    /// \return A reference to element \c n
    /// \throw std::out_of_range When \c n \c >= \c DIM
    reference at(std::size_t d) {
      TA_CHECK(d < DIM);
      return r_[d];
    }

    /// Coordinate array const accessor

    /// \return A const reference to the coordinate data array
    /// \throw nothing
    const array_type& data() const { return r_; }

    /// Coordinate array accessor

    /// \return A reference to the coordinate data array
    /// \throw nothing
    array_type& data() { return r_; }

    /// Coordinate size (aka dimension)

    /// \return The number of dimensions in the coordinate
    size_type size() const { return r_.size(); }

    /// Coordinate permutation operator

    /// Permute this coordinate.
    /// \param p Permutation object
    /// \return A reference to this object
    ArrayCoordinate_& operator ^= (const Permutation<DIM>& p) {
      array_type temp;
      detail::permute_array(p.begin(), p.end(), r_.begin(), temp.begin());
      std::swap(r_, temp);
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
    std::swap(c1.data(), c2.data());
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
    detail::permute_array(p.begin(), p.end(), c.begin(), result.begin());
    return result;
  }

  namespace detail {
    template <typename A>
    void print_array(std::ostream& out, const A& a) {
      std::size_t n = a.size();
      out << "[";
      for(std::size_t i = 0; i < n; ++i) {
        out << a[i];
        if (i != (n - 1))
          out << ",";
      }
      out << "]";
    }
  } // namespace detail

  /// Append an ArrayCoordinate to an output stream.
  template <typename I, unsigned int DIM, typename Tag>
  std::ostream& operator<<(std::ostream& output, const ArrayCoordinate<I,DIM,Tag>& c) {
    detail::print_array(output, c.data());
    return output;
  }

} // namespace TiledArray

#endif // TILEDARRAY_COORDINATES_H__INCLUDED
