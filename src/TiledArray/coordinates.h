#ifndef TILEDARRAY_COORDINATES_H__INCLUDED
#define TILEDARRAY_COORDINATES_H__INCLUDED

#include <TiledArray/coordinate_system.h>
#include <TiledArray/array_util.h>
#include <TiledArray/error.h>
#include <TiledArray/utility.h>
#include <boost/operators.hpp>
#include <boost/array.hpp>
#include <boost/type_traits.hpp>
#include <iterator>
#ifndef __GXX_EXPERIMENTAL_CXX0X__
#include <stdarg.h>
#endif // __GXX_EXPERIMENTAL_CXX0X__

namespace boost {
  template <typename T, std::size_t D>
  std::ostream& operator<<(std::ostream&, const boost::array<T,D>&);
} // namespace boost

namespace TiledArray {
  // Forward declaration of TiledArray Permutation.
  template <unsigned int DIM>
  class Permutation;

  /// ArrayCoordinate represents coordinates of a point in a DIM-dimensional orthogonal lattice).
  ///
  /// The purpose of Tag is to create multiple instances of the class
  /// with identical mathematical behavior but distinct types to allow
  /// overloading in end-user classes.
  template <typename I, unsigned int DIM, typename Tag>
  class ArrayCoordinate : public
      boost::addable< ArrayCoordinate<I,DIM,Tag>,                // point + point
      boost::subtractable< ArrayCoordinate<I,DIM,Tag>,           // point - point
      boost::less_than_comparable1< ArrayCoordinate<I,DIM,Tag>,  // point < point
      boost::equality_comparable1< ArrayCoordinate<I,DIM,Tag>    // point == point
      > > > >
  {
  public:
    BOOST_STATIC_ASSERT(boost::is_integral<I>::value);

    typedef ArrayCoordinate<I,DIM,Tag> ArrayCoordinate_;
    typedef I index;
    typedef I volume;
    typedef boost::array<index,DIM> Array;
    typedef typename Array::iterator iterator;
    typedef typename Array::const_iterator const_iterator;
    typedef typename Array::reverse_iterator reverse_iterator;
    typedef typename Array::const_reverse_iterator const_reverse_iterator;

    static const unsigned int dim = DIM;

    // Constructors/Destructor
    ArrayCoordinate() { r_.assign(index(0)); }
    template <typename InIter>
    explicit ArrayCoordinate(InIter first) {
      // should variadic constructor been chosen?
      // need to disambiguate the call if DIM==1
      // assume iterators if InIter is not an integral type
      // else assume wanted variadic constructor
      // this scheme follows what std::vector does
      BOOST_STATIC_ASSERT((DIM == 1u) || (! boost::is_integral<InIter>::value));
      detail::initialize_from_values(first, r_.begin(), DIM, boost::is_integral<InIter>());
    }
    ArrayCoordinate(const Array& init_values) : r_(init_values) { } // no throw
    ArrayCoordinate(const ArrayCoordinate& a) : r_(a.r_) { } // no throw
#ifdef __GXX_EXPERIMENTAL_CXX0X__
    ArrayCoordinate(ArrayCoordinate&& a) : r_(std::move(a.r_)) { } // no throw
#endif // __GXX_EXPERIMENTAL_CXX0X__

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// Constant index constructor.

    /// Constructs an ArrayCoordinate with the specified constants. For example,
    /// ArrayCoordinate<std::size_t, 4, ...> p(0, 1, 2, 3); would construct a
    /// point with the coordinates (0, 1, 2, 3).
    template <typename... Params>
    explicit ArrayCoordinate(Params... params) {
      BOOST_STATIC_ASSERT(detail::Count<Params...>::value == DIM);
      BOOST_STATIC_ASSERT(detail::is_integral_list<Params...>::value);
      detail::fill(r_.begin(), params...);
    }

    /// Constructs and returns a coordinate with the given constant values.
    template <typename... Params>
    static ArrayCoordinate_ make(Params... params) {
      return ArrayCoordinate_(params...);
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

    /// Constructs and returns a coordinate with the given constant values.
    static ArrayCoordinate_ make(const index c0, ...) {
      ArrayCoordinate_ result;
      result.r_.assign(0ul);
      va_list ap;
      va_start(ap, c0);

      result.r_[0] = c0;
      for(unsigned int i = 1; i < DIM; ++i)
        result.r_[i] = va_arg(ap, index);

      va_end(ap);

      return result;
    }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    ~ArrayCoordinate() {}

    /// Returns an iterator to the first coordinate
    iterator begin() { return r_.begin(); }

    /// Returns a constant iterator to the first coordinate.
    const_iterator begin() const { return r_.begin(); }

    /// Returns an iterator to one past the last coordinate.
    iterator end() { return r_.end(); }

    /// Returns a constant iterator to one past the last coordinate.
    const_iterator end() const { return r_.end(); }

    /// Returns a reverse iterator to the last coordinate
    reverse_iterator rbegin() { return r_.rbegin(); }

    /// Returns a constant reverse iterator to the last coordinate.
    const_reverse_iterator rbegin() const { return r_.rbegin(); }

    /// Returns a reverse iterator to one before the first coordinate.
    reverse_iterator rend() { return r_.rend(); }

    /// Returns a constant reverse iterator to one before the first coordinate.
    const_reverse_iterator rend() const { return r_.rend(); }

    /// Assignment operator
    ArrayCoordinate_&
    operator =(const ArrayCoordinate_& c) {
      r_ = c.r_;

      return (*this);
    }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    /// Move assignment operator
    ArrayCoordinate_&
    operator =(ArrayCoordinate_&& c) {
      r_ = std::move(c.r_);

      return (*this);
    }
#endif // __GXX_EXPERIMENTAL_CXX0X__

    /// Add operator
    ArrayCoordinate_& operator+=(const ArrayCoordinate_& c) {
      for(unsigned int d = 0; d < DIM; ++d)
        r_[d] += c.r_[d];
      return *this;
    }

    /// Subtract operator
    ArrayCoordinate_ operator-=(const ArrayCoordinate_& c) {
      for(unsigned int d = 0; d < DIM; ++d)
        r_[d] -= c.r_[d];
      return *this;
    }

    const index& operator[](size_t d) const
    {
#ifdef NDEBUG
      return r_[d];
#else
      return r_.at(d);
#endif
    }

    index& operator[](size_t d)
    {
#ifdef NDEBUG
      return r_[d];
#else
      return r_.at(d);
#endif
    }

    const index& at(size_t d) const {  return r_.at(d);  }
    index& at(size_t d) { return r_.at(d); }

    const Array& data() const { return r_; }
    Array& data() { return r_; }

    const ArrayCoordinate_ operator ^= (const Permutation<DIM>& p) {
      r_ = p ^ r_;
      return *this;
    }

    template <typename Archive>
    void serialize(const Archive& ar) {
      ar & r_;
    }

  private:
    /// last dimension is least significant
    Array r_;
  };

  /// Swap the data of c1 with c2.
  template <typename I, unsigned int DIM, typename Tag>
  void swap(ArrayCoordinate<I,DIM,Tag>& c1, ArrayCoordinate<I,DIM,Tag>& c2) { // no throw
    boost::swap(c1.data(), c2.data());
  }

  /// Add constant to coordinate.
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  ArrayCoordinate<I,DIM,Tag>& operator +=(ArrayCoordinate<I,DIM,Tag>& c, const I& s) {
    for(typename ArrayCoordinate<I,DIM,Tag>::iterator it = c.begin(); it != c.end(); ++it)
      *it += s;
    return c;
  }

  /// Returns a coordinates with a constant added to each element.
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  ArrayCoordinate<I,DIM,Tag> operator +(const ArrayCoordinate<I,DIM,Tag>& c, const I& s) {
    ArrayCoordinate<I,DIM,Tag> result(c);
    for(typename ArrayCoordinate<I,DIM,Tag>::iterator it = result.begin(); it != result.end(); ++it)
      *it += s;
    return result;
  }

  /// Subtract a constant from coordinate.
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  ArrayCoordinate<I,DIM,Tag>& operator -=(ArrayCoordinate<I,DIM,Tag>& c, const I& s) {
    for(typename ArrayCoordinate<I,DIM,Tag>::iterator it = c.begin(); it != c.end(); ++it)
      *it -= s;
    return c;
  }

  /// Returns a coordinates with a constant subtracted from each element.
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  ArrayCoordinate<I,DIM,Tag> operator -(const ArrayCoordinate<I,DIM,Tag>& c, const I& s) {
    ArrayCoordinate<I,DIM,Tag> result(c);
    for(typename ArrayCoordinate<I,DIM,Tag>::iterator it = result.begin(); it != result.end(); ++it)
      *it -= s;
    return result;
  }

  namespace detail {

    template <typename I, std::size_t DIM>
    bool less(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2) {
      return std::equal(a1.begin(), a1.end(), a2.begin(), std::less<I>());
    }

    template <typename I, std::size_t DIM>
    bool less_eq(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2) {
      return std::equal(a1.begin(), a1.end(), a2.begin(), std::less_equal<I>());
    }

    template <typename I, std::size_t DIM>
    bool greater(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2) {
      return std::equal(a1.begin(), a1.end(), a2.begin(), std::greater<I>());
    }

    template <typename I, std::size_t DIM>
    bool greater_eq(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2) {
      return std::equal(a1.begin(), a1.end(), a2.begin(), std::greater_equal<I>());
    }

  } // namespace detail

  /// Compare ArrayCoordinates Lexicographically.
  template <typename I, unsigned int DIM, typename Tag>
  bool operator<(const ArrayCoordinate<I,DIM,Tag>& c1, const ArrayCoordinate<I,DIM,Tag>& c2) {
    return std::equal(c1.begin(), c1.end(), c2.begin());
  }

  template <typename I, unsigned int DIM, typename Tag>
  bool operator==(const ArrayCoordinate<I,DIM,Tag>& c1, const ArrayCoordinate<I,DIM,Tag>& c2) {
    return std::equal(c1.begin(), c1.end(), c2.begin());
  }

  template <typename I>
  bool operator==(const boost::array<I,1>& a, const I& i) {
    return a[0] == i;
  }

  template <typename I>
  bool operator==(const I& i, const boost::array<I,1>& a) {
    return a[0] == i;
  }

  template <typename I, typename Tag>
  bool operator==(const ArrayCoordinate<I,1,Tag>& c, const I& i) {
    return c[0] == i;
  }

  template <typename I, typename Tag>
  bool operator==(const I& i, const ArrayCoordinate<I,1,Tag>& c) {
    return operator==(c[0], i);
  }

  template <typename I, typename Tag>
  bool operator!=(const ArrayCoordinate<I,1,Tag>& c, const I& i) {
    return ! operator==(c[0], i);
  }

  template <typename I, typename Tag>
  bool operator!=(const I& i, const ArrayCoordinate<I,1,Tag>& c) {
    return ! operator==(c[0], i);
  }

  /// Permute an ArrayCoordinate
  template <typename I, unsigned int DIM, typename Tag>
  ArrayCoordinate<I,DIM,Tag> operator ^(const Permutation<DIM>& perm, const ArrayCoordinate<I,DIM,Tag>& c) {
    ArrayCoordinate<I,DIM,Tag> result(c);
    return result ^= perm;
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
