#ifndef TILEDARRAY_COORDINATES_H__INCLUDED
#define TILEDARRAY_COORDINATES_H__INCLUDED

#include <coordinate_system.h>
#include <array_util.h>
#include <boost/operators.hpp>
#include <boost/array.hpp>
#include <stdarg.h>
#include <iterator>

namespace boost {
  template <typename T, std::size_t D>
  std::ostream& operator<<(std::ostream&, const boost::array<T,D>&);
} // namespace boost

namespace TiledArray {

  // Forward declaration of TiledArray Permutation.
  template <unsigned int DIM>
  class Permutation;
  template <unsigned int DIM, typename T>
  boost::array<T,DIM> operator^(const Permutation<DIM>&, const boost::array<T, static_cast<std::size_t>(DIM) >&);
  template <unsigned int DIM, typename T>
  boost::array<T,DIM> operator ^=(boost::array<T, static_cast<std::size_t>(DIM) >&, const Permutation<DIM>&);

  template <typename T, unsigned int DIM, typename Tag, typename CS>
  class ArrayCoordinate;
  template <typename Coord>
  Coord make_coord(const typename Coord::index, ...);
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  void swap(ArrayCoordinate<I,DIM,Tag,CS>&, ArrayCoordinate<I,DIM,Tag,CS>&);
  /// Add operator
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  ArrayCoordinate<I,DIM,Tag,CS>& operator +=(ArrayCoordinate<I,DIM,Tag,CS>& c, const I& s);
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  ArrayCoordinate<I,DIM,Tag,CS> operator +(const ArrayCoordinate<I,DIM,Tag,CS>& c, const I& s);
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  ArrayCoordinate<I,DIM,Tag,CS>& operator -=(ArrayCoordinate<I,DIM,Tag,CS>& c, const I& s);
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  ArrayCoordinate<I,DIM,Tag,CS> operator -(const ArrayCoordinate<I,DIM,Tag,CS>& c, const I& s);
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  bool operator<(const ArrayCoordinate<T,DIM,Tag,CS>&, const ArrayCoordinate<T,DIM,Tag,CS>&);
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  bool operator==(const ArrayCoordinate<T,DIM,Tag,CS>& c1, const ArrayCoordinate<T,DIM,Tag,CS>& c2);
  template <typename I, typename Tag, typename CS>
  bool operator==(const ArrayCoordinate<I,1,Tag,CS>&, const I&);
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  std::ostream& operator<<(std::ostream& output, const ArrayCoordinate<T,DIM,Tag,CS>& c);
  template <typename T, unsigned int DIM, typename Tag, typename CS>
  ArrayCoordinate<T,DIM,Tag,CS> operator^(const Permutation<DIM>& P, const ArrayCoordinate<T,DIM,Tag,CS>& C);
  template <typename I>
  bool operator==(const boost::array<I,1>&, const I&);
  template <typename I>
  bool operator==(const I&, const boost::array<I,1>&);
  template <typename I, typename Tag, typename CS>
  bool operator==(const I&, const ArrayCoordinate<I,1,Tag,CS>&);
  template <typename I, typename Tag, typename CS>
  bool operator!=(const ArrayCoordinate<I,1,Tag,CS>&, const I&);
  template <typename I, typename Tag, typename CS>
  bool operator!=(const I&, const ArrayCoordinate<I,1,Tag,CS>&);

  namespace detail {
    template <typename I, std::size_t DIM>
    bool less(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2);
    template <typename I, std::size_t DIM>
    bool less_eq(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2);
    template <typename I, std::size_t DIM>
    bool greater(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2);
    template <typename I, std::size_t DIM>
    bool greater_eq(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2);
    template <typename CS, typename I, std::size_t DIM>
    bool lex_less(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2);
  } // namespace detail

  /// ArrayCoordinate Tag strut: It is used to ensure type safety between different tiling domains.
  template<unsigned int Level>
  struct LevelTag { };

  /// ArrayCoordinate represents coordinates of a point in a DIM-dimensional orthogonal lattice).
  ///
  /// The purpose of Tag is to create multiple instances of the class
  /// with identical mathematical behavior but distinct types to allow
  /// overloading in end-user classes.
  template <typename I, unsigned int DIM, typename Tag, typename CS = CoordinateSystem<DIM> >
  class ArrayCoordinate : public
      boost::addable< ArrayCoordinate<I,DIM,Tag,CS>,                // point + point
      boost::subtractable< ArrayCoordinate<I,DIM,Tag,CS>,           // point - point
      boost::less_than_comparable1< ArrayCoordinate<I,DIM,Tag,CS>,  // point < point
      boost::equality_comparable1< ArrayCoordinate<I,DIM,Tag,CS>,   // point == point
      boost::incrementable< ArrayCoordinate<I,DIM,Tag,CS>,          // point++
      boost::decrementable< ArrayCoordinate<I,DIM,Tag,CS>           // point--
      > > > > > >
  {
  public:
    typedef ArrayCoordinate<I,DIM,Tag,CS> ArrayCoordinate_;
    typedef I index;
    typedef I volume;
    typedef CS coordinate_system;
    typedef boost::array<index,DIM> Array;
    typedef typename Array::iterator iterator;
    typedef typename Array::const_iterator const_iterator;
    typedef typename Array::reverse_iterator reverse_iterator;
    typedef typename Array::const_reverse_iterator const_reverse_iterator;
    static unsigned int dim() { return DIM; }

    // Constructors/Destructor
    explicit ArrayCoordinate(const index& init_value = 0ul) { r_.assign(init_value); }
    template <typename InIter>
    explicit ArrayCoordinate(InIter start, InIter finish) { std::copy(start,finish,r_.begin()); }
    ArrayCoordinate(const Array& init_values) : r_(init_values) { } // no throw
    ArrayCoordinate(const ArrayCoordinate& a) : r_(a.r_) { } // no throw
#ifdef __GXX_EXPERIMENTAL_CXX0X__
    ArrayCoordinate(ArrayCoordinate&& a) : r_(std::move(a.r_)) { } // no throw
#endif // __GXX_EXPERIMENTAL_CXX0X__
    /// Constant index constructor.

    /// Constructs an ArrayCoordinate with the specified constants. For example,
    /// ArrayCoordinate<std::size_t, 4, p(0, 1, 2, 3); would construct a point with the
    /// coordinates (0, 1, 2, 3).
    /// Note: The compiler gets confused when constructing a 2D array coordinate
    /// with this function. To work around this problem specify constant type.
    /// For example
    ArrayCoordinate(const index c0, const index c1, ...) {
      r_.assign(0ul);
      va_list ap;
      va_start(ap, c1);

      r_[0] = c0;
      r_[1] = c1;
      unsigned int ci = 0; // ci is used as an intermediate
      for(unsigned int i = 2; i < dim(); ++i) {
        ci = va_arg(ap, index);
        r_[i] = ci;
      }

      va_end(ap);
    }
    virtual ~ArrayCoordinate() {}

    static ArrayCoordinate_ make(const index c0, ...) {
      ArrayCoordinate_ result;
      result.r_.assign(0ul);
      va_list ap;
      va_start(ap, c0);

      result.r_[0] = c0;
      for(unsigned int i = 1; i < dim(); ++i)
        result.r_[i] = va_arg(ap, index);

      va_end(ap);

      return result;
    }

    /// Returns an iterator to the first coordinate
    iterator begin() {
      return r_.begin();
    }

    /// Returns a constant iterator to the first coordinate.
    const_iterator begin() const {
      return r_.begin();
    }

    /// Returns an iterator to one element past the last coordinate.
    iterator end() {
      return r_.end();
    }

    /// Returns a constant iterator to one element past the last coordinate.
    const_iterator end() const {
      return r_.end();
    }

    /// Returns a reverse iterator to the first coordinate
    reverse_iterator rbegin() {
      return r_.rbegin();
    }

    /// Returns a constant reverse iterator to the first coordinate.
    const_reverse_iterator rbegin() const {
      return r_.rbegin();
    }

    /// Returns a reverse iterator to one element past the last coordinate.
    reverse_iterator rend() {
      return r_.rend();
    }

    /// Returns a constant reverse iterator to one element past the last coordinate.
    const_reverse_iterator rend() const {
      return r_.rend();
    }

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

    ArrayCoordinate_& operator++() {
      const unsigned int lsdim = * coordinate_system::begin();
      ++(r_[lsdim]);
      return *this;
    }

    ArrayCoordinate_& operator--() {
      const unsigned int lsdim = * coordinate_system::begin();
      --(r_[lsdim]);
      return *this;
    }

    /// Add operator
    ArrayCoordinate_& operator+=(const ArrayCoordinate_& c) {
      for(unsigned int d = 0; d < dim(); ++d)
        r_[d] += c.r_[d];
      return *this;
    }

    /// Subtract operator
    ArrayCoordinate_ operator-=(const ArrayCoordinate_& c) {
      for(unsigned int d = 0; d < dim(); ++d)
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

  template <typename Coord>
  Coord make_coord(const typename Coord::index c0, ...) {
    Coord result;
    va_list ap;
    va_start(ap, c0);

    result[0] = c0;
    for(unsigned int i = 1; i < Coord::dim(); ++i)
      result[i] = va_arg(ap, typename Coord::index);

    va_end(ap);

    return result;
  }

  /// Swap the data of c1 with c2.
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  void swap(ArrayCoordinate<I,DIM,Tag,CS>& c1, ArrayCoordinate<I,DIM,Tag,CS>& c2) { // no throw
    boost::swap(c1.data(), c2.data());
  }

  /// Add operator
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  ArrayCoordinate<I,DIM,Tag,CS>& operator +=(ArrayCoordinate<I,DIM,Tag,CS>& c, const I& s) {
    for(typename ArrayCoordinate<I,DIM,Tag,CS>::iterator it = c.begin(); it != c.end(); ++it)
      *it += s;
    return c;
  }

  template <typename I, unsigned int DIM, typename Tag, typename CS>
  ArrayCoordinate<I,DIM,Tag,CS> operator +(const ArrayCoordinate<I,DIM,Tag,CS>& c, const I& s) {
    ArrayCoordinate<I,DIM,Tag,CS> result(c);
    for(typename ArrayCoordinate<I,DIM,Tag,CS>::iterator it = result.begin(); it != result.end(); ++it)
      *it += s;
    return result;
  }

  /// Subtract operator
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  ArrayCoordinate<I,DIM,Tag,CS>& operator -=(ArrayCoordinate<I,DIM,Tag,CS>& c, const I& s) {
    for(typename ArrayCoordinate<I,DIM,Tag,CS>::iterator it = c.begin(); it != c.end(); ++it)
      *it -= s;
    return c;
  }

  template <typename I, unsigned int DIM, typename Tag, typename CS>
  ArrayCoordinate<I,DIM,Tag,CS> operator -(const ArrayCoordinate<I,DIM,Tag,CS>& c, const I& s) {
    ArrayCoordinate<I,DIM,Tag,CS> result(c);
    for(typename ArrayCoordinate<I,DIM,Tag,CS>::iterator it = result.begin(); it != result.end(); ++it)
      *it -= s;
    return result;
  }

  namespace detail {
    /// Compare each element in the array.
    template <typename I, std::size_t DIM, typename C >
    struct Compare {
      bool operator ()(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2) {
        C c;
        for(unsigned int i = 0; i < DIM; ++i)
          if(! c(a1[i], a2[i]))
            return false;
        return true; // all members of c1 are less than c2
      }
    }; // struct Compare

    template <typename I, std::size_t DIM>
    bool less(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2) {
      Compare<I, DIM, std::less<I> > l;
      return l(a1, a2);
    }

    template <typename I, std::size_t DIM>
    bool less_eq(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2) {
      Compare<I, DIM, std::less_equal<I> > le;
      return le(a1, a2);
    }

    template <typename I, std::size_t DIM>
    bool greater(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2) {
      Compare<I, DIM, std::greater<I> > g;
      return g(a1, a2);
    }

    template <typename I, std::size_t DIM>
    bool greater_eq(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2) {
      Compare<I, DIM, std::greater_equal<I> > g;
      return g(a1, a2);
    }

    /// Compare ArrayCoordinates Lexicographically.
    template <typename I, std::size_t DIM, typename CS, typename C >
    struct LexCompare {
      bool operator ()(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2) {
        // Get order iterators.
        C c;
        for(typename CS::const_reverse_iterator it = CS::rbegin(); it != CS::rend(); ++it) {
          if(c(a2[*it], a1[*it]))
            return false;
          else if(c(a1[*it], a2[*it]))
            return true;
        }
        return false; // all elements were equal
      }
    }; // struct LexLess

    template <typename CS, typename I, std::size_t DIM>
    bool lex_less(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2) {
      LexCompare<I, DIM, CS, std::less<I> > ll;
      return ll(a1, a2);
    }

  } // namespace detail

  /// Compare ArrayCoordinates Lexicographically.
  template <typename I, unsigned int DIM, typename Tag, typename CS >
  bool operator<(const ArrayCoordinate<I,DIM,Tag,CS>& c1, const ArrayCoordinate<I,DIM,Tag,CS>& c2) {
    return detail::lex_less<CS, I, DIM>(c1.data(), c2.data());
  }

  template <typename I, unsigned int DIM, typename Tag, typename CS>
  bool operator==(const ArrayCoordinate<I,DIM,Tag,CS>& c1, const ArrayCoordinate<I,DIM,Tag,CS>& c2) {
    return c1.data() == c2.data();
  }

  template <typename I>
  bool operator==(const boost::array<I,1>& a, const I& i) {
    return a[0] == i;
  }

  template <typename I>
  bool operator==(const I& i, const boost::array<I,1>& a) {
    return a[0] == i;
  }

  template <typename I, typename Tag, typename CS>
  bool operator==(const ArrayCoordinate<I,1,Tag,CS>& c, const I& i) {
    return c[0] == i;
  }

  template <typename I, typename Tag, typename CS>
  bool operator==(const I& i, const ArrayCoordinate<I,1,Tag,CS>& c) {
    return operator==(c[0], i);
  }

  template <typename I, typename Tag, typename CS>
  bool operator!=(const ArrayCoordinate<I,1,Tag,CS>& c, const I& i) {
    return ! operator==(c[0], i);
  }

  template <typename I, typename Tag, typename CS>
  bool operator!=(const I& i, const ArrayCoordinate<I,1,Tag,CS>& c) {
    return ! operator==(c[0], i);
  }

  /// Permute an ArrayCoordinate
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  ArrayCoordinate<I,DIM,Tag,CS> operator ^(const Permutation<DIM>& perm, const ArrayCoordinate<I,DIM,Tag,CS>& c) {
    ArrayCoordinate<I,DIM,Tag,CS> result(c);
    return result ^= perm;
  }

  /// Append an ArrayCoordinate<...> to an output stream.
  template <typename I, unsigned int DIM, typename Tag, typename CS>
  std::ostream& operator<<(std::ostream& output, const ArrayCoordinate<I,DIM,Tag,CS>& c) {
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
