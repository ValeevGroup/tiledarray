#ifndef NUMERIC_H_
#define NUMERIC_H_

#include <coordinate_system.h>
#include <iosfwd>
#include <cmath>
#include <functional>
#include <boost/operators.hpp>
#include <boost/array.hpp>

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

  template <typename T, unsigned int DIM, typename Tag, typename CS>
  bool operator<(const ArrayCoordinate<T,DIM,Tag,CS>&, const ArrayCoordinate<T,DIM,Tag,CS>&);

  template <typename T, unsigned int DIM, typename Tag, typename CS>
  bool operator==(const ArrayCoordinate<T,DIM,Tag,CS>& c1, const ArrayCoordinate<T,DIM,Tag,CS>& c2);

  template <typename T, unsigned int DIM, typename Tag, typename CS>
  std::ostream& operator<<(std::ostream& output, const ArrayCoordinate<T,DIM,Tag,CS>& c);

  template <typename T, unsigned int DIM, typename Tag, typename CS>
  ArrayCoordinate<T,DIM,Tag,CS> operator^(const Permutation<DIM>& P, const ArrayCoordinate<T,DIM,Tag,CS>& C);

  /// ArrayCoordinate Tag strut: It is used to ensure type safety between different tiling domains.
  template<unsigned int Level>
  struct LevelTag { };

  /// ArrayCoordinate represents coordinates of a point in a DIM-dimensional orthogonal lattice).
  ///
  /// The purpose of Tag is to create multiple instances of the class
  /// with identical mathematical behavior but distinct types to allow
  /// overloading in end-user classes.
  template <typename I, unsigned int DIM, typename Tag, typename CS = CoordinateSystem<DIM> >
  class ArrayCoordinate :
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
    static const unsigned int dim() { return DIM; }

    // Constructors/Destructor
    ArrayCoordinate(const index& init_value = 0) { r_.assign(init_value); }
    template <typename InIter>
    ArrayCoordinate(InIter start, InIter finish) { std::copy(start,finish,r_.begin()); }
    ArrayCoordinate(const Array& init_values) : r_(init_values) { } // no throw
    ArrayCoordinate(const ArrayCoordinate& a) : r_(a.r_) { } // no throw
    /// Variable argument list constructor.
    ArrayCoordinate(const index c0, const index c1, ...) {
      va_list ap;
      va_start(ap, c1);

      r_[0] = c0;
      r_[1] = c1;
      for(unsigned int i = 2; i < dim(); ++i)
        r_[i] = va_arg(ap, index);

      va_end(ap);
    }
    ~ArrayCoordinate() {}

    static ArrayCoordinate_ make(const index c0, ...) {
      ArrayCoordinate_ result;
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

    /// Assignment operator
    ArrayCoordinate_&
    operator =(const ArrayCoordinate_& c) {
      std::copy(c.r_.begin(), c.r_.end(), r_.begin());

      return (*this);
    }

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

    friend bool operator < <>(const ArrayCoordinate_&, const ArrayCoordinate_&);
    friend bool operator == <>(const ArrayCoordinate_&, const ArrayCoordinate_&);
    friend std::ostream& operator << <>(std::ostream&, const ArrayCoordinate_&);

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

  namespace detail {
    /// Compare each element in the array to make sure it is
    template <typename I, unsigned int DIM, typename CS, typename L = std::less<I> >
    struct Less {
      bool operator ()(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2) {
        L l;
        for(unsigned int i = 0; i < DIM; ++i)
          if(! l(a1[i], a2[i]))
            return false;
        return true; // all members of c1 are less than c2
      }
    }; // struct Less

    template <typename I, unsigned int DIM, typename CS, typename L = std::less<I> >
    struct LessEq {
      bool operator ()(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2) {
        L l;
        for(unsigned int i = 0; i < DIM; ++i)
          if(l(a2[i], a1[i]))
            return false;
        return true; // all members of c1 are less than c2
      }
    }; // struct LessEq

    template <typename I, unsigned int DIM, typename CS>
    bool less(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2) {
      Less<I,DIM,CS> l;
      return l(a1, a2);
    }

    template <typename I, unsigned int DIM, typename CS, typename L>
    bool less(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2) {
      Less<I,DIM,CS,L> l;
      return l(a1, a2);
    }

    /// Compare ArrayCoordinates Lexicographically.
    template <typename I, unsigned int DIM, typename CS, typename L = std::less<I> >
    struct LexLess {
      bool operator ()(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2) {
        // Get order iterators.
        L l;
        for(typename CS::const_reverse_iterator it = CS::rbegin(); it != CS::rend(); ++it) {
          if(l(a2[*it], a1[*it]))
            return false;
          else if(l(a1[*it], a2[*it]))
            return true;
        }
        return false; // all elements were equal
      }
    }; // struct LexLess

    template <typename I, unsigned int DIM, typename CS>
    bool lex_less(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2) {
      LexLess<I,DIM,CS> ll;
      return ll(a1, a2);
    }

    template <typename I, unsigned int DIM, typename CS, typename L>
    bool lex_less(const boost::array<I,DIM>& a1, const boost::array<I,DIM>& a2) {
      LexLess<I,DIM,CS, L> ll;
      return ll(a1, a2);
    }
  } // namespace detail

  /// Compare ArrayCoordinates Lexicographically.
  template <typename I, unsigned int DIM, typename Tag, typename CS >
  bool operator<(const ArrayCoordinate<I,DIM,Tag,CS>& c1, const ArrayCoordinate<I,DIM,Tag,CS>& c2) {
    detail::LexLess<I,DIM,CS> ll;
    return ll(c1.data(), c2.data());
  }

  template <typename I, unsigned int DIM, typename Tag, typename CS>
  bool operator==(const ArrayCoordinate<I,DIM,Tag,CS>& c1, const ArrayCoordinate<I,DIM,Tag,CS>& c2) {
    return c1.r_ == c2.r_;
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
    for(unsigned int dim = 0; dim < DIM - 1; ++dim)
      output << c[dim] << ", ";
    output << c[DIM - 1] << ")";
    return output;
  }

} // namespace TiledArray

// We need this operator in the boost namespace so it works properly in other
// name spaces, specifically in boost::test namespace.
namespace boost {
  /// Append a boost::array<T,D> to an output stream.
  template <typename T, std::size_t D>
  std::ostream& operator<<(std::ostream& output, const boost::array<T,D>& a) {
    output << "{{";
    for(std::size_t d = 0; d < D - 1; ++d)
      output << a[d] << ", ";
    output << a[D - 1] << "}}";
    return output;
  }
} // namespace boost

#endif /*NUMERIC_H_*/
