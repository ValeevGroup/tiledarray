#ifndef NUMERIC_H_
#define NUMERIC_H_

#include <vector>
#include <cmath>
#include <boost/operators.hpp>
#include <boost/array.hpp>

#include <permutation.h>

using namespace boost;

namespace TiledArray {

template <typename T, unsigned int D, typename Tag, typename CS>
class ArrayCoordinate;

template <typename T, unsigned int D, typename Tag, typename CS>
bool operator<(const ArrayCoordinate<T,D,Tag,CS>&, const ArrayCoordinate<T,D,Tag,CS>&);

template <typename T, unsigned int D, typename Tag, typename CS>
bool operator==(const ArrayCoordinate<T,D,Tag,CS>& c1, const ArrayCoordinate<T,D,Tag,CS>& c2);

template <typename T, unsigned int D, typename Tag, typename CS>
std::ostream& operator<<(std::ostream& output, const ArrayCoordinate<T,D,Tag,CS>& c);

template <typename T, unsigned int D, typename Tag, typename CS>
ArrayCoordinate<T,D,Tag,CS> operator^(const Permutation<D>& P, const ArrayCoordinate<T,D,Tag,CS>& C);

namespace detail {
  typedef enum {c_dimension_order, fortran_dimension_order} DimensionOrder;
  struct Fortran_CoordinateSystem {
    static const DimensionOrder dimension_order = fortran_dimension_order;
  };
  struct C_CoordinateSystem {
    static const DimensionOrder dimension_order = c_dimension_order;
  };
}
typedef detail::C_CoordinateSystem DefaultCoordinateSystem;

  /// ArrayCoordinate represents coordinates of a point in a DIM-dimensional orthogonal lattice).
  ///
  /// CoordinateSystem is a policy class that specifies e.g. the order of significance of dimension.
  /// This allows to, for example, to define order of iteration to be compatible with C or Fortran arrays.
  ///
  /// The purpose of Tag is to create multiple instances of the class
  /// with identical mathematical behavior but distinct types to allow
  /// overloading in end-user classes.
  template <typename T, unsigned int D, typename Tag, typename CoordinateSystem = DefaultCoordinateSystem>
  class ArrayCoordinate :
      boost::addable< ArrayCoordinate<T,D,Tag,CoordinateSystem>,          // point + point
      boost::subtractable< ArrayCoordinate<T,D,Tag,CoordinateSystem>,    // point - point
      boost::less_than_comparable1< ArrayCoordinate<T,D,Tag,CoordinateSystem>,  // point < point
      boost::equality_comparable1< ArrayCoordinate<T,D,Tag,CoordinateSystem>  // point == point
//      boost:incrementable< ArrayCoordinate<T,D,Tag,CoordinateSystem>,        // ++point
//      boost ::decrementable< ArrayCoordinate<T,D,Tag,CoordinateSystem>     // --point
//      >
//      >
      >
      >
      >
      >
  {
    public:
    typedef T element;
    typedef T volume;
    typedef CoordinateSystem CS;
    typedef boost::array<element,D> Array;
    typedef typename Array::iterator iterator;
    typedef typename Array::const_iterator const_iterator;
    static const unsigned int DIM = D;
    
    ArrayCoordinate(const T& init_value = 0) { r_.assign(init_value); }
    ArrayCoordinate(const T* init_values) { std::copy(init_values,init_values+D,r_.begin()); }
    ArrayCoordinate(const Array& init_values) : r_(init_values) { }
    ~ArrayCoordinate() {}

    /// Returns an interator to the first coordinate
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

    ArrayCoordinate<T, D, Tag, CoordinateSystem>& operator++() {
      if (CoordinateSystem::dimension_order == detail::c_dimension_order) {
        T& least_significant = *r_.rbegin();
        ++least_significant;
        return *this;
      }
      if (CoordinateSystem::dimension_order == detail::fortran_dimension_order) {
        T& least_significant = *r_.begin();
        ++least_significant;
        return *this;
      }
    }
    ArrayCoordinate<T, D, Tag, CoordinateSystem>& operator--() {
      if (CoordinateSystem::dimension_order == detail::c_dimension_order) {
        T& least_significant = *r_.rbegin();
        --least_significant;
        return *this;
      }
      if (CoordinateSystem::dimension_order == detail::fortran_dimension_order) {
        T& least_significant = *r_.begin();
        --least_significant;
        return *this;
      }
    }
    
    /// Add operator
    ArrayCoordinate<T, D, Tag, CoordinateSystem>& operator+=(const ArrayCoordinate& c) {
      for(unsigned int d = 0; d < DIM; ++d)
        r_[d] += c.r_[d];
      return *this;
    }

    /// Subtract operator
    ArrayCoordinate<T, D, Tag, CoordinateSystem> operator-=(const ArrayCoordinate& c) {
      for(unsigned int d = 0; d < DIM; ++d)
        r_[d] -= c.r_[d];
      return *this;
    }

    ArrayCoordinate<T, D, Tag, CoordinateSystem> operator -() const {
      ArrayCoordinate<T, D, Tag> ret;
      for(unsigned int d = 0; d < DIM; ++d)
        ret.r_[d] = -r_[d];
      return ret;
    }

    const T& operator[](size_t d) const
    {
#ifdef NDEBUG
      return r_[d]; 
#else
      return r_.at(d);
#endif
    }

    T& operator[](size_t d)
    {
#ifdef NDEBUG
      return r_[d]; 
#else
      return r_.at(d);
#endif
    }

    friend bool operator < <>(const ArrayCoordinate<T,D,Tag,CoordinateSystem>&, const ArrayCoordinate<T,D,Tag,CoordinateSystem>&);
    friend bool operator == <>(const ArrayCoordinate<T,D,Tag,CoordinateSystem>&, const ArrayCoordinate<T,D,Tag,CoordinateSystem>&);
    friend std::ostream& operator << <>(std::ostream&, const ArrayCoordinate<T,D,Tag,CoordinateSystem>&);
    friend ArrayCoordinate<T,D,Tag,CS> operator^ <> (const Permutation<D>& P, const ArrayCoordinate<T,D,Tag,CS>& C);
    
  private:
    /// last dimension is least significant
    Array r_;
  };
  
  template <typename T, unsigned int D, typename Tag, typename CS>
  bool operator<(const ArrayCoordinate<T,D,Tag,CS>& c1, const ArrayCoordinate<T,D,Tag,CS>& c2) {
    if (CS::dimension_order == detail::c_dimension_order) {
      return std::lexicographical_compare(c1.r_.begin(),c1.r_.end(),c2.r_.begin(),c2.r_.end());
    }
    if (CS::dimension_order == detail::fortran_dimension_order) {
      return std::lexicographical_compare(c1.r_.rbegin(),c1.r_.rend(),c2.r_.rbegin(),c2.r_.rend());
    }
  }

  template <typename T, unsigned int D, typename Tag, typename CS>
  bool operator==(const ArrayCoordinate<T,D,Tag,CS>& c1, const ArrayCoordinate<T,D,Tag,CS>& c2) {
    return c1.r_ == c2.r_;
  }

  template <typename T, unsigned int D, typename Tag, typename CS>
  std::ostream& operator<<(std::ostream& output, const ArrayCoordinate<T,D,Tag,CS>& c) {
    output << "{";
    for(unsigned int dim = 0; dim < D - 1; ++dim)
      output << c[dim] << ", ";
    output << c[D - 1] << "}";
    return output;
  }

  /// apply permutation P to coordinate C
  template <typename T, unsigned int D, typename Tag, typename CS>
  ArrayCoordinate<T,D,Tag,CS> operator^(const Permutation<D>& P, const ArrayCoordinate<T,D,Tag,CS>& C) {
    const typename ArrayCoordinate<T,D,Tag,CS>::Array& _result = operator^<D,T>(P,C.r_);
    ArrayCoordinate<T,D,Tag,CS> result(_result);
    return result;
  }

  /// compute the volume of the orthotope bounded by the origin and C
  template <typename T, unsigned int D, typename Tag, typename CS>
  typename ArrayCoordinate<T,D,Tag,CS>::volume volume(const ArrayCoordinate<T,D,Tag,CS>& C) {
    typename ArrayCoordinate<T,D,Tag,CS>::volume result = 1;
    for(unsigned int dim = 0; dim < D; ++dim)
      result *= std::abs(C[dim]);
    return result;
  }

}

#endif /*NUMERIC_H_*/
