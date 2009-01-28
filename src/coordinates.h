#ifndef NUMERIC_H_
#define NUMERIC_H_

#include <assert.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <boost/operators.hpp>
#include <boost/array.hpp>

#include <permutation.h>

//using namespace boost; // Do not use "using namespace" in lib headers.

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

namespace {
  // sort dimensions by their order
  struct DimOrderPair {
    unsigned int dim, order;
    DimOrderPair(unsigned int d = 0,unsigned int o = 0) : dim(d), order(o) {}
    /// compare by order
    bool operator<(const DimOrderPair& other) const {
      return order < other.order;
    }
  };
};

namespace detail {

  typedef enum {decreasing_dimension_order, increasing_dimension_order, general_dimension_order} DimensionOrderType;
  template <unsigned int D>
  class DimensionOrder {
    typedef boost::array<unsigned int,D> IntArray;
  public:
    typedef typename IntArray::const_iterator const_iterator;
    typedef typename IntArray::const_reverse_iterator const_reverse_iterator;
    DimensionOrder(const DimensionOrderType& order) {
      // map dimensions to their importance order
      switch (order) {
        case increasing_dimension_order:
        for(unsigned int d = 0; d < D; ++d) {
          ord_[d] = d;
        }
        break;

        case decreasing_dimension_order:
        for(unsigned int d = 0; d < D; ++d) {
          ord_[d] = D-d-1;
        }
        break;

        case general_dimension_order:
        throw std::runtime_error("general dimension ordering is not supported");
        break;
      }

      //
      // compute dim_ by inverting ord_ map
      //
      std::vector<DimOrderPair> sorted_by_order(D);
      for(unsigned int d=0; d<D; ++d)
        sorted_by_order[d] = DimOrderPair(d,ord_[d]);
      std::sort(sorted_by_order.begin(),sorted_by_order.end());
      // construct the order->dimension map
      for(unsigned int d=0; d<D; ++d)
        dim_[d] = sorted_by_order[d].dim;

    }
    unsigned int dim2order(unsigned int d) const { return ord_[d]; }
    unsigned int order2dim(unsigned int o) const { return dim_[o]; }
    /// use this to start iteration over dimensions in increasing order of their significance
    const_iterator begin_order() const { return dim_.begin(); }
    /// use this to end iteration over dimensions in increasing order of their significance
    const_iterator end_order() const { return dim_.end(); }
    /// use this to start iteration over dimensions in decreasing order of their significance
    const_reverse_iterator rbegin_order() const { return dim_.rbegin(); }
    /// use this to end iteration over dimensions in decreasing order of their significance
    const_reverse_iterator rend_order() const { return dim_.rend(); }
  private:
    /// maps dimension to its order
    boost::array<unsigned int,D> ord_;
    /// maps order to dimension -- inverse of ord_
    boost::array<unsigned int,D> dim_;
  };
}; // namespace detail


  /// Specifies the details of a D-dimensional coordinate system. The default is for the last dimension to be least significant.
  template <unsigned int D, detail::DimensionOrderType Order = detail::decreasing_dimension_order>
  class CoordinateSystem {
    public:
      static const detail::DimensionOrder<D>& ordering() { return ordering_; }
      static const detail::DimensionOrderType dimension_order = Order;
    private:
      static detail::DimensionOrder<D> ordering_;
  };
  template <unsigned int D, detail::DimensionOrderType Order> detail::DimensionOrder<D> CoordinateSystem<D,Order>::ordering_(Order);

  /// ArrayCoordinate represents coordinates of a point in a DIM-dimensional orthogonal lattice).
  ///
  /// CoordinateSystem is a policy class that specifies e.g. the order of significance of dimension.
  /// This allows to, for example, to define order of iteration to be compatible with C or Fortran arrays.
  ///
  /// The purpose of Tag is to create multiple instances of the class
  /// with identical mathematical behavior but distinct types to allow
  /// overloading in end-user classes.
  template <typename T, unsigned int D, typename Tag, typename CS = CoordinateSystem<D> >
  class ArrayCoordinate :
      boost::addable< ArrayCoordinate<T,D,Tag,CS>,                // point + point
      boost::subtractable< ArrayCoordinate<T,D,Tag,CS>,           // point - point
      boost::less_than_comparable1< ArrayCoordinate<T,D,Tag,CS>,  // point < point
      boost::equality_comparable1< ArrayCoordinate<T,D,Tag,CS>,   // point == point
      boost::incrementable< ArrayCoordinate<T,D,Tag,CS>,          // point++
      boost::decrementable< ArrayCoordinate<T,D,Tag,CS>           // point--
      > > > > > >
  {
  public:
    typedef T index;
    typedef T volume;
    typedef CS CoordinateSystem;
    typedef boost::array<index,D> Array;
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

    /// Assignment operator
    ArrayCoordinate<T, D, Tag, CoordinateSystem>&
    operator =(const ArrayCoordinate<T, D, Tag, CoordinateSystem>& c) {
      std::copy(c.r_.begin(), c.r_.end(), r_.begin());

      return (*this);
    }

    ArrayCoordinate<T, D, Tag, CoordinateSystem>& operator++() {
      const unsigned int lsdim = *CoordinateSystem::ordering().begin_order();
      T& least_significant = r_[lsdim];
      ++least_significant;
      return *this;
    }

    ArrayCoordinate<T, D, Tag, CoordinateSystem>& operator--() {
      const unsigned int lsdim = *CoordinateSystem::ordering().begin_order();
      T& least_significant = r_[lsdim];
      --least_significant;
      return *this;
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

    const Array& data() const {
      return r_;
    }

    friend bool operator < <>(const ArrayCoordinate<T,D,Tag,CoordinateSystem>&, const ArrayCoordinate<T,D,Tag,CoordinateSystem>&);
    friend bool operator == <>(const ArrayCoordinate<T,D,Tag,CoordinateSystem>&, const ArrayCoordinate<T,D,Tag,CoordinateSystem>&);
    friend std::ostream& operator << <>(std::ostream&, const ArrayCoordinate<T,D,Tag,CoordinateSystem>&);
    friend ArrayCoordinate<T,D,Tag,CS> operator^ <> (const Permutation<D>& P, const ArrayCoordinate<T,D,Tag,CS>& C);

  private:
    /// last dimension is least significant
    Array r_;
  };

  // TODO how to recast this without using dimension_order
  template <typename T, unsigned int D, typename Tag, typename CS>
  bool operator<(const ArrayCoordinate<T,D,Tag,CS>& c1, const ArrayCoordinate<T,D,Tag,CS>& c2) {
    if (CS::dimension_order == detail::decreasing_dimension_order) {
      return std::lexicographical_compare(c1.r_.begin(),c1.r_.end(),c2.r_.begin(),c2.r_.end());
    }
    if (CS::dimension_order == detail::increasing_dimension_order) {
      return std::lexicographical_compare(c1.r_.rbegin(),c1.r_.rend(),c2.r_.rbegin(),c2.r_.rend());
    }
    abort();
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
      result *= std::abs(static_cast<long int>(C[dim]));
    return result;
  }

  /// compute dot product between 2 arrays
  template <typename T, unsigned int D>
  T dot_product(const boost::array<T,D>& A, const boost::array<T,D>& B) {
    T result = 0;
    for(unsigned int dim = 0; dim < D; ++dim)
      result += A[dim] * B[dim];
    return result;
  }

}

#endif /*NUMERIC_H_*/
