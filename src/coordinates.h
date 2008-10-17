#ifndef NUMERIC_H_
#define NUMERIC_H_

#include <vector>
#include <boost/operators.hpp>

using namespace boost;

namespace TiledArray {

  /// LatticeCoordinate represents coordinates of a point in a DIM-dimensional lattice.
  ///
  /// The purpose of Tag is to create multiple instances of the class
  /// with identical mathematical behavior but distinct types to allow
  /// overloading in classes using LatticePoint.
  template <typename T, unsigned int D, typename Tag>
  struct LatticeCoordinate :
      boost::addable< LatticeCoordinate<T,D,Tag>,          // point + point
      boost::subtractable< LatticeCoordinate<T,D,Tag>,    // point - point
      boost::less_than_comparable1< LatticeCoordinate<T,D,Tag>,  // point < point
      boost::equality_comparable1< LatticeCoordinate<T,D,Tag>  // point == point
//      boost:incrementable< LatticeCoordinate<T,D,Tag>,        // ++point
//      boost ::decrementable< LatticeCoordinate<T,D,Tag>     // --point
//      >
//      >
      >
      >
      >
      >
  {
    public:
    typedef T Element;
    static const unsigned int DIM = D;
    
    LatticeCoordinate(const T& init_value = 0) : r(D,init_value) {}
    ~LatticeCoordinate() {}

    LatticeCoordinate& operator++() { ++(*r.rbegin()); }
    LatticeCoordinate& operator--() { --(*r.rbegin()); }
    LatticeCoordinate& operator+=(const LatticeCoordinate& c) const {
      for(unsigned int d=0; d<DIM; ++d)
        r[d] += c.r[d];
      return *this;
    }
    LatticeCoordinate operator-=(const LatticeCoordinate& c) const {
      for(unsigned int d=0; d<DIM; ++d)
        r[d] -= c.r[d];
      return *this;
    }
    
    const T& operator[](size_t d) const { return r[d]; }

    /// last dimension is least significant
    /// default is to use standard vector
    std::vector<T> r;
  };
  
  template <typename T, unsigned int D, typename Tag>
  bool operator<(const LatticeCoordinate<T,D,Tag>& c1, const LatticeCoordinate<T,D,Tag>& c2) {
    unsigned int d = 0;
    bool result = true;
    // compare starting with the most significant dimension
    while (result && d < D) {
      result = result && (c1.r[d] < c2.r[d]);
    }
    return result;
  }
  template <typename T, unsigned int D, typename Tag>
  bool operator==(const LatticeCoordinate<T,D,Tag>& c1, const LatticeCoordinate<T,D,Tag>& c2) {
    return c1.r == c2.r;
  }

  template <typename T, unsigned int D, typename Tag>
  std::ostream& operator<<(std::ostream& output, const LatticeCoordinate<T,D,Tag>& c) {
    output << "{";
    for (unsigned int dim = 0; dim < D-1; ++dim)
      output << c[dim] << ", ";
    output << c[D-1] << "}";
    return output;
  }


};

#endif /*NUMERIC_H_*/
