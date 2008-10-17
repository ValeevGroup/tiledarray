#ifndef NUMERIC_H_
#define NUMERIC_H_

#include <vector>
#include <boost/operators.hpp>

using namespace boost;

namespace TiledArray {

template <typename T, unsigned int D, typename Tag>
class LatticeCoordinate;

template <typename T, unsigned int D, typename Tag>
bool operator<(const LatticeCoordinate<T,D,Tag>&, const LatticeCoordinate<T,D,Tag>&);

template <typename T, unsigned int D, typename Tag>
bool operator==(const LatticeCoordinate<T,D,Tag>& c1, const LatticeCoordinate<T,D,Tag>& c2);

template <typename T, unsigned int D, typename Tag>
std::ostream& operator<<(std::ostream& output, const LatticeCoordinate<T,D,Tag>& c);


  /// LatticeCoordinate represents coordinates of a point in a DIM-dimensional lattice.
  ///
  /// The purpose of Tag is to create multiple instances of the class
  /// with identical mathematical behavior but distinct types to allow
  /// overloading in classes using LatticePoint.
  template <typename T, unsigned int D, typename Tag>
  class LatticeCoordinate :
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
    typedef std::vector<Element>::iterator iterator;
    typedef std::vector<Element>::const_iterator const_iterator;
    static const unsigned int DIM = D;
    
    LatticeCoordinate(const T& init_value = 0) : r_(D,init_value) {}
    ~LatticeCoordinate() {}

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

    LatticeCoordinate<T, D, Tag>& operator++() { ++(*r_.rbegin()); }
    LatticeCoordinate<T, D, Tag>& operator--() { --(*r_.rbegin()); }
    
    /// Add operator
    LatticeCoordinate<T, D, Tag>& operator+=(const LatticeCoordinate& c) const {
      for(unsigned int d = 0; d < DIM; ++d)
        r_[d] += c.r_[d];
      return *this;
    }

    /// Subtract operator
    LatticeCoordinate<T, D, Tag> operator-=(const LatticeCoordinate& c) const {
      for(unsigned int d = 0; d < DIM; ++d)
        r_[d] -= c.r_[d];
      return *this;
    }

    LatticeCoordinate<T, D, Tag> operator -() const {
      LatticeCoordinate<T, D, Tag> ret;
      for(unsigned int d = 0; d < DIM; ++d)
        ret[d] = -r_[d];
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

    friend bool operator < <>(const LatticeCoordinate<T,D,Tag>&, const LatticeCoordinate<T,D,Tag>&);
    friend bool operator == <>(const LatticeCoordinate<T,D,Tag>&, const LatticeCoordinate<T,D,Tag>&);
    friend std::ostream& operator << <>(std::ostream&, const LatticeCoordinate<T,D,Tag>&);
    
  private:
    /// last dimension is least significant
    /// default is to use standard vector
    std::vector<T> r_;
  };
  
  template <typename T, unsigned int D, typename Tag>
  bool operator<(const LatticeCoordinate<T,D,Tag>& c1, const LatticeCoordinate<T,D,Tag>& c2) {
    unsigned int d = 0;
    bool result = true;
    // compare starting with the most significant dimension
    while(result && d < D) {
      result = result && (c1.r_[d] < c2.r_[d]);
    }
    return result;
  }

  template <typename T, unsigned int D, typename Tag>
  bool operator==(const LatticeCoordinate<T,D,Tag>& c1, const LatticeCoordinate<T,D,Tag>& c2) {
    return c1.r_ == c2.r_;
  }

  template <typename T, unsigned int D, typename Tag>
  std::ostream& operator<<(std::ostream& output, const LatticeCoordinate<T,D,Tag>& c) {
    output << "{";
    for(unsigned int dim = 0; dim < D - 1; ++dim)
      output << c[dim] << ", ";
    output << c[D - 1] << "}";
    return output;
  }


};

#endif /*NUMERIC_H_*/
