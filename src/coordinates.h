#ifndef NUMERIC_H_
#define NUMERIC_H_

#include <vector>
#include <boost/operators.hpp>

using namespace boost;

namespace TiledArray {

template <typename T, unsigned int D, typename Tag>
class ArrayCoordinate;

template <typename T, unsigned int D, typename Tag>
bool operator<(const ArrayCoordinate<T,D,Tag>&, const ArrayCoordinate<T,D,Tag>&);

template <typename T, unsigned int D, typename Tag>
bool operator==(const ArrayCoordinate<T,D,Tag>& c1, const ArrayCoordinate<T,D,Tag>& c2);

template <typename T, unsigned int D, typename Tag>
std::ostream& operator<<(std::ostream& output, const ArrayCoordinate<T,D,Tag>& c);


  /// ArrayCoordinate represents coordinates of a point in a DIM-dimensional lattice.
  ///
  /// The purpose of Tag is to create multiple instances of the class
  /// with identical mathematical behavior but distinct types to allow
  /// overloading in classes using LatticePoint.
  template <typename T, unsigned int D, typename Tag>
  class ArrayCoordinate :
      boost::addable< ArrayCoordinate<T,D,Tag>,          // point + point
      boost::subtractable< ArrayCoordinate<T,D,Tag>,    // point - point
      boost::less_than_comparable1< ArrayCoordinate<T,D,Tag>,  // point < point
      boost::equality_comparable1< ArrayCoordinate<T,D,Tag>  // point == point
//      boost:incrementable< ArrayCoordinate<T,D,Tag>,        // ++point
//      boost ::decrementable< ArrayCoordinate<T,D,Tag>     // --point
//      >
//      >
      >
      >
      >
      >
  {
    public:
    typedef T Element;
    typedef typename std::vector<Element>::iterator iterator;
    typedef typename std::vector<Element>::const_iterator const_iterator;
    static const unsigned int DIM = D;
    
    ArrayCoordinate(const T& init_value = 0) : r_(D,init_value) {}
    ArrayCoordinate(const T* init_values) : r_(D) { std::copy(init_values,init_values+D,r_.begin()); }
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

    ArrayCoordinate<T, D, Tag>& operator++() {
      T& least_significant = *r_.rbegin();
      ++least_significant;
      return *this;
    }
    ArrayCoordinate<T, D, Tag>& operator--() {
      T& least_significant = *r_.rbegin();
      --least_significant;
      return *this;
    }
    
    /// Add operator
    ArrayCoordinate<T, D, Tag>& operator+=(const ArrayCoordinate& c) {
      for(unsigned int d = 0; d < DIM; ++d)
        r_[d] += c.r_[d];
      return *this;
    }

    /// Subtract operator
    ArrayCoordinate<T, D, Tag> operator-=(const ArrayCoordinate& c) {
      for(unsigned int d = 0; d < DIM; ++d)
        r_[d] -= c.r_[d];
      return *this;
    }

    ArrayCoordinate<T, D, Tag> operator -() const {
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

    friend bool operator < <>(const ArrayCoordinate<T,D,Tag>&, const ArrayCoordinate<T,D,Tag>&);
    friend bool operator == <>(const ArrayCoordinate<T,D,Tag>&, const ArrayCoordinate<T,D,Tag>&);
    friend std::ostream& operator << <>(std::ostream&, const ArrayCoordinate<T,D,Tag>&);
    
  private:
    /// last dimension is least significant
    /// default is to use standard vector
    std::vector<T> r_;
  };
  
  template <typename T, unsigned int D, typename Tag>
  bool operator<(const ArrayCoordinate<T,D,Tag>& c1, const ArrayCoordinate<T,D,Tag>& c2) {
    unsigned int d = 0;
    bool result = true;
    // compare starting with the most significant dimension
    while(result && d < D) {
      if (c1.r_[d] > c2.r_[d]) {
        result = false;
      }
      else {
        if (c1.r_[d] < c2.r_[d])
          d = D;
        else
          ++d;
      }          
    }
    return result;
  }

  template <typename T, unsigned int D, typename Tag>
  bool operator==(const ArrayCoordinate<T,D,Tag>& c1, const ArrayCoordinate<T,D,Tag>& c2) {
    return c1.r_ == c2.r_;
  }

  template <typename T, unsigned int D, typename Tag>
  std::ostream& operator<<(std::ostream& output, const ArrayCoordinate<T,D,Tag>& c) {
    output << "{";
    for(unsigned int dim = 0; dim < D - 1; ++dim)
      output << c[dim] << ", ";
    output << c[D - 1] << "}";
    return output;
  }


}

#endif /*NUMERIC_H_*/
