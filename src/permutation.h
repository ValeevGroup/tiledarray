#ifndef PERMUTATION_H_
#define PERMUTATION_H_

#include <cassert>
#include <iostream>
#include <algorithm>
#include <stdarg.h>
#include <boost/array.hpp>

namespace TiledArray {

  // weirdly necessary forward declarations
  template <unsigned int D>
  class Permutation;
  template <unsigned int D>
  bool operator==(const Permutation<D>& p1, const Permutation<D>& p2);
  template <unsigned int D>
  std::ostream& operator<<(std::ostream& output, const Permutation<D>& p);

  /// Permutation
  template <unsigned int D>
  class Permutation
  {
  public:
    typedef size_t Index;
    typedef boost::array<Index,D> Array;
    static const unsigned int DIM = D;

    static const Permutation& unit() { return unit_permutation; }

    Permutation() {
      p_ = unit_permutation.p_;
    }

    Permutation(const Index* source) {
      std::copy(source,source+D,p_.begin());
      assert(valid_permutation());
    }

    Permutation(const Array& source) : p_(source) {
      assert(valid_permutation());
    }

    Permutation(const Index p0, ...) {
      va_list ap;
      va_start(ap, p0);

      p_[0] = p0;
      for(unsigned int i = 1; i < D; ++i)
        p_[i] = va_arg(ap, Index);

      va_end(ap);

      assert(valid_permutation());
    }

    ~Permutation() {}

    const Index& operator[](unsigned int i) const {
#ifdef NDEBUG
      return p_[i];
#else
      return p_.at(i);
#endif
    }

    Permutation& operator=(const Permutation& other) { p_ = other.p_; return *this; }
    /// return *this * other
    Permutation operator^(const Permutation& other) const {
      Array _result;
      for(unsigned int d=0; d<D; ++d)
        _result[d] = p_[other[d]];
      Permutation result(_result);
      return result;
    }

    friend bool operator== <> (const Permutation<D>& p1, const Permutation<D>& p2);
    friend std::ostream& operator<< <> (std::ostream& output, const Permutation& p);

  private:
    static Permutation unit_permutation;

    // return false if this is not a valid permutation
    bool valid_permutation() {
      Array count;
      count.assign(0);
      for(unsigned int d=0; d < D; ++d) {
        const Index& i = p_[d];
        if(i >= D) return false;
        if(count[i] > 0) return false;
        ++count[i];
      }
      return true;
    }

    Array p_;
  };

  namespace {
    template <unsigned int D>
    Permutation<D>
    make_unit_permutation() {
      typename Permutation<D>::Index _result[D];
      for(unsigned int d=0; d<D; ++d) _result[d] = d;
      return Permutation<D>(_result);
    }
  }

  template <unsigned int D>
  Permutation<D> Permutation<D>::unit_permutation = make_unit_permutation<D>();

  template <unsigned int D>
  bool operator==(const Permutation<D>& p1, const Permutation<D>& p2) {
    return p1.p_ == p2.p_;
  }

  template <unsigned int D>
  bool operator!=(const Permutation<D>& p1, const Permutation<D>& p2) {
    return ! operator==(p1, p2);
  }

  template <unsigned int D>
  std::ostream& operator<<(std::ostream& output, const Permutation<D>& p) {
    output << "{";
    for (unsigned int dim = 0; dim < D-1; ++dim)
      output << dim << "->" << p.p_[dim] << ", ";
    output << D-1 << "->" << p.p_[D-1] << "}";
    return output;
  }

  /// permute an array
  template <unsigned int DIM, typename T>
  boost::array<T,DIM> operator^(const Permutation<DIM>& perm, const boost::array<T, static_cast<std::size_t>(DIM) >& orig) {
    boost::array<T,DIM> result;
    for(unsigned int dim = 0; dim < DIM; ++dim)
      result[perm[dim]] = orig[dim];
    return result;
  }

  template <unsigned int DIM, typename T>
  boost::array<T,DIM> operator ^=(boost::array<T, static_cast<std::size_t>(DIM) >& a, const Permutation<DIM>& perm) {
    return (a = perm ^ a);
  }

  /// Permute obj
  template <unsigned int DIM, typename Object>
  Object operator ^(const Permutation<DIM>& perm, const Object& obj) {
    Object result(obj);
    return result ^= perm;
  }

}

#endif /*PERMUTATION_H_*/
