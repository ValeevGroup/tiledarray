#ifndef PERMUTATION_H_
#define PERMUTATION_H_

#include <cassert>
#include <iosfwd>
#include <algorithm>
#include <stdarg.h>
#include <boost/array.hpp>

namespace TiledArray {

  // weirdly necessary forward declarations
  template <unsigned int DIM>
  class Permutation;
  template <unsigned int DIM>
  bool operator==(const Permutation<DIM>& p1, const Permutation<DIM>& p2);
  template <unsigned int DIM>
  std::ostream& operator<<(std::ostream& output, const Permutation<DIM>& p);

  // Boost forward declaration

  /// Permutation

  /// Permutation class is used as an argument in all permutation operations on
  /// other objects. Permutations are performed with the following syntax:
  ///
  ///   b = p ^ a; // assign permeation of a into b given the permutation p.
  ///   a ^= p;    // permute a given the permutation p.
  template <unsigned int DIM>
  class Permutation
  {
  public:
    typedef std::size_t index_type;
    typedef boost::array<index_type,DIM> Array;
    typedef typename Array::const_iterator const_iterator;

    static const unsigned int dim() { return DIM; }

    static const Permutation& unit() { return unit_permutation; }

    Permutation() {
      p_ = unit_permutation.p_;
    }

    template <typename InIter>
    Permutation(InIter first, InIter last) {
      for(typename Array::iterator it = p_.begin(); it != p_.end(); ++it, ++first) {
        assert(first != last);
        *it = *first;
      }

      assert(valid_permutation());
    }

    Permutation(const Array& source) : p_(source) {
      assert(valid_permutation());
    }

    Permutation(const index_type p0, ...) {
      va_list ap;
      va_start(ap, p0);

      p_[0] = p0;
      for(unsigned int i = 1; i < DIM; ++i)
        p_[i] = va_arg(ap, index_type);

      va_end(ap);

      assert(valid_permutation());
    }

    ~Permutation() {}

    const_iterator begin() const { return p_.begin(); }
    const_iterator end() { return p_.end(); }

    const index_type& operator[](unsigned int i) const {
#ifdef NDEBUG
      return p_[i];
#else
      return p_.at(i);
#endif
    }

    Permutation<DIM>& operator=(const Permutation<DIM>& other) { p_ = other.p_; return *this; }
    /// return *this * other
    Permutation<DIM>& operator^=(const Permutation<DIM>& other) {
      p_ ^= other;
      return *this;
    }

    /// Returns the reverse permutation and will satisfy the following conditions.
    /// given c2 = p ^ c1
    /// c1 == ((-p) ^ c2);
    Permutation operator -() const {
      return *this ^ unit();
    }

    /// Return a reference to the array that represents the permutation.
    Array& data() { return p_; }
    const Array& data() const { return p_; }

    friend bool operator== <> (const Permutation<DIM>& p1, const Permutation<DIM>& p2);
    friend std::ostream& operator<< <> (std::ostream& output, const Permutation& p);

  private:
    static Permutation unit_permutation;

    // return false if this is not a valid permutation
    bool valid_permutation() {
      Array count;
      count.assign(0);
      for(unsigned int d=0; d < DIM; ++d) {
        const index_type& i = p_[d];
        if(i >= DIM) return false;
        if(count[i] > 0) return false;
        ++count[i];
      }
      return true;
    }

    Array p_;
  };

  namespace {
    template <unsigned int DIM>
    Permutation<DIM>
    make_unit_permutation() {
      typename Permutation<DIM>::index_type _result[DIM];
      for(unsigned int d=0; d<DIM; ++d) _result[d] = d;
      return Permutation<DIM>(_result, _result + DIM);
    }
  }

  template <unsigned int DIM>
  Permutation<DIM> Permutation<DIM>::unit_permutation = make_unit_permutation<DIM>();

  template <unsigned int DIM>
  bool operator==(const Permutation<DIM>& p1, const Permutation<DIM>& p2) {
    return p1.p_ == p2.p_;
  }

  template <unsigned int DIM>
  bool operator!=(const Permutation<DIM>& p1, const Permutation<DIM>& p2) {
    return ! operator==(p1, p2);
  }

  template <unsigned int DIM>
  std::ostream& operator<<(std::ostream& output, const Permutation<DIM>& p) {
    output << "{";
    for (unsigned int dim = 0; dim < DIM-1; ++dim)
      output << dim << "->" << p.p_[dim] << ", ";
    output << DIM-1 << "->" << p.p_[DIM-1] << "}";
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
