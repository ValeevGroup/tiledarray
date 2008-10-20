#ifndef PERMUTATION_H_
#define PERMUTATION_H_

#include <vector>
#include <cassert>
#include <iostream>
#include <algorithm>

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
    static const unsigned int DIM = D;

    static const Permutation& unit() { return unit_permutation; }
    
    Permutation(const Index* source) : p_(DIM) {
      std::copy(source,source+D,p_.begin());
      assert(valid_permutation());
    }
    Permutation(const std::vector<Index>& source) : p_(source) {
      assert(valid_permutation());
    }
    ~Permutation() {}
    
    const Index& operator[](unsigned int i) const {
      return p_[i];
    }
    
    Permutation& operator=(const Permutation& other) { p_ = other.p_; return *this; }
    /// return *this * other
    Permutation operator^(const Permutation& other) const {
      std::vector<typename Permutation<D>::Index > _result(D);
      for(unsigned int d=0; d<D; ++d)
        _result[d] = p_[other[d]];
      return Permutation(_result);
    }
    
    friend bool operator== <> (const Permutation<D>& p1, const Permutation<D>& p2);
    friend std::ostream& operator<< <> (std::ostream& output, const Permutation& p);
    
    private:
    Permutation();
    static Permutation unit_permutation;

    // return false if this is not a valid permutation
    bool valid_permutation() {
      std::vector<unsigned int> count(D,0);
      for(unsigned int d=0; d<D; ++d) {
        const Index& i = p_[d];
        if (i>=D) return false;
        if (count[i] > 0) return false;
        ++count[i];
      }
      return true;
    }
    
    std::vector<Index> p_;
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
  std::ostream& operator<<(std::ostream& output, const Permutation<D>& p) {
    output << "{";
    for (unsigned int dim = 0; dim < D-1; ++dim)
      output << dim << "->" << p.p_[dim] << ", ";
    output << D-1 << "->" << p.p_[D-1] << "}";
    return output;
  }

}

#endif /*PERMUTATION_H_*/
