#ifndef TILEDARRAY_PERMUTATION_H__INCLUED
#define TILEDARRAY_PERMUTATION_H__INCLUED

#include <TiledArray/error.h>
//#include <TiledArray/utility.h>
#include <TiledArray/type_traits.h>
#include <world/enable_if.h>
#include <world/array.h>
//#include <iosfwd>
//#include <algorithm>
#include <stdarg.h>
//#include <iterator>
//#include <functional>
#include <numeric>
#include <stdarg.h>

namespace TiledArray {

  // weirdly necessary forward declarations
  class Permutation;
  bool operator==(const Permutation&, const Permutation&);
  std::ostream& operator<<(std::ostream&, const Permutation&);

  namespace detail {

    /// Place holder object to represent a no permutation operation.
    struct NoPermutation {
      template <typename Archive>
      void serialize(Archive&) { TA_ASSERT(false); }
    };

    /// Copies an iterator range into an array type container.

    /// Permutes iterator range  \c [first_o, \c last_o) base on the permutation
    /// \c [first_p, \c last_p) and places it in \c result. The result object
    /// must define operator[](std::size_t).
    /// \arg \c [first_p, \c last_p) is an iterator range to the permutation
    /// \arg \c [irst_o is an input iterator to the beginning of the original array
    /// \arg \c result is a random access iterator that will contain the resulting permuted array
    template <typename InIter0, typename InIter1, typename RandIter>
    inline void permute_array(InIter0 first_p, InIter0 last_p, InIter1 first_o, RandIter first_r) {
      TA_STATIC_ASSERT(detail::is_input_iterator<InIter0>::value);
      TA_STATIC_ASSERT(detail::is_input_iterator<InIter1>::value);
      TA_STATIC_ASSERT(detail::is_random_iterator<RandIter>::value);
      for(; first_p != last_p; ++first_p)
        first_r[*first_p] = *first_o++;
    }
  } // namespace detail

  /// permute a std::array
  template <typename T, std::size_t N>
  inline std::array<T,N> operator^(const Permutation&, const std::array<T, N>&);

  /// permute a std::vector<T>
  template <typename T, typename A>
  inline std::vector<T> operator^(const Permutation&, const std::vector<T, A>&);

  template <typename T, typename A>
  inline std::vector<T> operator^=(std::vector<T, A>&, const Permutation&);

  inline Permutation operator ^(const Permutation&, const Permutation&);

  template <typename T, std::size_t N>
  inline std::array<T,N> operator ^=(std::array<T,N>&, const Permutation&);

  /// Permutation

  /// Permutation class is used as an argument in all permutation operations on
  /// other objects. Permutations are performed with the following syntax:
  ///
  ///   b = p ^ a; // assign permeation of a into b given the permutation p.
  ///   a ^= p;    // permute a given the permutation p.
  class Permutation {
  private:

    // Used to select the correct constructor based on input template types
    struct Enabler { };

  public:
    typedef Permutation Permutation_;
    typedef std::size_t index_type;
    typedef std::vector<index_type> Array;
    typedef Array::const_iterator const_iterator;

    unsigned int dim() const { return p_.size(); }

    Permutation() : p_() { }

    template <typename InIter>
    Permutation(InIter first, InIter last,
        typename madness::enable_if<detail::is_input_iterator<InIter>, Enabler >::type = Enabler()) :
        p_(first, last)
    {
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }

    template <typename A>
    Permutation(const A& a) : p_(a.begin(), a.end()) {
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }

    Permutation(const Permutation& other) : p_(other.p_) { }


    Permutation(const index_type v) : p_(1,v) {
      TA_ASSERT( v == 0 ); // The only valid value
    }

    Permutation(const index_type p0, const index_type p1) : p_(2) {
      p_[0] = p0;
      p_[1] = p1;
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }
    Permutation(const index_type p0, const index_type p1, const index_type p2) : p_(3) {
      p_[0] = p0;
      p_[1] = p1;
      p_[2] = p2;
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }

    Permutation(const index_type p0, const index_type p1, const index_type p2,
        const index_type p3) : p_(4) {
      p_[0] = p0;
      p_[1] = p1;
      p_[2] = p2;
      p_[3] = p3;
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }

    Permutation(const index_type p0, const index_type p1, const index_type p2,
        const index_type p3, const index_type p4) : p_(5) {
      p_[0] = p0;
      p_[1] = p1;
      p_[2] = p2;
      p_[3] = p3;
      p_[4] = p4;
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }

    Permutation(const index_type p0, const index_type p1, const index_type p2,
        const index_type p3, const index_type p4, const index_type p5) : p_(6) {
      p_[0] = p0;
      p_[1] = p1;
      p_[2] = p2;
      p_[3] = p3;
      p_[4] = p4;
      p_[5] = p5;
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }

    Permutation(const index_type p0, const index_type p1, const index_type p2,
        const index_type p3, const index_type p4, const index_type p5, const index_type p6) : p_(7) {
      p_[0] = p0;
      p_[1] = p1;
      p_[2] = p2;
      p_[3] = p3;
      p_[4] = p4;
      p_[5] = p5;
      p_[6] = p6;
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }

    Permutation(const index_type p0, const index_type p1, const index_type p2,
        const index_type p3, const index_type p4, const index_type p5, const index_type p6,
        const index_type p7) : p_(8) {
      p_[0] = p0;
      p_[1] = p1;
      p_[2] = p2;
      p_[3] = p3;
      p_[4] = p4;
      p_[5] = p5;
      p_[6] = p6;
      p_[7] = p7;
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }

    Permutation(const index_type p0, const index_type p1, const index_type p2,
        const index_type p3, const index_type p4, const index_type p5, const index_type p6,
        const index_type p7, const index_type p8) : p_(9) {
      p_[0] = p0;
      p_[1] = p1;
      p_[2] = p2;
      p_[3] = p3;
      p_[4] = p4;
      p_[5] = p5;
      p_[6] = p6;
      p_[7] = p7;
      p_[8] = p8;
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }

    Permutation(const index_type p0, const index_type p1, const index_type p2,
        const index_type p3, const index_type p4, const index_type p5, const index_type p6,
        const index_type p7, const index_type p8, const index_type p9) : p_(10) {
      p_[0] = p0;
      p_[1] = p1;
      p_[2] = p2;
      p_[3] = p3;
      p_[4] = p4;
      p_[5] = p5;
      p_[6] = p6;
      p_[7] = p7;
      p_[8] = p8;
      p_[9] = p9;
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }


    ~Permutation() {}

    const_iterator begin() const { return p_.begin(); }
    const_iterator end() const { return p_.end(); }

    const index_type& operator[](unsigned int i) const {
      return p_[i];
    }

    Permutation& operator=(const Permutation& other) { p_ = other.p_; return *this; }

    /// return *this ^ other
    Permutation& operator^=(const Permutation& other) {
      p_ ^= other;
      return *this;
    }

    /// Returns the reverse permutation and will satisfy the following conditions.
    /// given c2 = p ^ c1
    /// c1 == ((-p) ^ c2);
    Permutation operator -() const {
      return *this ^ make_unit_permutation(dim());
    }

    /// Return a reference to the array that represents the permutation.
    const Array& data() const { return p_; }

    template <typename Archive>
    void serialize(Archive& ar) {
      ar & p_;
    }

  private:

    static Permutation make_unit_permutation(std::size_t n) {
      Permutation::Array _result(n);
      for(unsigned int d = 0; d < n; ++d) _result[d] = d;
      return Permutation(_result);
    }

    // return false if this is not a valid permutation
    template <typename InIter>
    bool valid_(InIter first, InIter last) {
      Array count(std::distance(first, last));
      std::fill(count.begin(), count.end(), 0);
      for(; first != last; ++first) {
        const index_type& i = *first;
        if(count[i] > 0) return false;
        ++count[i];
      }
      return true;
    }

    Array p_;
  };

  inline bool operator==(const Permutation& p1, const Permutation& p2) {
    return (p1.dim() == p2.dim()) && std::equal(p1.data().begin(), p1.data().end(), p2.data().begin());
  }

  inline bool operator!=(const Permutation& p1, const Permutation& p2) {
    return ! operator==(p1, p2);
  }

  inline std::ostream& operator<<(std::ostream& output, const Permutation& p) {
    std::size_t n = p.dim();
    output << "{";
    for (unsigned int dim = 0; dim < n - 1; ++dim)
      output << dim << "->" << p.data()[dim] << ", ";
    output << n - 1 << "->" << p.data()[n - 1] << "}";
    return output;
  }

  /// permute a std::array
  template <typename T, std::size_t N>
  inline std::array<T,N> operator^(const Permutation& perm, const std::array<T, N>& orig) {
    TA_ASSERT(perm.dim() == orig.size());
    std::array<T,N> result;
    detail::permute_array(perm.begin(), perm.end(), orig.begin(), result.begin());
    return result;
  }

  /// permute a std::vector<T>
  template <typename T, typename A>
  inline std::vector<T> operator^(const Permutation& perm, const std::vector<T, A>& orig) {
    TA_ASSERT(orig.size() == perm.dim());
    std::vector<T> result(perm.dim());
    detail::permute_array<Permutation::const_iterator, typename std::vector<T, A>::const_iterator, typename std::vector<T, A>::iterator>
      (perm.begin(), perm.end(), orig.begin(), result.begin());
    return result;
  }



  inline Permutation operator ^(const Permutation& perm, const Permutation& p) {
    TA_ASSERT(perm.dim() == p.dim());
    return Permutation(perm ^ p.data());
  }

  namespace detail {
    /// permute a std::array
    template <typename T, std::size_t N>
    inline const std::array<T,N>& operator^(const NoPermutation&, const std::array<T, N>& orig) {
      return orig;
    }

    /// permute a std::vector<T>
    template <typename T, typename A>
    inline const std::vector<T>& operator^(const NoPermutation&, const std::vector<T, A>& orig) {
      return orig;
    }

    inline const Permutation& operator ^(const NoPermutation&, const Permutation& p) {
      return p;
    }

  } // namespace detail

  template <typename T, std::size_t N>
  inline std::array<T,N> operator ^=(std::array<T,N>& a, const Permutation& perm) {
    TA_ASSERT(perm.dim() == N);
    return (a = perm ^ a);
  }

  template <typename T, typename A>
  inline std::vector<T> operator^=(std::vector<T, A>& orig, const Permutation& perm) {
    TA_ASSERT(orig.size() == perm.dim());
    orig = perm ^ orig;

    return orig;
  }

} // namespace TiledArray

#endif // TILEDARRAY_PERMUTATION_H__INCLUED
