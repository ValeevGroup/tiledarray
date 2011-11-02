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
  template <unsigned int>
  class Permutation;
  template <unsigned int DIM>
  bool operator==(const Permutation<DIM>&, const Permutation<DIM>&);
  template <unsigned int DIM>
  std::ostream& operator<<(std::ostream&, const Permutation<DIM>&);

  namespace detail {
    template <typename InIter0, typename InIter1, typename RandIter>
    void permute(InIter0, InIter0, InIter1, RandIter);
  } // namespace detail

  /// Permutation

  /// Permutation class is used as an argument in all permutation operations on
  /// other objects. Permutations are performed with the following syntax:
  ///
  ///   b = p ^ a; // assign permeation of a into b given the permutation p.
  ///   a ^= p;    // permute a given the permutation p.
  template <unsigned int DIM>
  class Permutation {
  private:

    // Used to select the correct constructor based on input template types
    struct Enabler { };

  public:
    typedef Permutation<DIM> Permutation_;
    typedef std::size_t index_type;
    typedef std::array<index_type,DIM> Array;
    typedef typename Array::const_iterator const_iterator;

    static unsigned int dim() { return DIM; }

    static const Permutation& unit() { return unit_permutation; }

    Permutation() {
      p_ = unit_permutation.p_;
    }

    template <typename InIter>
    Permutation(InIter first, typename madness::enable_if<detail::is_input_iterator<InIter>, Enabler >::type = Enabler()) {
      for(std::size_t d = 0; d < DIM; ++d, ++first)
        p_[d] = *first;
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }

    Permutation(const Array& source) : p_(source) {
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }

    Permutation(const Permutation& other) : p_(other.p_) { }


    Permutation(index_type v) {
      p_[0] = v;
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }

    Permutation(const index_type p0, const index_type p1, ...) {
      std::fill_n(p_.begin(), DIM, 0ul);
      va_list ap;
      va_start(ap, p1);

      p_[0] = p0;
      p_[1] = p1;
      index_type pi = 0; // ci is used as an intermediate
      for(unsigned int i = 2; i < DIM; ++i) {
        pi = va_arg(ap, index_type);
        p_[i] = pi;
      }

      va_end(ap);
      TA_ASSERT( valid_(p_.begin(), p_.end()) );
    }

    ~Permutation() {}

    const_iterator begin() const { return p_.begin(); }
    const_iterator end() const { return p_.end(); }

    const index_type& operator[](unsigned int i) const {
      return p_[i];
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

    template <typename Archive>
    void serialize(Archive& ar) {
      ar & p_;
    }

    friend bool operator== <> (const Permutation<DIM>& p1, const Permutation<DIM>& p2);
    friend std::ostream& operator<< <> (std::ostream& output, const Permutation& p);

  private:
    static Permutation unit_permutation;

    // return false if this is not a valid permutation
    template <typename InIter>
    bool valid_(InIter first, InIter last) {
      Array count;
      std::fill(count.begin(), count.end(), 0);
      for(; first != last; ++first) {
        const index_type& i = *first;
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
      return Permutation<DIM>(_result);
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

  namespace detail {

    /// Copies an iterator range into an array type container.

    /// Permutes iterator range  \c [first_o, \c last_o) base on the permutation
    /// \c [first_p, \c last_p) and places it in \c result. The result object
    /// must define operator[](std::size_t).
    /// \arg \c [first_p, \c last_p) is an iterator range to the permutation
    /// \arg \c [irst_o is an input iterator to the beginning of the original array
    /// \arg \c result is a random access iterator that will contain the resulting permuted array
    template <typename InIter0, typename InIter1, typename RandIter>
    void permute_array(InIter0 first_p, InIter0 last_p, InIter1 first_o, RandIter first_r) {
      TA_STATIC_ASSERT(detail::is_input_iterator<InIter0>::value);
      TA_STATIC_ASSERT(detail::is_input_iterator<InIter1>::value);
      TA_STATIC_ASSERT(detail::is_random_iterator<RandIter>::value);
      for(; first_p != last_p; ++first_p)
        first_r[*first_p] = *first_o++;
    }

  } // namespace detail

  /// permute a std::array
  template <unsigned int DIM, typename T>
  std::array<T,DIM> operator^(const Permutation<DIM>& perm, const std::array<T, static_cast<std::size_t>(DIM) >& orig) {
    std::array<T,DIM> result;
    detail::permute_array(perm.begin(), perm.end(), orig.begin(), result.begin());
    return result;
  }

  /// permute a std::vector<T>
  template <unsigned int DIM, typename T, typename A>
  std::vector<T> operator^(const Permutation<DIM>& perm, const std::vector<T, A>& orig) {
    TA_ASSERT((orig.size() == DIM));
    std::vector<T> result(DIM);
    detail::permute_array<typename Permutation<DIM>::const_iterator, typename std::vector<T, A>::const_iterator, typename std::vector<T, A>::iterator>
      (perm.begin(), perm.end(), orig.begin(), result.begin());
    return result;
  }

  template <unsigned int DIM, typename T>
  std::vector<T> operator^=(std::vector<T>& orig, const Permutation<DIM>& perm) {
    orig = perm ^ orig;

    return orig;
  }

  template<unsigned int DIM>
  Permutation<DIM> operator ^(const Permutation<DIM>& perm, const Permutation<DIM>& p) {
    Permutation<DIM> result(perm ^ p.data());
    return result;
  }

  template <unsigned int DIM, typename T>
  std::array<T,DIM> operator ^=(std::array<T, static_cast<std::size_t>(DIM) >& a, const Permutation<DIM>& perm) {
    return (a = perm ^ a);
  }


} // namespace TiledArray

#endif // TILEDARRAY_PERMUTATION_H__INCLUED
