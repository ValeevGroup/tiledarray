#ifndef PERMUTATION_H_
#define PERMUTATION_H_

#include <error.h>
#include <array_util.h>
//#include <iosfwd>
//#include <algorithm>
#include <vector>
//#include <stdarg.h>
//#include <iterator>
#include <boost/array.hpp>

namespace TiledArray {

  // weirdly necessary forward declarations
  template <unsigned int DIM>
  class Permutation;
  template <unsigned int DIM>
  bool operator==(const Permutation<DIM>&, const Permutation<DIM>&);
  template <unsigned int DIM>
  bool operator!=(const Permutation<DIM>&, const Permutation<DIM>&);
  template <unsigned int DIM>
  std::ostream& operator<<(std::ostream&, const Permutation<DIM>&);
  template <unsigned int DIM, typename T>
  boost::array<T,DIM> operator^(const Permutation<DIM>&, const boost::array<T, static_cast<std::size_t>(DIM) >&);
  template <unsigned int DIM, typename T>
  std::vector<T> operator^(const Permutation<DIM>&, const std::vector<T>&);
  template <unsigned int DIM, typename T>
  std::vector<T> operator^=(std::vector<T>&, const Permutation<DIM>&);
  template <unsigned int DIM, typename T>
  boost::array<T,DIM> operator ^=(boost::array<T, static_cast<std::size_t>(DIM) >&, const Permutation<DIM>&);
  template<unsigned int DIM>
  Permutation<DIM> operator ^(const Permutation<DIM>&, const Permutation<DIM>&);

  namespace detail {
    template <typename InIter0, typename InIter1, typename Cont>
    Cont& permute(InIter0, InIter0, InIter1, InIter1, Cont&);
  } // namespace detail

  // Boost forward declaration

  /// Permutation

  /// Permutation class is used as an argument in all permutation operations on
  /// other objects. Permutations are performed with the following syntax:
  ///
  ///   b = p ^ a; // assign permeation of a into b given the permutation p.
  ///   a ^= p;    // permute a given the permutation p.
  template <unsigned int DIM>
  class Permutation {
  public:
    typedef Permutation<DIM> Permutation_;
    typedef std::size_t index_type;
    typedef boost::array<index_type,DIM> Array;
    typedef typename Array::const_iterator const_iterator;

    static unsigned int dim() { return DIM; }

    static const Permutation& unit() { return unit_permutation; }

    Permutation() {
      p_ = unit_permutation.p_;
    }

    template <typename InIter>
    Permutation(InIter first, InIter last) {
      TA_ASSERT( valid_(first, last) ,
          std::runtime_error("Permutation::Permutation(...): invalid permutation supplied") );
      TA_ASSERT( std::distance(first, last),
          std::runtime_error("Permutation::Permutation(...): iterator range [first, last) is too short") );
      for(typename Array::iterator it = p_.begin(); it != p_.end(); ++it, ++first)
        *it = *first;
    }

    Permutation(const Array& source) : p_(source) {
      TA_ASSERT( valid_(source.begin(), source.end()) ,
          std::runtime_error("Permutation::Permutation(...): invalid permutation supplied") );
    }

    Permutation(const Permutation& other) : p_(other.p_) { }

    Permutation(const index_type p0, ...) {
      va_list ap;
      va_start(ap, p0);

      p_[0] = p0;
      for(unsigned int i = 1; i < DIM; ++i)
        p_[i] = va_arg(ap, index_type);

      va_end(ap);

      TA_ASSERT( valid_(begin(), end()) ,
          std::runtime_error("Permutation::Permutation(...): invalid permutation supplied") );
    }

    ~Permutation() {}

    const_iterator begin() const { return p_.begin(); }
    const_iterator end() const { return p_.end(); }

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
    template <typename InIter>
    bool valid_(InIter first, InIter last) {
      Array count;
      count.assign(0);
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

  namespace detail {

    template <typename InIter0, typename InIter1, typename Cont>
    Cont& permute(InIter0 first_p, InIter0 last_p, InIter1 first_o, InIter1 last_o, Cont& result) {
      for(; first_p != last_p && first_o != last_o; ++first_p, ++first_o) {
        result[*first_p] = *first_o;
      }

      return result;
    }

    template<typename T, typename D = std::size_t>
    class perm_range {
    public:
      typedef T value_type;
      typedef D diff_type;

      perm_range(const diff_type step, const value_type end) :
          step_(step), end_(end) { }

      void initialize(const value_type start) { current_ = start; }
      value_type current() const { return current_; }
      bool end() const { return current_ < end_; }
      void increment() { current_ += step_; }

    private:
      perm_range();
      perm_range(const perm_range&);

      const diff_type step_;
      const value_type end_;
      value_type current_;
    }; // class perm_range

    template<typename T, typename D>
    class perm_iterator {
      typedef T value_type;
      typedef D diff_type;
      typedef perm_range<T,D> range_type;

      template<typename InIterS, typename InIterP>
      perm_iterator(InIterS first_s, InIterS last_s, InIterP first_p, InIterP last_p, const value_type start) :
          start_(start), current_(std::distance(first_s, last_s) - 1)
      {
        std::vector<diff_type> weight(std::distance(first_s, last_s));
        calc_weight(first_s, last_s, weight.begin());

        std::vector<diff_type> size_p(weight.size());
        permute(first_s, last_s, first_p, last_p, size_p);

        std::vector<diff_type> weight_p(weight.size());
        permute(weight.begin(), weight.end(), first_p, last_p, weight_p);

        typename std::vector<diff_type>::const_iterator w_it = weight_p.begin();
        typename std::vector<diff_type>::const_iterator s_it = size_p.begin();
        for(; s_it != size_p.end(); ++s_it, ++w_it)
          ranges_.push_back(range_type(*w_it, start + (*w_it) * (*s_it)));
      }

      void initialize() {
        for(typename std::vector<range_type>::iterator it = ranges_.begin(); it != ranges_.end(); ++it)
          *it->initialize(start_);
      }

      value_type current() { return ranges_.back().current(); }

      bool end() const {
        return ranges_.front().end();
      }

      void increment() {

      }

    private:
      std::vector<range_type> ranges_;
      const value_type start_;
      std::size_t current_;
    };

  } // namespace detail

  /// permute an array
  template <unsigned int DIM, typename T>
  boost::array<T,DIM> operator^(const Permutation<DIM>& perm, const boost::array<T, static_cast<std::size_t>(DIM) >& orig) {
    boost::array<T,DIM> result;
    return detail::permute(perm.begin(), perm.end(), orig.begin(), orig.end(), result);
  }

  /// permute an array
  template <unsigned int DIM, typename T>
  std::vector<T> operator^(const Permutation<DIM>& perm, const std::vector<T>& orig) {
    TA_ASSERT((orig.size() == DIM),
        std::runtime_error("operator^(const Permutation<DIM>&, const std::vector<T>&): The permutation dimension is not equal to the vector size."));
    std::vector<T> result(DIM);
    return detail::permute(perm.begin(), perm.end(), orig.begin(), orig.end(), result);
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
  boost::array<T,DIM> operator ^=(boost::array<T, static_cast<std::size_t>(DIM) >& a, const Permutation<DIM>& perm) {
    return (a = perm ^ a);
  }

} // namespace TiledArray

#endif /*PERMUTATION_H_*/
