#ifndef TILEDARRAY_PERMUTATION_H__INCLUED
#define TILEDARRAY_PERMUTATION_H__INCLUED

#include <TiledArray/error.h>
#include <TiledArray/coordinate_system.h>
#include <TiledArray/utility.h>
#include <TiledArray/array_util.h>
//#include <iosfwd>
//#include <algorithm>
#include <vector>
#include <stdarg.h>
//#include <iterator>
//#include <functional>
#include <numeric>
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
    BOOST_STATIC_ASSERT(DIM < TA_MAX_DIM);

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
    Permutation(InIter first) {
      // should variadic constructor been chosen?
      // need to disambiguate the call if DIM==1
      // assume iterators if InIter is not an integral type
      // else assume wanted variadic constructor
      // this scheme follows what std::vector does
      BOOST_STATIC_ASSERT((DIM == 1u) || (! boost::is_integral<InIter>::value));
      detail::initialize_from_values(first, p_.begin(), DIM, boost::is_integral<InIter>());
      TA_ASSERT( valid_(p_.begin(), p_.end()) , std::runtime_error,
          "Invalid permutation supplied." );
    }

    Permutation(const Array& source) : p_(source) {
      TA_ASSERT( valid_(p_.begin(), p_.end()) , std::runtime_error,
          "Invalid permutation supplied.");
    }

    Permutation(const Permutation& other) : p_(other.p_) { }

#ifdef __GXX_EXPERIMENTAL_CXX0X__
    template <typename... Params>
    Permutation(Params... params) {
      BOOST_STATIC_ASSERT(detail::Count<Params...>::value == DIM);
      BOOST_STATIC_ASSERT(detail::is_integral_list<Params...>::value);
      detail::fill(p_.begin(), params...);
    }
#else
    Permutation(const index_type p0, ...) {
      va_list ap;
      va_start(ap, p0);

      p_[0] = p0;
      for(unsigned int i = 1; i < DIM; ++i)
        p_[i] = va_arg(ap, index_type);

      va_end(ap);

      TA_ASSERT( valid_(begin(), end()) , std::runtime_error,
          "Invalid permutation supplied.");
    }
#endif // __GXX_EXPERIMENTAL_CXX0X__

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
    void permute(InIter0 first_p, InIter0 last_p, InIter1 first_o, RandIter first_r) {
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter0>::value);
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter1>::value);
      BOOST_STATIC_ASSERT(detail::is_random_iterator<RandIter>::value);
      for(; first_p != last_p; ++first_p)
        (* (first_r + *first_p)) = *first_o++;
    }

    /// ForLoop defines a for loop operation for a random access iterator.
    template<typename F, typename RandIter>
    struct ForLoop {
    private:
      BOOST_STATIC_ASSERT(detail::is_random_iterator<RandIter>::value);
    public:
      typedef F Functor;
      typedef typename std::iterator_traits<RandIter>::value_type value_t;
      typedef typename std::iterator_traits<RandIter>::difference_type diff_t;
      /// Construct a ForLoop

      /// \arg \c f is the function to be executed on each loop iteration.
      /// \arg \c n is the end point offset from the starting point (i.e. last =
      /// first + n;).
      /// \arg \c s is the step size for the loop (optional).
      ForLoop(F f, diff_t n, diff_t s = 1ul) : func_(f), n_(n), step_(s) { }

      /// Execute the loop given a random access iterator as the starting point.
      // Do not pass iterator by reference here since we will be modifying it
      // in the function.
      void operator ()(RandIter first) {
        const RandIter end = first + n_;
        // Use < operator because first will not always land on end_.
        for(; first < end; first += step_)
          func_(first);
      }

    private:
      F func_;      ///< Function to run on each loop iteration
      diff_t n_;    ///< End of the iterator range
      diff_t step_; ///< Step size for the iterator
    }; // struct perm_range

    /// NestedForLoop constructs and executes a nested for loop object.
    template<unsigned int DIM, typename F, typename RandIter>
    struct NestedForLoop : public NestedForLoop<DIM - 1, ForLoop<F, RandIter>, RandIter > {
    private:
      BOOST_STATIC_ASSERT(detail::is_random_iterator<RandIter>::value);
    public:
      typedef ForLoop<F, RandIter> F1;
      typedef NestedForLoop<DIM - 1, F1, RandIter> NestedForLoop1;

      /// Constructs the nested for loop object

      /// \arg \c func is the current loops function body.
      /// \arg \c [e_first, \c e_last) is the end point offset for the current
      /// loop and subsequent loops.
      /// \arg \c [s_first, \c s_last) is the step size for the current loop and
      /// subsequent loops.
      template<typename InIter>
      NestedForLoop(F func, InIter e_first, InIter e_last, InIter s_first, InIter s_last) :
        NestedForLoop1(F1(func, *e_first, *s_first), e_first + 1, e_last, s_first + 1, s_last)
      {
        BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      }

      /// Run the nested loop (call the next higher loop object).
      void operator()(RandIter it) { NestedForLoop1::operator ()(it); }
    }; // struct NestedForLoop

    /// NestedForLoop constructs and executes a nested for loop object.

    /// This specialization represents the terminating step for constructing the
    /// nested loop object, stores the loop object and runs the object.
    template<typename F, typename RandIter>
    struct NestedForLoop<0, F, RandIter> {
    private:
      BOOST_STATIC_ASSERT(detail::is_random_iterator<RandIter>::value);
    public:

      /// \arg \c func is the current loops function body.
      /// \arg \c [e_first, \c e_last) is the end point offset for the loops.
      /// first and last should be equal here.
      /// \arg \c [s_first, \c s_last) is the step size for the loops. first and
      /// last should be equal here.
      template<typename InIter>
      NestedForLoop(F func, InIter, InIter, InIter, InIter) : f_(func)
      { }

      /// Run the actual loop object
      void operator()(RandIter it) { f_(it); }
    private:
      F f_;
    }; // struct NestedForLoop

    /// Function object that assigns the content of one iterator to another iterator.
    template<typename OutIter, typename InIter>
    struct AssignmentOp {
    private:
      BOOST_STATIC_ASSERT(detail::is_output_iterator<OutIter>::value);
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
    public:
      AssignmentOp(InIter first, InIter last) : current_(first), last_(last) { }

      void operator ()(OutIter& it) {
        TA_ASSERT(current_ != last_, std::runtime_error,
            "The iterator is the end of the range.");
        *it = *current_++;
      }

    private:
      AssignmentOp();
      InIter current_;
      InIter last_;
    }; // struct AssignmentOp

    /// Function object that assigns the content of one iterator to another iterator.

    /// This specialization uses a pointer explicitly.
    template<typename T, typename InIter>
    struct AssignmentOp<T*, InIter> {
    private:
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
    public:
      AssignmentOp(InIter first, InIter last) : current_(first), last_(last) { }

      void operator ()(T* it) {
        TA_ASSERT(current_ != last_, std::runtime_error,
            "The iterator is the end of the range.");
        *it = *current_++;
      }

    private:
      AssignmentOp();
      InIter current_;
      InIter last_;
    }; // struct AssignmentOp

    /// Permutes of n-dimensional container

    /// This function object permutes an n-dimensional container. The container
    /// must define the following functions: size(), weight(), volume(),
    /// begin(), and end(). It must also define the const_iterator type.
    template<typename Cont>
    struct Permute {
      /// Construct a permute function object. \c c is the container that will
      /// permuted.
      Permute(const Cont& c) : cont_(c) { }

      /// Perform the permutation and place the resulting permuted n-dimensional
      /// array in a new array starting at \c it. It must be a random access
      /// iterator, and the distance between first and last iterators must be
      /// equal to the volume of the original container.
      /// \arg \c p is the permutation that will be applied to the original container.
      /// \arg \c [first, \c last) is the iterator range for the resulting array.
      template<unsigned int DIM, typename RandIter>
      void operator ()(const Permutation<DIM>& p, RandIter first, RandIter last) {
        BOOST_STATIC_ASSERT(detail::is_random_iterator<RandIter>::value);
        typedef boost::array<typename Cont::ordinal_type, static_cast<std::size_t>(DIM)> size_array;
        TA_ASSERT(static_cast<typename Cont::volume_type>(std::distance(first, last)) == cont_.volume(),
            std::runtime_error,
            "The distance between first and last must be equal to the volume of the original container.");

        // Calculate the sizes and weights of the permuted array
        size_array p_size;
        permute(p.begin(), p.end(), cont_.size().begin(), p_size.begin());
        size_array weight;
        if(cont_.order() == decreasing_dimension_order)
          calc_weight(p_size.rbegin(), p_size.rend(), weight.rbegin());
        else
          calc_weight(p_size.begin(), p_size.end(), weight.begin());

        // Calculate the step sizes for the nested loops
        Permutation<DIM> ip = -p;
        size_array step;
        permute(ip.begin(), ip.end(), weight.begin(), step.begin());

        // Calculate the loop end offsets.
        size_array end;
        std::transform(cont_.size().begin(), cont_.size().end(), step.begin(),
            end.begin(), std::multiplies<typename Cont::ordinal_type>());

        if(cont_.order() == decreasing_dimension_order) {
          NestedForLoop<DIM, AssignmentOp<RandIter, typename Cont::const_iterator >, RandIter >
              do_loop(AssignmentOp<RandIter, typename Cont::const_iterator >(cont_.begin(), cont_.end()),
              end.rbegin(), end.rend(), step.rbegin(), step.rend());
          do_loop(first);
        } else {
          p_size.front() = std::accumulate(weight.begin(), weight.end(), 1ul, std::multiplies<typename Cont::ordinal_type>()) + 1ul;
          NestedForLoop<DIM, AssignmentOp<RandIter, typename Cont::const_iterator >, RandIter >
              do_loop(AssignmentOp<RandIter, typename Cont::const_iterator >(cont_.begin(), cont_.end()),
              end.begin(), end.end(), step.begin(), step.end());
          do_loop(first);
        }
      }

    private:

      Permute();

      const Cont& cont_;
    }; // struct Permute

  } // namespace detail

  /// permute a boost::array
  template <unsigned int DIM, typename T>
  boost::array<T,DIM> operator^(const Permutation<DIM>& perm, const boost::array<T, static_cast<std::size_t>(DIM) >& orig) {
    boost::array<T,DIM> result;
    detail::permute(perm.begin(), perm.end(), orig.begin(), result.begin());
    return result;
  }

  /// permute a std::vector<T>
  template <unsigned int DIM, typename T>
  std::vector<T> operator^(const Permutation<DIM>& perm, const std::vector<T>& orig) {
    TA_ASSERT((orig.size() == DIM), std::runtime_error,
        "The permutation dimension is not equal to the vector size.");
    std::vector<T> result(DIM);
    detail::permute<typename Permutation<DIM>::const_iterator, typename std::vector<T>::const_iterator, typename std::vector<T>::iterator>
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
  boost::array<T,DIM> operator ^=(boost::array<T, static_cast<std::size_t>(DIM) >& a, const Permutation<DIM>& perm) {
    return (a = perm ^ a);
  }

} // namespace TiledArray

#endif // TILEDARRAY_PERMUTATION_H__INCLUED
