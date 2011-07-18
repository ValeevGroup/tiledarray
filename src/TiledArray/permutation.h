#ifndef TILEDARRAY_PERMUTATION_H__INCLUED
#define TILEDARRAY_PERMUTATION_H__INCLUED

#include <TiledArray/error.h>
#include <TiledArray/coordinate_system.h>
#include <TiledArray/utility.h>
#include <world/enable_if.h>
#include <TiledArray/type_traits.h>
//#include <iosfwd>
//#include <algorithm>
#include <vector>
#include <stdarg.h>
//#include <iterator>
//#include <functional>
#include <numeric>
#include <stdarg.h>

namespace TiledArray {

  // weirdly necessary forward declarations
  template <typename>
  class Range;
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
      TA_ASSERT( valid_(p_.begin(), p_.end()) , std::runtime_error,
          "Invalid permutation supplied." );
    }

    Permutation(const Array& source) : p_(source) {
      TA_ASSERT( valid_(p_.begin(), p_.end()) , std::runtime_error,
          "Invalid permutation supplied.");
    }

    Permutation(const Permutation& other) : p_(other.p_) { }


    Permutation(index_type v) {
      p_[0] = v;
      TA_ASSERT( valid_(p_.begin(), p_.end()) , std::runtime_error,
          "Invalid permutation supplied." );
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
      TA_ASSERT( valid_(p_.begin(), p_.end()) , std::runtime_error,
          "Invalid permutation supplied." );
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
    void permute(InIter0 first_p, InIter0 last_p, InIter1 first_o, RandIter first_r) {
      TA_STATIC_ASSERT(detail::is_input_iterator<InIter0>::value);
      TA_STATIC_ASSERT(detail::is_input_iterator<InIter1>::value);
      TA_STATIC_ASSERT(detail::is_random_iterator<RandIter>::value);
      for(; first_p != last_p; ++first_p)
        (* (first_r + *first_p)) = *first_o++;
    }

    /// ForLoop defines a for loop operation for a random access iterator.
    template<typename F, typename RandIter>
    struct ForLoop {
    private:
      TA_STATIC_ASSERT(detail::is_random_iterator<RandIter>::value);
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
      TA_STATIC_ASSERT(detail::is_random_iterator<RandIter>::value);
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
        TA_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
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
      TA_STATIC_ASSERT(detail::is_random_iterator<RandIter>::value);
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
      TA_STATIC_ASSERT(detail::is_output_iterator<OutIter>::value);
      TA_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
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
      TA_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
    public:
      AssignmentOp(InIter first, InIter last) : current_(first), last_(last) { }

      void operator ()(T* it) {
        TA_ASSERT(current_ != last_, std::runtime_error,
            "The iterator is the end of the range.");
        *it = *current_++;
      }

    private:
      InIter current_;
      InIter last_;
    }; // struct AssignmentOp

    /// Function object that assigns the content of one iterator to another iterator.
    template<typename InIter, typename Alloc >
    struct UninitializedCopy {
    private:
      TA_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
    public:
      typedef typename Alloc::pointer pointer;
      typedef typename Alloc::value_type value_type;

      UninitializedCopy(InIter first, InIter last, const Alloc& a) : current_(first), last_(last), alloc_(a) { }

      void operator ()(pointer it) {
        TA_ASSERT(current_ != last_, std::runtime_error,
            "The iterator is the end of the range.");
        copy_data(it);
      }

    private:
      template <typename T>
      typename madness::enable_if<std::is_fundamental<T> >::type
      copy_data(T* p) { *p = *current_++; }

      template <typename T>
      typename madness::disable_if<std::is_fundamental<T> >::type
      copy_data(T* p) { alloc_.construct(p, *current_++); }

      InIter current_;
      InIter last_;
      const Alloc& alloc_;
    }; // struct UninitializedCopy

    /// Permutes of n-dimensional type container

    /// The copy/assignment operation is the inner most loop operation. It
    /// assigns data to the permuted array. It should assign the data from the
    /// original n-dimensional array in order to the permuted array.
    /// \tparam CS The coordinate system type associated with the object that
    /// will be permuted.
    template<typename CS, typename Op>
    struct Permute {
      /// Construct a permute function object.

      /// \param r The range object of the original object.
      /// \param op The copy/assignment operation used to copy/assign the
      /// permuted data
      Permute(const Range<CS>& r, Op op) : range_(r), op_(op) { }

      /// Perform the data of an n-dimensional container

      /// \tparam RandIter Random access iterator type for the output data
      /// \tparam InIter Input iterator type for the input data
      /// \param[in] p The permutation to be applied to the n-d array container
      /// \param[out] first_out The first iterator for the data of the output
      /// array
      /// \param[out] last_out The last iterator for the data of the output
      /// array
      /// \throw std::runtime_error When the distance between first_out and
      /// last_out, or first_in and last_in is not equal to the volume of the
      /// range object given in the constructor.
      template<typename RandIter>
      void operator ()(const Permutation<CS::dim>& p, RandIter first_out, RandIter last_out)
      {
        TA_STATIC_ASSERT(detail::is_random_iterator<RandIter>::value);
        TA_ASSERT(static_cast<typename Range<CS>::volume_type>(std::distance(first_out, last_out)) == range_.volume(),
            std::runtime_error,
            "The distance between first_out and last_out must be equal to the volume of the original container.");

        // Calculate the sizes and weights of the permuted array
        typename Range<CS>::size_array p_size;
        permute(p.begin(), p.end(), range_.size().begin(), p_size.begin());
        typename Range<CS>::size_array weight = CS::calc_weight(p_size);

        // Calculate the step sizes for the nested loops
        Permutation<CS::dim> ip = -p;
        typename Range<CS>::size_array step;
        permute(ip.begin(), ip.end(), weight.begin(), step.begin());

        // Calculate the loop end offsets.
        typename Range<CS>::size_array end;
        std::transform(range_.size().begin(), range_.size().end(), step.begin(),
            end.begin(), std::multiplies<typename Range<CS>::ordinal_index>());

        // create the nested for loop object which will permute the data
        NestedForLoop<CS::dim, Op, RandIter >
            do_loop(op_, CS::begin(end), CS::end(end), CS::begin(step), CS::end(step));
        do_loop(first_out);
      }

    private:

      /// Default construction not allowed.
      Permute();

      const Range<CS>& range_; ///< Range object for the original array
      Op op_;
    }; // struct Permute

  } // namespace detail

  /// permute a std::array
  template <unsigned int DIM, typename T>
  std::array<T,DIM> operator^(const Permutation<DIM>& perm, const std::array<T, static_cast<std::size_t>(DIM) >& orig) {
    std::array<T,DIM> result;
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
  std::array<T,DIM> operator ^=(std::array<T, static_cast<std::size_t>(DIM) >& a, const Permutation<DIM>& perm) {
    return (a = perm ^ a);
  }

} // namespace TiledArray

#endif // TILEDARRAY_PERMUTATION_H__INCLUED
