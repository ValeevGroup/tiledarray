#ifndef TA_ARRAY_VIEW_H__INCLUDED
#define TA_ARRAY_VIEW_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/permutation.h>
#include <boost/array.hpp>
#include <boost/type_traits.hpp>

namespace TiledArray {
  namespace detail {

    // Forward declarations
    template<typename T>
    class ArrayRef;
    template<typename T>
    void swap(ArrayRef<T>& , ArrayRef<T>& );
    template<typename T, typename U>
    bool operator==(const ArrayRef<T>& , const ArrayRef<U>& );
    template<typename T, typename U>
    bool operator!=(const ArrayRef<T>& , const ArrayRef<U>& );
    template<typename T, typename U>
    bool operator<(const ArrayRef<T>& , const ArrayRef<U>& );
    template<typename T, typename U>
    bool operator>(const ArrayRef<T>&, const ArrayRef<U>&);
    template<typename T, typename U>
    bool operator<=(const ArrayRef<T>&, const ArrayRef<U>&);
    template<typename T, typename U>
    bool operator>=(const ArrayRef<T>&, const ArrayRef<U>&);
    template<typename T, typename U, std::size_t N>
    bool operator==(const ArrayRef<T>&, const boost::array<U, N>&);
    template<typename T, typename U, std::size_t N>
    bool operator!=(const ArrayRef<T>&, const boost::array<U, N>&);
    template<typename T, typename U, std::size_t N>
    bool operator<(const ArrayRef<T>&, const boost::array<U, N>&);
    template<typename T, typename U, std::size_t N>
    bool operator>(const ArrayRef<T>&, const boost::array<U, N>&);
    template<typename T, typename U, std::size_t N>
    bool operator<=(const ArrayRef<T>&, const boost::array<U, N>&);
    template<typename T, typename U, std::size_t N>
    bool operator>=(const ArrayRef<T>&, const boost::array<U, N>&);
    template<typename T, typename U, std::size_t N>
    bool operator==(const boost::array<T, N>&, const ArrayRef<U>&);
    template<typename T, typename U, std::size_t N>
    bool operator!=(const boost::array<T, N>&, const ArrayRef<U>&);
    template<typename T, typename U, std::size_t N>
    bool operator<(const boost::array<T, N>&, const ArrayRef<U>&);
    template<typename T, typename U, std::size_t N>
    bool operator>(const boost::array<T, N>&, const ArrayRef<U>&);
    template<typename T, typename U, std::size_t N>
    bool operator<=(const boost::array<T, N>&, const ArrayRef<U>&);
    template<typename T, typename U, std::size_t N>
    bool operator>=(const boost::array<T, N>&, const ArrayRef<U>&);
    template <unsigned int DIM, typename T>
    ArrayRef<T>& operator^=(ArrayRef<T>&, const Permutation<DIM>&);
    template <unsigned int DIM, typename T>
    boost::array<typename boost::remove_const<T>::type, static_cast<std::size_t>(DIM) >
    operator^(const Permutation<DIM>&, const ArrayRef<T>&);

    /// Stores a reference to an array.

    /// ArrayRef is a reference to an array. It does not allocate any memory of
    /// its own. The view is only valid as long as the object it points to is
    /// valid. It is specifically designed to duplicate the interface of
    /// boost::array (with the exception of the static member variable static_size
    /// and data members are private). In addition to the boost::array like
    /// interface, additional constructors are provided constructors: default,
    /// copy, from a pair of pointers, from a boost::arrays, and a pair. It also
    /// provides a convention operator to boost::array types.
    template<typename T>
    class ArrayRef {
    public:
      // types
      typedef ArrayRef<T>                           ArrayRef_;
      typedef typename boost::remove_const<T>::type value_type;
      typedef T*                                    iterator;
      typedef const T*                              const_iterator;
      typedef std::reverse_iterator<iterator>       reverse_iterator;
      typedef std::reverse_iterator<const_iterator> const_reverse_iterator;
      typedef T&                                    reference;
      typedef const T&                              const_reference;
      typedef std::size_t                           size_type;
      typedef std::ptrdiff_t                        difference_type;

      /// Default Constructor
      ArrayRef() : first(NULL), last(NULL) { }
      /// Copy Constuctor
      ArrayRef(const ArrayRef_& other) : first(other.first), last(other.last) { }
      /// Construct a view of a boost array.
      template<typename U, std::size_t N>
      ArrayRef(boost::array<U,N>& a) : first(a.c_array()), last(a.c_array() + N) { }
      template<typename U, std::size_t N>
      ArrayRef(const boost::array<U,N>& a) : first(a.data()), last(a.data() + N) { }

      /// Construct a view from a pair that contains two pointers.

      /// p.first should be a pointer to the beginning of the array range and
      /// p.second should be a pointer to the end of the array range. If
      /// exceptions are enabled, this constructor will check that
      /// p.first <= p.second.
      template<typename U>
      ArrayRef(const std::pair<U*, U*>& p) : first(p.first), last(p.second) {
        TA_ASSERT(p.first <= p.second, std::range_error,
            "The first pointer is after the last pointer in memory.");
      }

      /// Construct a view from a pair of pointers.

      /// f should be a pointer to the beginning of the array range and l should
      /// be a pointer to the end of the array range.
      ArrayRef(T* f, T* l) : first(f), last(l) {
        TA_ASSERT(f <= l, std::range_error,
            "The first pointer is after the last pointer in memory.");
      }

      /// Copy assignment operator

      /// This will change array reference so that it references the same memory
      /// as the original array reference. No memory is copied.
      template<typename U>
      ArrayRef& operator=(const ArrayRef<U>& other) {
        first = other.first;
        last = other.last;
      }

      /// Assignment operator for a boost array.

      /// This assignment will change the array reference so it references the
      /// given boost array. No memory is copied.
      template<typename U, std::size_t N>
      ArrayRef& operator=(const boost::array<U,N>& a) {
        first = a.c_array();
        last = a.c_array() + N;
      }

      /// Assignment operator for a boost

      /// This assignment will change the array reference so it references the
      /// given given memory range in the pair. If exceptions are enabled, a
      /// std::runtime_error is thrown when p.first <= p.second.
      template<typename U>
      ArrayRef& operator=(const std::pair<U*, U*>& p) {
        TA_ASSERT(p.first <= p.second, std::range_error,
            "The first pointer is after the last pointer in memory.");
        first = p.first;
        last = p.second;

        return *this;
      }

      /// Type conversion to boost array.
      template<std::size_t N>
      operator boost::array<T,N>() const {
        TA_ASSERT(size() == N, std::runtime_error,
            "The number of elements in the boost array is not equal to the size of the array view.");
        boost::array<T,N> result;
        std::copy(first, last, result.begin());
        return result;
      }

      // iterator support
      iterator begin() { return first; }
      const_iterator begin() const { return first; }
      iterator end() { return last; }
      const_iterator end() const { return last; }

      // reverse iterator support
      reverse_iterator rbegin() { return reverse_iterator(last); }
      const_reverse_iterator rbegin() const { return const_reverse_iterator(last); }
      reverse_iterator rend() { return reverse_iterator(first); }
      const_reverse_iterator rend() const { return const_reverse_iterator(first); }

      // capacity
      size_type size() const { return std::distance(first, last); }
      bool empty() const { return first == last; }
      size_type max_size() const { return std::distance(first, last); }

      // element access
      reference operator[](size_type n) { return first[n]; }
      const_reference operator[](size_type n) const { return first[n]; }
      reference at(size_type n) {
        if(n > std::distance(first, last))
          TA_EXCEPTION(std::out_of_range, "Element n is out of range.");
        return first[n];
      }

      const_reference at(size_type n) const {
        if(n > std::distance(first, last))
          TA_EXCEPTION(std::out_of_range, "Element n is out of range.");
        return first[n];
      }

      reference front() { return *first; }
      const_reference front() const { return *first; }
      reference back() { return *(last - 1); }
      const_reference back() const { return *(last - 1); }
      const T* data() const { return first; }
      T* c_array() { return first; }

      // modifiers
      void swap(ArrayRef_& other) {
        std::swap(first, other.first);
        std::swap(last, other.last);
      }

      void assign(const T& val) {
        std::fill(first, last, val);
      }

    private:
      T* first;
      T* last;
    }; // class ArrayRef

    /// Array reference swap
    template<typename T>
    void swap(ArrayRef<T>& a1, ArrayRef<T>& a2) {
      a1.swap(a2);
    }

    /// Array reference equality comparison

    /// Returns true if all elements of the two arrays are equal.
    template<typename T, typename U>
    bool operator==(const ArrayRef<T>& a1, const ArrayRef<U>& a2) {
      return a1.size() == a2.size() &&
          std::equal(a1.begin(), a1.end(), a2.begin());
    }

    /// Array reference inequality comparison

    /// Returns true if any one of the elements is not equal.
    template<typename T, typename U>
    bool operator!=(const ArrayRef<T>& a1, const ArrayRef<U>& a2) {
      return !(a1 == a2);
    }

    /// Array reference less-than comparison

    /// This does a lexicographical less-than comparison between the two arrays.
    template<typename T, typename U>
    bool operator<(const ArrayRef<T>& a1, const ArrayRef<U>& a2) {
      return std::lexicographical_compare(a1.begin(), a1.end(), a2.begin(), a2.end());
    }

    /// Array reference greater-than comparison

    /// This does a lexicographical greater-than comparison between the two
    /// arrays.
    template<typename T, typename U>
    bool operator>(const ArrayRef<T>& a1, const ArrayRef<U>& a2) {
      return a2 < a1;
    }

    /// Array reference less-than-or-equal-to comparison

    /// This does a lexicographical less-than-or-equal-to comparison between the
    /// two arrays.
    template<typename T, typename U>
    bool operator<=(const ArrayRef<T>& a1, const ArrayRef<U>& a2) {
      return !(a2 < a1);
    }

    /// Array reference greater-than-or-equal-to comparison

    /// This does a lexicographical greater-than-or-equal-to comparison between
    /// the two arrays.
    template<typename T, typename U>
    bool operator>=(const ArrayRef<T>& a1, const ArrayRef<U>& a2) {
      return !(a1 < a2);
    }

    /// Array reference equality comparison

    /// Returns true if all elements of the two arrays are equal.
    template<typename T, typename U, std::size_t N>
    bool operator==(const ArrayRef<T>& a1, const boost::array<U, N>& a2) {
      return a1.size() == a2.size() &&
          std::equal(a1.begin(), a1.end(), a2.begin());
    }

    /// Array reference inequality comparison

    /// Returns true if any one of the elements is not equal.
    template<typename T, typename U, std::size_t N>
    bool operator!=(const ArrayRef<T>& a1, const boost::array<U, N>& a2) {
      return !(a1 == a2);
    }

    /// Array reference less-than comparison

    /// This does a lexicographical less-than comparison between the two arrays.
    template<typename T, typename U, std::size_t N>
    bool operator<(const ArrayRef<T>& a1, const boost::array<U, N>& a2) {
      return std::lexicographical_compare(a1.begin(), a1.end(), a2.begin(), a2.end());
    }

    /// Array reference greater-than comparison

    /// This does a lexicographical greater-than comparison between the two
    /// arrays.
    template<typename T, typename U, std::size_t N>
    bool operator>(const ArrayRef<T>& a1, const boost::array<U, N>& a2) {
      return a2 < a1;
    }

    /// Array reference less-than-or-equal-to comparison

    /// This does a lexicographical less-than-or-equal-to comparison between the
    /// two arrays.
    template<typename T, typename U, std::size_t N>
    bool operator<=(const ArrayRef<T>& a1, const boost::array<U, N>& a2) {
      return !(a2 < a1);
    }

    /// Array reference greater-than-or-equal-to comparison

    /// This does a lexicographical greater-than-or-equal-to comparison between
    /// the two arrays.
    template<typename T, typename U, std::size_t N>
    bool operator>=(const ArrayRef<T>& a1, const boost::array<U, N>& a2) {
      return !(a1 < a2);
    }

    /// Array reference equality comparison

    /// Returns true if all elements of the two arrays are equal.
    template<typename T, typename U, std::size_t N>
    bool operator==(const boost::array<T, N>& a1, const ArrayRef<U>& a2) {
      return a1.size() == a2.size() &&
          std::equal(a1.begin(), a1.end(), a2.begin());
    }

    /// Array reference inequality comparison

    /// Returns true if any one of the elements is not equal.
    template<typename T, typename U, std::size_t N>
    bool operator!=(const boost::array<T, N>& a1, const ArrayRef<U>& a2) {
      return !(a1 == a2);
    }

    /// Array reference less-than comparison

    /// This does a lexicographical less-than comparison between the two arrays.
    template<typename T, typename U, std::size_t N>
    bool operator<(const boost::array<T, N>& a1, const ArrayRef<U>& a2) {
      return std::lexicographical_compare(a1.begin(), a1.end(), a2.begin(), a2.end());
    }

    /// Array reference greater-than comparison

    /// This does a lexicographical greater-than comparison between the two
    /// arrays.
    template<typename T, typename U, std::size_t N>
    bool operator>(const boost::array<T, N>& a1, const ArrayRef<U>& a2) {
      return a2 < a1;
    }

    /// Array reference less-than-or-equal-to comparison

    /// This does a lexicographical less-than-or-equal-to comparison between the
    /// two arrays.
    template<typename T, typename U, std::size_t N>
    bool operator<=(const boost::array<T, N>& a1, const ArrayRef<U>& a2) {
      return !(a2 < a1);
    }

    /// Array reference greater-than-or-equal-to comparison

    /// This does a lexicographical greater-than-or-equal-to comparison between
    /// the two arrays.
    template<typename T, typename U, std::size_t N>
    bool operator>=(const boost::array<T, N>& a1, const ArrayRef<U>& a2) {
      return !(a1 < a2);
    }

    /// permute an array reference
    template <unsigned int DIM, typename T>
    ArrayRef<T>& operator^=(ArrayRef<T>& a, const Permutation<DIM>& perm) {
      TA_ASSERT(a.size() == DIM, std::runtime_error,
          "Dimensions of the array must match that of the permutation.");
      T temp[DIM];
      detail::permute(perm.begin(), perm.end(), a.begin(), static_cast<T*>(temp));
      std::copy(temp, temp + DIM, a.begin());
      return a;
    }

    /// permute an array reference
    template <unsigned int DIM, typename T>
    boost::array<typename boost::remove_const<T>::type, static_cast<std::size_t>(DIM) >
    operator^(const Permutation<DIM>& perm, const ArrayRef<T>& a) {
      TA_ASSERT(a.size() == DIM, std::runtime_error,
          "Dimensions of the array must match that of the permutation.");
      boost::array<typename boost::remove_const<T>::type, static_cast<std::size_t>(DIM) > result;
      detail::permute(perm.begin(), perm.end(), a.begin(), result.begin());
      return result;
    }
  } // namespace detail
} // namespace TiledArray

#endif // TA_ARRAY_VIEW_H__INCLUDED
