#ifndef TA_ARRAY_UTIL_H__INCLUDED
#define TA_ARRAY_UTIL_H__INCLUDED

#include <type_traits.h>
#include <boost/array.hpp>
#include <iosfwd>
#include <numeric>

namespace TiledArray {
  namespace detail {

    /// Calculate the weighted dimension values.
    template<typename InIter, typename OutIter>
    void calc_weight(InIter first, InIter last, OutIter result) { // no throw
      for(typename std::iterator_traits<OutIter>::value_type weight = 1; first != last; ++first, ++result) {
        BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
        BOOST_STATIC_ASSERT(detail::is_output_iterator<OutIter>::value);
        *result = weight;
        weight *= *first;
      }
    }

    /// Calculate the index of an ordinal.
    /// \arg \c o is the ordinal value.
    /// \arg \c [first, last) is the iterator range that contains the array
    /// weights from most significant to least significant.
    /// \arg \c result is an iterator to the index, which points to the most
    /// significant element.
    template<typename Ord, typename InIter, typename OutIter>
    void calc_index(Ord o, InIter first, InIter last, OutIter result) {
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      BOOST_STATIC_ASSERT(detail::is_output_iterator<OutIter>::value);
      for(; first != last; ++first, ++result) {
        *result = o % *first;
        o -= *result * *first;
      }
    }

    /// Calculate the volume of an N-dimensional orthogonal.
    template <typename T, std::size_t DIM>
    T volume(const boost::array<T,DIM>& a) { // no throw when T is a standard type
      return std::accumulate(a.begin(), a.end(), T(1), std::multiplies<T>());
    }

    template <typename InIter>
    void print_array(std::ostream& output, InIter first, InIter last) {
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      if(first != last)
        output << *first++;
      for(; first != last; ++first)
        output << ", " << *first;
    }
  } // namespace detail
} // namespace TiledArray

#endif // TA_ARRAY_UTIL_H__INCLUDED
