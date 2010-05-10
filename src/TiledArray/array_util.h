#ifndef TA_ARRAY_UTIL_H__INCLUDED
#define TA_ARRAY_UTIL_H__INCLUDED

#include <TiledArray/type_traits.h>
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

    /// Calculate the ordinal index of an array.

    /// \var \c [index_first, \c index_last) is a pair of iterators to the coordinate index.
    /// \var \c weight_first is an iterator to the array weights.
    template<typename IndexInIter, typename WeightInIter>
    typename std::iterator_traits<IndexInIter>::value_type
    calc_ordinal(IndexInIter index_first, IndexInIter index_last, WeightInIter weight_first) {
      BOOST_STATIC_ASSERT(detail::is_input_iterator<IndexInIter>::value);
      BOOST_STATIC_ASSERT(detail::is_input_iterator<WeightInIter>::value);
      return std::inner_product(index_first, index_last, weight_first,
          typename std::iterator_traits<IndexInIter>::value_type(1));
    }

    /// Calculate the ordinal index of an array.

    /// \var \c [index_first, \c index_last) is a pair of iterators to the coordinate index.
    /// \var \c weight_first is an iterator to the array weights.
    template<typename IndexInIter, typename WeightInIter, typename StartInIter>
    typename std::iterator_traits<IndexInIter>::value_type
    calc_ordinal(IndexInIter index_first, IndexInIter index_last, WeightInIter weight_first, StartInIter start_first) {
      BOOST_STATIC_ASSERT(detail::is_input_iterator<IndexInIter>::value);
      BOOST_STATIC_ASSERT(detail::is_input_iterator<WeightInIter>::value);
      BOOST_STATIC_ASSERT(detail::is_input_iterator<WeightInIter>::value);

      typename std::iterator_traits<IndexInIter>::value_type o = 1;
      for(; index_first != index_last; ++index_first, ++weight_first, ++start_first)
        o *= (*index_first - *start_first) * *weight_first;

      return o;
    }

    /// Calculate the volume of an N-dimensional orthogonal.
    template <typename InIter>
    typename std::iterator_traits<InIter>::value_type
    volume(InIter first, InIter last) { // no throw when T is a standard type
      BOOST_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      typedef typename std::iterator_traits<InIter>::value_type value_type;
      return std::accumulate(first, last, value_type(1), std::multiplies<value_type>());
    }

    /// Calculate the volume of an N-dimensional orthogonal.
    template <typename T, std::size_t DIM>
    T volume(const boost::array<T,DIM>& a) { // no throw when T is a standard type
      return volume(a.begin(), a.end());
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
