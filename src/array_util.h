#ifndef TA_ARRAY_UTIL_H__INCLUDED
#define TA_ARRAY_UTIL_H__INCLUDED

#include <boost/array.hpp>
//#include <numeric>

namespace TiledArray {
  namespace detail {

    /// Calculate the weighted dimension values.
    template<typename InIter, typename OutIter>
    void calc_weight(InIter first, InIter last, OutIter result) { // no throw
      for(typename std::iterator_traits<OutIter>::value_type weight = 1; first != last; ++first, ++result) {
        *result = weight;
        weight *= *first;
      }
    }

    /// Calculate the volume of an N-dimensional orthogonal.
    template <typename T, std::size_t DIM>
    T volume(const boost::array<T,DIM>& a) { // no throw when T is a standard type
      T result = 1;
      for(std::size_t d = 0; d < DIM; ++d)
        result *= ( a[d] < 0 ? -a[d] : a[d] );
      return result;
//      return std::accumulate(a.begin(), a.end(), T(1), std::multiplies<T>());
    }

  } // namespace detail
} // namespace TiledArray

#endif // TA_ARRAY_UTIL_H__INCLUDED
