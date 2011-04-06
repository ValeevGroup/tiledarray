#ifndef TA_ARRAY_UTIL_H__INCLUDED
#define TA_ARRAY_UTIL_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/type_traits.h>
#include <boost/array.hpp>
#include <iosfwd>
#include <numeric>

namespace TiledArray {
  namespace detail {

    template<typename InIterLeft, typename InIterRight, typename OutIterLeft, typename OutIterRight>
    void pack_size(InIterLeft lfirst, InIterLeft llast, InIterRight rfirst, InIterRight rlast, OutIterLeft lresult, OutIterRight rresult) {

    }
    template <typename InIter>
    void print_array(std::ostream& output, InIter first, InIter last) {
      TA_STATIC_ASSERT(detail::is_input_iterator<InIter>::value);
      if(first != last)
        output << *first++;
      for(; first != last; ++first)
        output << ", " << *first;
    }
  } // namespace detail
} // namespace TiledArray

#endif // TA_ARRAY_UTIL_H__INCLUDED
