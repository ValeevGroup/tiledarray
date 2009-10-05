#ifndef TILEDARRAY_TYPE_TRAITS_H__INCLUDED
#define TILEDARRAY_TYPE_TRAITS_H__INCLUDED

#include <boost/type_traits.hpp>

namespace TiledArray {
  namespace detail {

    template <bool B, typename T>
    struct add_const {
      typedef const typename boost::remove_const<T>::type type;
    };

    template <typename T>
    struct add_const<false, T> {
      typedef typename boost::remove_const<T>::type type;
    };

  } // namespace detail
} // namespace TiledArray
#endif // TILEDARRAY_TYPE_TRAITS_H__INCLUDED
