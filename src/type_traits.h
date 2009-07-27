#ifndef TILEDARRAY_TYPE_TRAITS_H__INCLUDED
#define TILEDARRAY_TYPE_TRAITS_H__INCLUDED

namespace TiledArray {
  namespace detail {

    template <typename T, typename V>
    struct mirror_const {
      typedef T type;
      typedef V value;
      typedef V& reference;
      typedef V* pointer;
    };

    template <typename T, typename V>
    struct mirror_const<const T, V> {
      typedef const T type;
      typedef const V value;
      typedef const V& reference;
      typedef const V* pointer;
    };

  } // namespace detail
} // namespace TiledArray
#endif // TILEDARRAY_TYPE_TRAITS_H__INCLUDED
