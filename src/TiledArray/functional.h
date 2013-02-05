#ifndef TILEDARRAY_FUNCTIONAL_H__INCLUDED
#define TILEDARRAY_FUNCTIONAL_H__INCLUDED

namespace TiledArray {
  namespace detail {

    /// Square function object

    /// \tparam T argument and result type
    template <typename T>
    struct Square {
      typedef T result_type;
      typedef T argument_type;

      /// Square \c t

      /// \param t The value to be squared
      /// \return t * t
      result_type operator()(argument_type t) const { return t * t; }

    }; // class Square

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_FUNCTIONAL_H__INCLUDED
