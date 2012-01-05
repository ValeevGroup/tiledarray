#ifndef TILEDARRAY_EXPRESSIONS_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_H__INCLUDED


#include <TiledArray/binary_tensor.h>
#include <TiledArray/unary_tensor.h>
#include <TiledArray/permute_tensor.h>
#include <TiledArray/contraction_tensor.h>
#include <TiledArray/unary_tiled_tensor.h>
#include <TiledArray/binary_tiled_tensor.h>
#include <TiledArray/contraction_tiled_tensor.h>
#include <TiledArray/functional.h>

namespace TiledArray {
  namespace expressions {

    template <typename LeftExp, typename RightExp>
    BinaryTensor<LeftExp, RightExp, std::plus<typename LeftExp::value_type> >
    operator+(const ReadableTensor<LeftExp>& left, const ReadableTensor<RightExp>& right) {
      return make_binary_tensor(left, right, std::plus<typename LeftExp::value_type>());
    }

    template <typename RightExp>
    UnaryTensor<RightExp, std::binder1st<std::plus<typename RightExp::value_type> > >
    operator+(const typename ReadableTensor<RightExp>::value_type& left, const ReadableTensor<RightExp>& right) {
      return make_unary_tensor(right, std::bind1st(std::plus<typename RightExp::value_type>(), left));
    }

    template <typename LeftExp>
    UnaryTensor<LeftExp, std::binder2nd<std::plus<typename LeftExp::value_type> > >
    operator+(const ReadableTensor<LeftExp>& left, const typename ReadableTensor<LeftExp>::value_type& right) {
      return make_unary_tensor(left, std::bind2nd(std::plus<typename LeftExp::value_type>(), right));
    }

    template <typename LeftExp, typename RightExp>
    BinaryTensor<LeftExp, RightExp, std::minus<typename LeftExp::value_type> >
    operator-(const ReadableTensor<LeftExp>& left, const ReadableTensor<RightExp>& right) {
      return make_binary_tensor(left, right, std::minus<typename LeftExp::value_type>());
    }

    template <typename RightExp>
    UnaryTensor<RightExp, std::binder1st<std::minus<typename RightExp::value_type> > >
    operator-(const typename ReadableTensor<RightExp>::value_type& left, const ReadableTensor<RightExp>& right) {
      return make_unary_tensor(right, std::bind1st(std::minus<typename RightExp::value_type>(), left));
    }

    template <typename LeftExp>
    UnaryTensor<LeftExp, std::binder2nd<std::minus<typename LeftExp::value_type> > >
    operator-(const ReadableTensor<LeftExp>& left, const typename ReadableTensor<LeftExp>::value_type& right) {
      return make_unary_tensor(left, std::bind2nd(std::minus<typename LeftExp::value_type>(), right));
    }

    template <typename RightExp>
    UnaryTensor<RightExp, std::binder1st<std::multiplies<typename RightExp::value_type> > >
    operator*(const typename ReadableTensor<RightExp>::value_type& left, const ReadableTensor<RightExp>& right) {
      return make_unary_tensor(right, std::bind1st(std::multiplies<typename RightExp::value_type>(), left));
    }

    template <typename LeftExp>
    UnaryTensor<LeftExp, std::binder2nd<std::multiplies<typename LeftExp::value_type> > >
    operator*(const ReadableTensor<LeftExp>& left, const typename ReadableTensor<LeftExp>::value_type& right) {
      return make_unary_tensor(left, std::bind2nd(std::multiplies<typename LeftExp::value_type>(), right));
    }

    template <typename ArgExp>
    UnaryTensor<ArgExp, std::negate<typename ArgExp::value_type> >
    operator-(const ReadableTensor<ArgExp>& arg) {
      return make_unary_tensor(arg, std::negate<typename ArgExp::value_type>());
    }

    template <typename ArgExp>
    PermuteTensor<ArgExp>
    operator^(const Permutation& p, const ReadableTensor<ArgExp>& arg) {
      return make_permute_tensor(arg, p);
    }

    template <typename ArgExp>
    const ArgExp& operator^(const TiledArray::detail::NoPermutation& p, const ReadableTensor<ArgExp>& arg) {
      return make_permute_tensor(arg, p);
    }


    // Tiled Tensor expression factory functions

    template <typename LeftExp, typename RightExp>
    BinaryTiledTensor<LeftExp, RightExp, std::plus<typename LeftExp::value_type::value_type> >
    operator+(const ReadableTiledTensor<LeftExp>& left, const ReadableTiledTensor<RightExp>& right) {
      return make_binary_tiled_tensor(left, right, std::plus<typename LeftExp::value_type::value_type>());
    }

    template <typename RightExp>
    UnaryTiledTensor<RightExp, TiledArray::detail::Binder1st<std::plus<typename RightExp::value_type::value_type> > >
    operator+(const typename ReadableTiledTensor<RightExp>::value_type& left, const ReadableTiledTensor<RightExp>& right) {
      return make_unary_tiled_tensor(right,
          TiledArray::detail::bind1st(std::plus<typename RightExp::value_type::value_type>(), left));
    }

    template <typename LeftExp>
    UnaryTiledTensor<LeftExp, TiledArray::detail::Binder2nd<std::plus<typename LeftExp::value_type::value_type> > >
    operator+(const ReadableTiledTensor<LeftExp>& left, const typename ReadableTiledTensor<LeftExp>::value_type& right) {
      return make_unary_tiled_tensor(left,
          TiledArray::detail::bind2nd(std::plus<typename LeftExp::value_type::value_type>(), right));
    }

    template <typename LeftExp, typename RightExp>
    BinaryTiledTensor<LeftExp, RightExp, std::minus<typename LeftExp::value_type::value_type> >
    operator-(const ReadableTiledTensor<LeftExp>& left, const ReadableTiledTensor<RightExp>& right) {
      return make_binary_tiled_tensor(left, right, std::minus<typename LeftExp::value_type::value_type>());
    }

    template <typename RightExp>
    UnaryTiledTensor<RightExp, TiledArray::detail::Binder1st<std::minus<typename RightExp::value_type::value_type> > >
    operator-(const typename ReadableTiledTensor<RightExp>::value_type& left, const ReadableTiledTensor<RightExp>& right) {
      return make_unary_tiled_tensor(right,
          TiledArray::detail::bind1st(std::minus<typename RightExp::value_type::value_type>(), left));
    }

    template <typename LeftExp>
    UnaryTiledTensor<LeftExp, TiledArray::detail::Binder2nd<std::minus<typename LeftExp::value_type::value_type> > >
    operator-(const ReadableTiledTensor<LeftExp>& left, const typename ReadableTiledTensor<LeftExp>::value_type& right) {
      return make_unary_tiled_tensor(left,
          TiledArray::detail::bind2nd(std::minus<typename LeftExp::value_type::value_type>(), right));
    }

    template <typename LeftExp, typename RightExp>
    ContractionTiledTensor<LeftExp, RightExp>
    operator*(const ReadableTiledTensor<LeftExp>& left, const ReadableTiledTensor<RightExp>& right) {
      return make_contraction_tiled_tensor(left, right);
    }

    template <typename RightExp>
    UnaryTiledTensor<RightExp, TiledArray::detail::Binder1st<std::multiplies<typename RightExp::value_type::value_type> > >
    operator*(const typename ReadableTiledTensor<RightExp>::value_type::value_type& left, const ReadableTiledTensor<RightExp>& right) {
      return make_unary_tiled_tensor(right,
          TiledArray::detail::bind1st(std::multiplies<typename RightExp::value_type::value_type>(), left));
    }

    template <typename LeftExp>
    UnaryTiledTensor<LeftExp, TiledArray::detail::Binder2nd<std::multiplies<typename LeftExp::value_type::value_type> > >
    operator*(const ReadableTiledTensor<LeftExp>& left, const typename ReadableTiledTensor<LeftExp>::value_type& right) {
      return make_unary_tiled_tensor(left,
          TiledArray::detail::bind2nd(std::multiplies<typename LeftExp::value_type::value_type>(), right));
    }

    template <typename ArgExp>
    UnaryTiledTensor<ArgExp, std::negate<typename ArgExp::value_type::value_type> >
    operator-(const ReadableTiledTensor<ArgExp>& arg) {
      return make_unary_tiled_tensor(arg, std::negate<typename ArgExp::value_type::value_type>());
    }

  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_H__INCLUDED
