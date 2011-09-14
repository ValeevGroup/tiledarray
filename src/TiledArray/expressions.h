#ifndef TILEDARRAY_EXPRESSIONS_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_H__INCLUDED


#include <TiledArray/binary_tensor.h>
#include <TiledArray/unary_tensor.h>
#include <TiledArray/permute_tensor.h>
#include <TiledArray/contraction_tensor.h>
#include <TiledArray/unary_tiled_tensor.h>
#include <TiledArray/type_traits.h>

namespace TiledArray {
  namespace expressions {

    template <typename LeftExp, typename RightExp>
    BinaryTensor<LeftExp, RightExp, std::plus<typename LeftExp::value_type> >
    operator+(const ReadableTensor<LeftExp>& left, const ReadableTensor<RightExp>& right) {
      return BinaryTensor<LeftExp, RightExp, std::plus<typename LeftExp::value_type> >(left.derived(),
          right.derived(), std::plus<typename LeftExp::value_type>());
    }

    template <typename RightExp>
    UnaryTensor<RightExp, std::binder1st<std::plus<typename RightExp::value_type> > >
    operator+(const typename ReadableTensor<RightExp>::value_type& left, const ReadableTensor<RightExp>& right) {
      return UnaryTensor<RightExp, std::binder1st<std::plus<typename RightExp::value_type> > >(right.derived(),
          std::bind1st(std::plus<typename RightExp::value_type>(), left));
    }

    template <typename LeftExp>
    UnaryTensor<LeftExp, std::binder2nd<std::plus<typename LeftExp::value_type> > >
    operator+(const ReadableTensor<LeftExp>& left, const typename ReadableTensor<LeftExp>::value_type& right) {
      return UnaryTensor<LeftExp, std::binder2nd<std::plus<typename LeftExp::value_type> > >(left.derived(),
          std::bind2nd(std::plus<typename LeftExp::value_type>(), right));
    }

    template <typename LeftExp, typename RightExp>
    BinaryTensor<LeftExp, RightExp, std::minus<typename LeftExp::value_type> >
    operator-(const ReadableTensor<LeftExp>& left, const ReadableTensor<RightExp>& right) {
      return BinaryTensor<LeftExp, RightExp, std::minus<typename LeftExp::value_type> >(left.derived(),
          right.derived(), std::minus<typename LeftExp::value_type>());
    }

    template <typename RightExp>
    UnaryTensor<RightExp, std::binder1st<std::minus<typename RightExp::value_type> > >
    operator-(const typename ReadableTensor<RightExp>::value_type& left, const ReadableTensor<RightExp>& right) {
      return UnaryTensor<RightExp, std::binder1st<std::minus<typename RightExp::value_type> > >(right.derived(),
          std::bind1st(std::minus<typename RightExp::value_type>(), left));
    }

    template <typename LeftExp>
    UnaryTensor<LeftExp, std::binder2nd<std::minus<typename LeftExp::value_type> > >
    operator-(const ReadableTensor<LeftExp>& left, const typename ReadableTensor<LeftExp>::value_type& right) {
      return UnaryTensor<LeftExp, std::binder2nd<std::minus<typename LeftExp::value_type> > >(left.derived(),
          std::bind2nd(std::minus<typename LeftExp::value_type>(), right));
    }

    template <typename RightExp>
    UnaryTensor<RightExp, std::binder1st<std::multiplies<typename RightExp::value_type> > >
    operator*(const typename ReadableTensor<RightExp>::value_type& left, const ReadableTensor<RightExp>& right) {
      return UnaryTensor<RightExp, std::binder1st<std::multiplies<typename RightExp::value_type> > >(right.derived(),
          std::bind1st(std::multiplies<typename RightExp::value_type>(), left));
    }

    template <typename LeftExp>
    UnaryTensor<LeftExp, std::binder2nd<std::multiplies<typename LeftExp::value_type> > >
    operator*(const ReadableTensor<LeftExp>& left, const typename ReadableTensor<LeftExp>::value_type& right) {
      return UnaryTensor<LeftExp, std::binder2nd<std::multiplies<typename LeftExp::value_type> > >(left.derived(),
          std::bind2nd(std::multiplies<typename LeftExp::value_type>(), right));
    }

    template <typename ArgExp>
    UnaryTensor<ArgExp, std::negate<typename ArgExp::value_type> >
    operator-(const ReadableTensor<ArgExp>& arg) {
      return UnaryTensor<ArgExp, std::negate<typename ArgExp::value_type> >(arg.derived(),
          std::negate<typename ArgExp::value_type>());
    }

    template <unsigned int DIM, typename ArgExp>
    PermuteTensor<ArgExp, DIM>
    operator^(const Permutation<DIM>& p, const ReadableTensor<ArgExp>& arg) {
      return PermuteTensor<ArgExp, DIM>(arg.derived(), p);
    }

  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_H__INCLUDED
