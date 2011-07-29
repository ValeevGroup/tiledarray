#ifndef TILEDARRAY_EXPRESSIONS_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_H__INCLUDED


#include <TiledArray/binary_tensor.h>
#include <TiledArray/unary_tensor.h>
#include <TiledArray/permute_tensor.h>
#include <TiledArray/contraction_tensor.h>
#include <TiledArray/type_traits.h>

namespace TiledArray {
  namespace expressions {

    template <typename LeftExp, typename RightExp>
    typename madness::enable_if_c<!(std::is_scalar<LeftExp>::value || std::is_scalar<RightExp>::value),
        BinaryTensor<LeftExp, RightExp, std::plus<typename LeftExp::value_type> > >::type
    operator+(const LeftExp& left, const RightExp& right) {
      return BinaryTensor<LeftExp, RightExp, std::plus<typename LeftExp::value_type> >(left,
          right, std::plus<typename LeftExp::value_type>());
    }

    template <typename LeftExp, typename RightExp>
    typename madness::enable_if_c<std::is_scalar<LeftExp>::value && (! std::is_scalar<RightExp>::value),
        UnaryTensor<RightExp, std::binder1st<std::plus<typename RightExp::value_type> > > >::type
    operator+(const LeftExp& left, const RightExp& right) {
      return UnaryTensor<RightExp, std::binder1st<std::plus<typename RightExp::value_type> > >(right,
          std::bind1st(std::plus<typename RightExp::value_type>(), left));
    }

    template <typename LeftExp, typename RightExp>
    typename madness::enable_if_c<(! std::is_scalar<LeftExp>::value) && std::is_scalar<RightExp>::value,
        UnaryTensor<LeftExp, std::binder2nd<std::plus<typename LeftExp::value_type> > > >::type
    operator+(const LeftExp& left, const RightExp& right) {
      return UnaryTensor<LeftExp, std::binder2nd<std::plus<typename LeftExp::value_type> > >(left,
          std::bind2nd(std::plus<typename LeftExp::value_type>(), right));
    }

    template <typename LeftExp, typename RightExp>
    typename madness::enable_if_c<!(std::is_scalar<LeftExp>::value || std::is_scalar<RightExp>::value),
        BinaryTensor<LeftExp, RightExp, std::minus<typename LeftExp::value_type> > >::type
    operator-(const LeftExp& left, const RightExp& right) {
      return BinaryTensor<LeftExp, RightExp, std::minus<typename LeftExp::value_type> >(left,
          right, std::minus<typename LeftExp::value_type>());
    }

    template <typename LeftExp, typename RightExp>
    typename madness::enable_if_c<std::is_scalar<LeftExp>::value && (! std::is_scalar<RightExp>::value),
        UnaryTensor<RightExp, std::binder1st<std::minus<typename RightExp::value_type> > > >::type
    operator-(const LeftExp& left, const RightExp& right) {
      return UnaryTensor<RightExp, std::binder1st<std::minus<typename RightExp::value_type> > >(right,
          std::bind1st(std::minus<typename RightExp::value_type>(), left));
    }

    template <typename LeftExp, typename RightExp>
    typename madness::enable_if_c<(! std::is_scalar<LeftExp>::value) && std::is_scalar<RightExp>::value,
        UnaryTensor<LeftExp, std::binder2nd<std::minus<typename LeftExp::value_type> > > >::type
    operator-(const LeftExp& left, const RightExp& right) {
      return UnaryTensor<LeftExp, std::binder2nd<std::minus<typename LeftExp::value_type> > >(left,
          std::bind2nd(std::minus<typename LeftExp::value_type>(), right));
    }

    template <typename LeftExp, typename RightExp>
    typename madness::enable_if_c<std::is_scalar<LeftExp>::value && (! std::is_scalar<RightExp>::value),
        UnaryTensor<RightExp, std::binder1st<std::multiplies<typename RightExp::value_type> > > >::type
    operator*(const LeftExp& left, const RightExp& right) {
      return UnaryTensor<RightExp, std::binder1st<std::multiplies<typename RightExp::value_type> > >(right,
          std::bind1st(std::multiplies<typename RightExp::value_type>(), left));
    }

    template <typename LeftExp, typename RightExp>
    typename madness::enable_if_c<(! std::is_scalar<LeftExp>::value) && std::is_scalar<RightExp>::value,
        UnaryTensor<LeftExp, std::binder2nd<std::multiplies<typename LeftExp::value_type> > > >::type
    operator*(const LeftExp& left, const RightExp& right) {
      return UnaryTensor<LeftExp, std::binder2nd<std::multiplies<typename LeftExp::value_type> > >(left,
          std::bind2nd(std::multiplies<typename LeftExp::value_type>(), right));
    }

    template <typename ArgExp>
    UnaryTensor<ArgExp, std::negate<typename ArgExp::value_type> >
    operator-(const ArgExp& arg) {
      return UnaryTensor<ArgExp, std::negate<typename ArgExp::value_type> >(arg, std::negate<typename ArgExp::value_type>());
    }

    template <unsigned int DIM, typename ArgExp>
    PermuteTensor<ArgExp, DIM>
    operator^(const Permutation<DIM>& p, const ArgExp& arg) {
      return PermuteTensor<ArgExp, DIM>(arg, p);
    }

  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_H__INCLUDED
