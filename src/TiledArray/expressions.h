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
#include <world/typestuff.h>

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

    namespace detail {

      /// Task function for reducing a tile

      /// \tparam T The \c Tensor value type
      /// \tparam R The \c Tensor range type
      /// \tparam A The \c Tensor allocator type
      /// \tparam Op The reduction operation type
      /// \param t The tensor to be reduced
      /// \param op The reduction operation
      /// \return The reduced value of \c t
      template <typename T, typename Op>
      typename madness::detail::result_of<Op>::type reduce_tile(const T& t, const Op& op) {
        typename T::value_type result = typename T::value_type();
        for(typename T::const_iterator it = t.begin(); it != t.end(); ++it)
          op(result, *it);

        return result;
      }

      /// Task function for reducing tiles

      /// \tparam Arg The arg type: \c Array or \c \c ReadableTiledTensor
      /// \tparam Op The reduction operation type
      /// \param arg The array or tile tensor object to be reduced
      /// \param op The reduction operation
      /// \return The reduced value of all local tiles.
      /// \note The last unused parameter is used to set the dependency between
      /// the evaluation of \c arg and this task.
      template <typename Arg, typename Op>
      typename madness::detail::result_of<Op>::type reduce_tiles(const Arg& arg, const Op& op, bool) {
        TiledArray::detail::ReduceTask<Op>
            reduce_task(arg.get_world(), op);
        typename Arg::const_iterator end = arg.end();
        for(typename Arg::const_iterator it = arg.begin(); it != end; ++it)
          reduce_task.add(arg.get_world().taskq.add(& reduce_tile<typename Arg::value_type, Op>,
              *it, op, madness::TaskAttributes::hipri()));

        return reduce_task.submit().get();
      }

      /// Evaluate a \c ReadableTiledTensor

      /// \tparam Exp The readable tiled tensor expression type
      /// \param arg The tiled tensor to evaluate
      /// \return A future to a bool that will be set once \c arg has been evaluated.
      template <typename Exp>
      inline madness::Future<bool> eval(const ReadableTiledTensor<Exp>& arg) {
        return const_cast<ReadableTiledTensor<Exp>& >(arg).derived().eval(arg.vars(),
            std::shared_ptr<TiledArray::Pmap>(
            new TiledArray::detail::BlockedPmap(arg.get_world(), arg.size())));
      }

      /// Evaluate an \c Array

      /// \tparam T The array element type
      /// \tparam CS The array coordinate system type
      /// \param arg The array to evaluate
      /// \return A future to a bool that will be set once \c arg has been evaluated.
      template <typename T, typename CS>
      inline madness::Future<bool> eval(const Array<T, CS>& array) {
        return const_cast<Array<T, CS>&>(array).eval();
      }

      template <typename T>
      struct plus {
        typedef T result_type;
        typedef T argument_type;
        typedef std::plus<T> std_op_type;

        result_type operator()() const { return result_type(); }

        void operator()(result_type& result, const argument_type& arg) const {
          result += arg;
        }

        void operator()(result_type& result, const argument_type& arg1, const argument_type& arg2) const {
          result += arg1 + arg2;
        }

        template <typename Archive>
        void serialize(const Archive&) { }
      };

    } // namespace detail

    /// Task function for reducing a tile

    /// \tparam T The \c Tensor value type
    /// \tparam R The \c Tensor range type
    /// \tparam A The \c Tensor allocator type
    /// \tparam Op The reduction operation type
    /// \param t The tensor to be reduced
    /// \param op The reduction operation
    /// \return The reduced value of \c t
    template <typename T, typename R, typename A, typename Op>
    typename madness::detail::result_of<Op>::type reduce(const Tensor<T, R, A>& t, const Op& op) {
      return detail::reduce_tile(t, op);
    }

    /// Reduce an \c Array or a \c ReadableTiledTensor

    /// This function will reduce all elements of \c arg . The result of the
    /// reduction is returned on all nodes. The function will block, until the
    /// reduction is complete, but it will continue to process tasks while
    /// waiting.
    /// \f[
    /// C = \sum_{i_1, i_2, \dots}  A_{i_1, i_2, \dots}
    /// \f]
    /// \tparam Arg The \c Array or \c ReadableTiledTensor type
    /// \tparam Op The reduction operation type
    /// \param arg The array or tile tensor object to be reduced
    /// \param op The reduction operation
    /// \return The reduced value of the tensor.
    template <typename Exp, typename Op>
    inline typename madness::detail::result_of<Op>::type
    reduce(const ReadableTiledTensor<Exp>& arg, const Op& op) {
      typename madness::detail::result_of<Op>::type result =
          arg.get_world().taskq.add(arg.get_world().rank(),
          detail::reduce_tiles<Exp, Op>, arg.derived(), op, detail::eval(arg),
          madness::TaskAttributes::hipri()).get();
      arg.get_world().gop.reduce(& result, 1, typename Op::std_op_type());
      return result;
    }

    /// Calculate the dot product of two tiled tensors

    /// \f[
    /// C = \sum_{i_1, i_2, \dots}  A_{i_1, i_2, \dots} B_{i_1, i_2, \dots}
    /// \f]
    /// \param left The left tensor argument ( \c A )
    /// \param right The right tiled tensor argument ( \c B )
    /// \return The sum of the products of each element in \c left and \c right ( \c C )
    template <typename LeftArg, typename RightArg>
    inline typename math::ContractionValue<typename LeftArg::value_type::value_type,
        typename RightArg::value_type::value_type>::type
    dot(const ReadableTiledTensor<LeftArg>& left, const ReadableTiledTensor<RightArg>& right) {
      typedef typename math::ContractionValue<typename LeftArg::value_type::value_type,
              typename RightArg::value_type::value_type>::type result_type;
      return reduce(make_binary_tiled_tensor(left.derived(), right.derived(),
          std::multiplies<result_type>()), detail::plus<result_type>());
    }

    template <typename Arg>
    typename Arg::value_type::value_type
    norm2(const ReadableTiledTensor<Arg>& arg) {
      return std::sqrt(reduce(make_unary_tiled_tensor(arg.derived(),
          TiledArray::detail::Square<typename Arg::value_type::value_type>()),
          detail::plus<typename Arg::value_type::value_type>()));
    }

  } // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_H__INCLUDED
