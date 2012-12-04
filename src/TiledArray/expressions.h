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

      /// This task will reduce the elements of all tile into a scalar value

      /// \tparam Arg The tiled tensor reduction argument type
      /// \tparam Op The reduction operation type
      template <typename Arg, typename Op>
      class ReduceTiles : public madness::TaskInterface {
      public:
        typedef typename madness::detail::result_of<Op>::type result_type;
      private:

        const Arg arg_; ///< The tiled tensor argument
        const Op op_; ///< The reduction operation
        madness::Future<result_type> result_; ///< The reduction result

        template <typename Exp>
        static madness::Future<typename Exp::value_type>
        tile(const ReadableTiledTensor<Exp>& arg, const std::size_t i) {
          return arg[i];
        }

        template <typename T, typename CS>
        static madness::Future<typename Array<T, CS>::value_type>
        tile(const Array<T, CS>& array, const std::size_t i) {
          return array.find(i);
        }

      public:

        /// Constructor

        /// \param arg The tiled tensor to be reduced
        /// \param op The reduction operatioin
        /// \param dep The evaluation dependancy
        ReduceTiles(const Arg& arg, const Op& op, madness::Future<bool>& dep) :
            madness::TaskInterface(madness::TaskAttributes::hipri()),
            arg_(arg), op_(op), result_()
        {
          if(! dep.probe()) {
            madness::DependencyInterface::inc();
            dep.register_callback(this);
          }
        }

        /// Result accessor

        /// \return A future for the result of this task
        const madness::Future<result_type>& result() const {
          return result_;
        }

        /// Task function
        virtual void run(const madness::TaskThreadEnv&) {
          // Create reduce task object
          TiledArray::detail::ReduceTask<Op> reduce_task(arg_.get_world(), op_);

          // Spawn reduce tasks for each local tile.
          typename Arg::pmap_interface::const_iterator end = arg_.get_pmap()->end();
          typename Arg::pmap_interface::const_iterator it = arg_.get_pmap()->begin();
          if(arg_.is_dense()) {
            for(; it != end; ++it)
              reduce_task.add(arg_.get_world().taskq.add(& reduce_tile<typename Arg::value_type, Op>,
                  tile(arg_, *it), op_, madness::TaskAttributes::hipri()));
          } else {
            for(; it != end; ++it)
              if(! arg_.is_zero(*it))
                reduce_task.add(arg_.get_world().taskq.add(& reduce_tile<typename Arg::value_type, Op>,
                    tile(arg_, *it), op_, madness::TaskAttributes::hipri()));
          }

          // Set the result future
          result_.set(reduce_task.submit());
        }
      }; // class class ReduceTiles


      /// Evaluate a \c ReadableTiledTensor

      /// \tparam Exp The readable tiled tensor expression type
      /// \param arg The tiled tensor to evaluate
      /// \return A future to a bool that will be set once \c arg has been evaluated.
      template <typename Exp>
      inline madness::Future<bool> eval(const ReadableTiledTensor<Exp>& arg) {
        return const_cast<ReadableTiledTensor<Exp>& >(arg).derived().eval(arg.vars(),
            std::shared_ptr<TiledArray::Pmap<typename Exp::size_type> >(
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
      // Evaluate the argument tensor
      madness::Future<bool> arg_eval = detail::eval(arg);

      // Spawn a task that will generate reduction tasks for each local tile
      detail::ReduceTiles<Exp, Op>* reduce_task =
          new detail::ReduceTiles<Exp, Op>(arg.derived(), op, arg_eval);

      // Spawn the task
      madness::Future<typename madness::detail::result_of<Op>::type> local_result =
          reduce_task->result();
      arg.get_world().taskq.add(reduce_task);

      // Wait for the local reduction result
      typename madness::detail::result_of<Op>::type result = local_result.get();

      // All to all global reduction
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
