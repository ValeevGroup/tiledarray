#ifndef TILEDARRAY_UNARY_TENSOR_H__INCLUDED
#define TILEDARRAY_UNARY_TENSOR_H__INCLUDED

#include <TiledArray/tensor_expression.h>
#include <TiledArray/tensor.h>

namespace TiledArray {
  namespace expressions {
    namespace detail {

      template <typename Op>
      class UnaryTileOp {
      public:
        typedef typename Op::result_type value_type;
        typedef Tensor<value_type> result_type;
        typedef const Tensor<value_type>& argument_type;

      private:

        class transform_op {
        public:
          typedef typename Op::result_type result_type;
          typedef typename Op::argument_type argument_type;

        private:
          Op op_;
          result_type scale_;

        public:
          transform_op(Op op) : op_(op), scale_(1) { }
          transform_op(const transform_op& other) :
            op_(other.op_), scale_(other.scale_)
          { }
          transform_op& operator=(const transform_op& other) {
            op_ = other.op_;
            scale_ = other.scale_;
            return *this;
          }

          void scale(const result_type& value) { scale_ = value; }

          result_type operator()(const argument_type& arg) const { return scale_ * op_(arg); }
        } op_;

      public:

        UnaryTileOp(const Op& op) : op_(op) { }
        UnaryTileOp(const UnaryTileOp<Op>& other) : op_(other.op_) { }
        UnaryTileOp<Op>& operator=(const UnaryTileOp<Op>& other) {
          op_ = other.op_;
          return *this;
        }

        void scale(const value_type value) { op_.scale(value); }

        result_type operator()(argument_type arg) const {
          return result_type(arg.range(), ::TiledArray::detail::make_tran_it(arg.begin(), op_));
        }

      }; // class UnaryTileOp

    }  // namespace detail

    template <typename Op>
    detail::UnaryTileOp<Op> make_unary_tile_op(const Op& op) {
      return detail::UnaryTileOp<Op>(op);
    }

    namespace detail {

      /// Tensor that is composed from an argument tensor

      /// The tensor elements are constructed using a unary transformation
      /// operation.
      /// \tparam Arg The argument type
      /// \tparam Op The Unary transform operator type.
      template <typename Exp, typename Op>
      class UnaryTensorImpl : public TensorExpressionImpl<typename Op::result_type> {
      public:
        typedef UnaryTensorImpl<Exp, Op> UnaryTensorImpl_; ///< This object type
        typedef Exp arg_tensor_type; ///< The argument tensor type
        typedef TensorExpressionImpl<typename Op::result_type> TensorExpressionImpl_; ///< The base class type
        typedef typename TensorExpressionImpl_::TensorImpl_ TensorImpl_; ///< The base, base class type
        typedef typename TensorExpressionImpl_::size_type size_type; ///< Size type
        typedef typename TensorExpressionImpl_::range_type range_type; ///< Range type
        typedef typename TensorExpressionImpl_::pmap_interface pmap_interface; ///< Process map interface type
        typedef typename TensorExpressionImpl_::trange_type trange_type; ///< tiled range type
        typedef typename TensorExpressionImpl_::value_type value_type; ///< value type
        typedef typename TensorExpressionImpl_::const_reference const_reference; ///< const reference type
        typedef typename TensorExpressionImpl_::const_iterator const_iterator; ///< const iterator type

      private:
        // Not allowed
        UnaryTensorImpl(const UnaryTensorImpl_& other);
        UnaryTensorImpl_& operator=(const UnaryTensorImpl_&);

      public:

        /// Constructor

        /// \param arg The argument
        /// \param op The element transform operation
        UnaryTensorImpl(const arg_tensor_type& arg, const Op& op) :
            TensorExpressionImpl_(arg.get_world(), arg.vars(), arg.trange(),
                (arg.is_dense() ? 0ul : arg.size())),
            arg_(arg),
            op_(op)
        { }

        /// Virtual destructor
        virtual ~UnaryTensorImpl() { }

      private:

        void eval_tile(const size_type i, const typename arg_tensor_type::value_type& tile) {
          value_type result = op_(tile);
          TensorExpressionImpl_::set(i, madness::move(result));
        }

        /// Function for evaluating this tensor's tiles

        /// This function is run inside a task, and will run after \c eval_children
        /// has completed. It should spwan additional tasks that evaluate the
        /// individule result tiles.
        virtual void eval_tiles() {
          // Set the scale factor
          op_.scale(TensorExpressionImpl_::scale());

          // Make sure all local tiles are present.
          const typename pmap_interface::const_iterator end = TensorImpl_::pmap()->end();
          typename pmap_interface::const_iterator it = TensorImpl_::pmap()->begin();
          if(arg_.is_dense()) {
            for(; it != end; ++it)
              TensorImpl_::get_world().taskq.add(this,
                  & UnaryTensorImpl_::eval_tile, *it, arg_.move(*it));
          } else {
            for(; it != end; ++it)
              if(! arg_.is_zero(*it))
                TensorImpl_::get_world().taskq.add(this,
                    & UnaryTensorImpl_::eval_tile, *it, arg_.move(*it));
          }

          arg_.release();
        }

        /// Function for evaluating child tensors

        /// This function should return true when the child

        /// This function should evaluate all child tensors.
        /// \param vars The variable list for this tensor (may be different from
        /// the variable list used to initialize this tensor).
        /// \param pmap The process map for this tensor
        virtual madness::Future<bool> eval_children(const expressions::VariableList& vars,
            const std::shared_ptr<pmap_interface>& pmap) {
          TensorExpressionImpl_::vars(vars);
          return arg_.eval(vars, pmap);
        }

        /// Construct the shape object

        /// This function is used by derived classes to create a shape object. It
        /// is run inside a task with the proper dependencies to ensure data
        /// consistancy. This function is only called when the tensor is not dense.
        /// \param shape The existing shape object
        virtual void make_shape(TiledArray::detail::Bitset<>& shape) const {
          TA_ASSERT(shape.size() == arg_.size());
          shape = arg_.get_shape();
        }

        arg_tensor_type arg_; ///< Argument
        Op op_; ///< The unary element opertation
      }; // class UnaryTensorImpl

    } // namespace detail

    template <typename Exp, typename Op>
    TensorExpression<typename Op::result_type>
    make_unary_tensor(const Exp& arg, const Op& op) {
      typedef detail::UnaryTensorImpl<Exp, Op> impl_type;
      std::shared_ptr<detail::TensorExpressionImpl<typename Op::result_type> > pimpl(
          new detail::UnaryTensorImpl<Exp, Op>(arg, op),
          madness::make_deferred_deleter<detail::UnaryTensorImpl<Exp, Op> >(arg.get_world()));
      return TensorExpression<typename Op::result_type>(pimpl);
    }

  }  // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_UNARY_TENSOR_H__INCLUDED
