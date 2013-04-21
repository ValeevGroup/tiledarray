/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef TILEDARRAY_BINARY_TENSOR_H__INCLUDED
#define TILEDARRAY_BINARY_TENSOR_H__INCLUDED

#include <TiledArray/tensor_expression.h>
#include <TiledArray/tensor.h>
#include <TiledArray/bitset.h>

namespace TiledArray {
  namespace expressions {

    /// Place-holder object for a zero tensor.
    template <typename T>
    struct ZeroTensor {
      typedef T value_type;
    }; // struct ZeroTensor

    namespace detail {

      template <typename Op>
      class binary_transform_op {
      public:
        typedef typename Op::result_type result_type;
        typedef typename Op::first_argument_type first_argument_type;
        typedef typename Op::second_argument_type second_argument_type;

      private:
        Op op_;
        result_type scale_;

      public:
        binary_transform_op(Op op) : op_(op), scale_(1) { }
        binary_transform_op(const binary_transform_op<Op>& other) :
          op_(other.op_), scale_(other.scale_)
        { }
        binary_transform_op<Op>& operator=(const binary_transform_op<Op>& other) {
          op_ = other.op_;
          scale_ = other.scale_;
          return *this;
        }

        void scale(const result_type& value) { scale_ = value; }

        result_type operator()(const first_argument_type& first, const second_argument_type& second) const {
          return scale_ * op_(first, second);
        }
      }; // class binary_transform_op

      template <typename Op>
      class binary_and_op {
      public:
        typedef Tensor<typename madness::detail::result_of<Op>::type> result_type;
        typedef typename result_type::value_type value_type;

      private:
        typedef binary_and_op<Op> binary_and_op_;


        binary_transform_op<Op> op_; ///< The binary operation

      public:

        binary_and_op(Op op) : op_(op) { }

        /// Comparing two tensors for is_dense() quarry.

        /// \param l Left tensor is_dense result.
        /// \param r Right tensor is_dense result.
        /// \return True if the result tensor is dense.
        static bool is_dense(const bool left_dense, const bool right_dense) {
          return left_dense || right_dense;
        }

        /// Comparing two tiles for is_zero() quarry.

        /// \param l Left tile is_zero result.
        /// \param r Right tile is_zero result.
        /// \return True if the result tile is zero.
        static bool is_zero(const bool left_zero, const bool right_zero) {
          return left_zero && right_zero;
        }

        /// Construct a new bitset for shape.

        /// \param[in] result The shape that will store the results
        /// \param left The left argument shape.
        /// \param right The right argument shape.
        template <typename Left, typename Right>
        static void shape(::TiledArray::detail::Bitset<>& result, const Left& left, const Right& right) {
          if(left.is_dense()) {
            result.flip();
          } else {
            if(right.is_dense()) {
              result.flip();
            } else {
              result = left.get_shape();
              result |= right.get_shape();
            }
          }
        }

        /// Apply a scaling factor to this operation

        /// \param value The scaling factor for this operation
        void scale(const value_type value) { op_.scale(value); }

        template <typename Left, typename Right>
        result_type operator()(const Left& left, const Right& right) {
          return result_type(left.range(), left.begin(), right.begin(), op_);
        }

        template <typename Left, typename T>
        result_type operator()(const Left& left, const ZeroTensor<T>&) {
          return result_type(left.range(), left.begin(),
              std::bind2nd(op_, typename ZeroTensor<T>::value_type(0)));
        }

        template <typename T, typename Right>
        result_type operator()(const ZeroTensor<T>&, const Right& right) {
          return result_type(right.range(), right.begin(),
              std::bind1st(op_, typename ZeroTensor<T>::value_type(0)));
        }

      }; // class binary_and_op

      template <typename Op>
      class binary_or_op {
      public:
        typedef Tensor<typename madness::detail::result_of<Op>::type> result_type;
        typedef typename result_type::value_type value_type;

      private:
        typedef binary_or_op<Op> binary_or_op_;

        binary_transform_op<Op> op_;

      public:

        binary_or_op(Op op) : op_(op) { }

        /// Comparing two tensors for is_dense() quarry.

        /// \param l Left tensor is_dense result.
        /// \param r Right tensor is_dense result.
        /// \return True if the result tensor is dense.
        static bool is_dense(const bool left_dense, const bool right_dense) {
          return left_dense && right_dense;
        }

        /// Comparing two tiles for is_zero() quarry.

        /// \param l Left tile is_zero result.
        /// \param r Right tile is_zero result.
        /// \return True if the result tile is zero.
        static bool is_zero(const bool left_zero, const bool right_zero) {
          return left_zero || right_zero;
        }

        /// Construct a new bitset for shape.

        /// \param[in] result The shape that will store the results
        /// \param left The left argument shape.
        /// \param right The right argument shape.
        template <typename Left, typename Right>
        static void shape(::TiledArray::detail::Bitset<>& result, const Left& left, const Right& right) {
          if(left.is_dense()) {
            result = right.get_shape();
          } else {
            result = left.get_shape();
            if(! right.is_dense())
              result &= right.get_shape();
          }
        }

        /// Apply a scaling factor to this operation

        /// \param value The scaling factor for this operation
        void scale(const value_type value) { op_.scale(value); }

        template <typename Left, typename Right>
        result_type operator()(const Left& left, const Right& right) {
          return result_type(left.range(), left.begin(), right.begin(), op_);
        }

        template <typename T, typename Right>
        result_type operator()(const ZeroTensor<T>&, const Right&) {
          TA_ASSERT(false); // Should not be used.
          return result_type();
        }

        template <typename Left, typename T>
        result_type operator()(const Left&, const ZeroTensor<T>&) {
          TA_ASSERT(false); // Should not be used.
          return result_type();
        }

      }; // class binary_or_op



    } // namespace detail

    template <typename Op>
    struct BinaryOpSelect {
      typedef detail::binary_and_op<Op> type;
    }; // struct BinaryOpSelect

    template <typename T>
    struct BinaryOpSelect<std::multiplies<T> > {
      typedef detail::binary_or_op<std::multiplies<T> > type;
    }; // struct BinaryOpSelect

    template <typename Op>
    typename BinaryOpSelect<Op>::type make_binary_tile_op(const Op& op) {
      return typename BinaryOpSelect<Op>::type(op);
    }

    namespace detail {

      /// Tensor that is composed from two argument tensors

      /// A binary operator is used to transform the individual elements of the tiles.
      /// \tparam Left The left argument type
      /// \tparam Right The right argument type
      /// \tparam Op The binary transform operator type.
      template <typename LExp, typename RExp, typename Op>
      class BinaryTensorImpl :
        public TensorExpressionImpl<typename Op::result_type>
      {
      public:
        typedef Op op_type;
        typedef BinaryTensorImpl<LExp, RExp, Op> BinaryTensorImpl_;
        typedef TensorExpressionImpl<typename op_type::result_type> TensorExpressionImpl_;
        typedef typename TensorExpressionImpl_::TensorImpl_ TensorImpl_;
        typedef LExp left_tensor_type;
        typedef RExp right_tensor_type;
        typedef typename TensorExpressionImpl_::size_type size_type;
        typedef typename TensorExpressionImpl_::range_type range_type;
        typedef typename TensorExpressionImpl_::shape_type shape_type;
        typedef typename TensorExpressionImpl_::pmap_interface pmap_interface;
        typedef typename TensorExpressionImpl_::trange_type trange_type;
        typedef typename TensorExpressionImpl_::value_type value_type;
        typedef typename TensorExpressionImpl_::const_reference const_reference;
        typedef typename TensorExpressionImpl_::const_iterator const_iterator;

      private:
        // Not allowed
        BinaryTensorImpl_& operator=(const BinaryTensorImpl_&);
        BinaryTensorImpl(const BinaryTensorImpl_&);

      public:

        /// Construct a unary tensor op

        /// \param arg The argument
        /// \param op The element transform operation
        BinaryTensorImpl(const left_tensor_type& left, const right_tensor_type& right, const Op& op) :
            TensorExpressionImpl_(left.get_world(), left.vars(), left.trange(),
                (op.is_dense(left.is_dense(), right.is_dense()) ? 0ul : left.size())),
            op_(op), left_(left), right_(right)
        {
          TA_ASSERT(left_.size() == right_.size());
        }

        virtual ~BinaryTensorImpl() { }

      private:

        static bool done(const bool left, const bool right) { return left && right; }

        template <typename L, typename R>
        void eval_tile(const size_type i, const L& left, const R& right) {
          TensorExpressionImpl_::set(i, value_type(op_(left, right)));
        }

        /// Function for evaluating this tensor's tiles

        /// This function is run inside a task, and will run after \c eval_children
        /// has completed. It should spawn additional tasks that evaluate the
        /// individual result tiles.
        virtual void eval_tiles() {
          typedef ZeroTensor<typename left_tensor_type::value_type::value_type> zero_left_type;
          typedef ZeroTensor<typename right_tensor_type::value_type::value_type> zero_right_type;

          // Set the scale factor
          op_.scale(TensorExpressionImpl_::scale());

          // Construct local iterator
          typename pmap_interface::const_iterator it = TensorImpl_::pmap()->begin();
          const typename pmap_interface::const_iterator end = TensorImpl_::pmap()->end();

          if(left_.is_dense() && right_.is_dense() && TensorImpl_::is_dense()) {
            // Evaluate tiles where both arguments and the result are dense
            for(; it != end; ++it) {
              const size_type i = *it;
              TensorImpl_::get_world().taskq.add(this,
                  & BinaryTensorImpl_::template eval_tile<typename left_tensor_type::value_type, typename right_tensor_type::value_type>,
                  i, left_.move(i), right_.move(i));
            }
          } else {
            // Evaluate tiles where the result or one of the arguments is sparse
            for(; it != end; ++it) {
              const size_type i = *it;
              if(! TensorImpl_::is_zero(i)) {
                if(left_.is_zero(i)) {
                  TensorImpl_::get_world().taskq.add(this,
                    & BinaryTensorImpl_::template eval_tile<zero_left_type, typename right_tensor_type::value_type>,
                    i, zero_left_type(), right_.move(i));
                } else if(right_.is_zero(i)) {
                  TensorImpl_::get_world().taskq.add(this,
                    & BinaryTensorImpl_::template eval_tile<typename left_tensor_type::value_type, zero_right_type>,
                    i, left_.move(i), zero_right_type());
                } else {
                  TensorImpl_::get_world().taskq.add(this,
                    & BinaryTensorImpl_::template eval_tile<typename left_tensor_type::value_type, typename right_tensor_type::value_type>,
                    i, left_.move(i), right_.move(i));
                }
              } else {
                // Cleanup unused tiles
                if(! left_.is_zero(i))
                  left_.move(i);
                if(! right_.is_zero(i))
                  right_.move(i);
              }
            }
          }

          left_.release();
          right_.release();
        }

        /// Function for evaluating child tensors

        /// This function should return true when the child

        /// This function should evaluate all child tensors.
        /// \param vars The variable list for this tensor (may be different from
        /// the variable list used to initialize this tensor).
        /// \param pmap The process map for this tensor
        virtual madness::Future<bool> eval_children(const expressions::VariableList& vars,
            const std::shared_ptr<pmap_interface>& pmap) {
          // The default behavior, where left vars equal right vars, is to do the
          // tile permutation (if necessary) in this object since it is less
          // expensive.
          // Note: This function assumes, vars == left vars
          const VariableList* left_vars = & left_.vars();
          const VariableList* right_vars = & right_.vars();

          if(*left_vars != *right_vars) {
            // Deside who is going to permute
            if(*left_vars == vars) {
              right_vars = left_vars; // Permute right argument
            } else if(*right_vars == vars) {
              left_vars = right_vars; // Permute left argument
              TensorExpressionImpl_::vars(*right_vars);
            } else {
              left_vars = right_vars = & vars; // Permute left and right arguments
              TensorExpressionImpl_::vars(vars);
            }
          }

          madness::Future<bool> left_done = left_.eval(*left_vars, pmap->clone());
          madness::Future<bool> right_done = right_.eval(*right_vars, pmap);
          return TensorImpl_::get_world().taskq.add(& BinaryTensorImpl_::done,
              left_done, right_done, madness::TaskAttributes::hipri());
        }

        /// Construct the shape object

        /// This function is used by derived classes to create a shape object. It
        /// is run inside a task with the proper dependencies to ensure data
        /// consistency. This function is only called when the tensor is not dense.
        /// \param shape The existing shape object
        virtual void make_shape(shape_type& shape) const {
          op_.shape(shape, left_, right_);
        }

        op_type op_; ///< binary element operator
        left_tensor_type left_; ///< Left argument
        right_tensor_type right_; ///< Right argument
      }; // class BinaryTensorImpl

    } // namespace detail

    template <typename LExp, typename RExp, typename Op>
    TensorExpression<typename Op::result_type>
    make_binary_tensor(const LExp& left, const RExp& right, const Op& op) {
      typedef detail::BinaryTensorImpl<LExp, RExp, Op> impl_type;
      std::shared_ptr<detail::TensorExpressionImpl<typename Op::result_type> > pimpl(
          new impl_type(left, right, op),
          madness::make_deferred_deleter<impl_type>(left.get_world()));
      return TensorExpression<typename Op::result_type>(pimpl);
    }

  }  // namespace expressions
}  // namespace TiledArray

#endif // TILEDARRAY_BINARY_TILED_TENSOR_H__INCLUDED
