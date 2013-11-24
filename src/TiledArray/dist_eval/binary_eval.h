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

#ifndef TILEDARRAY_DIST_EVAL_BINARY_EVAL_H__INCLUDED
#define TILEDARRAY_DIST_EVAL_BINARY_EVAL_H__INCLUDED

#include <TiledArray/dist_eval/dist_eval.h>
#include <TiledArray/tensor.h>

namespace TiledArray {
  namespace detail {

    /// Binary, distributed tensor evaluator

    /// This object is used to evaluate the tiles of a distributed binary
    /// expression.
    /// \tparam Left The left argument type
    /// \tparam Right The right argument type
    /// \tparam Op The binary transform operator type
    /// \tparam Policy The tensor policy class
    template <typename Left, typename Right, typename Op, typename Policy>
    class BinaryEvalImpl :
      public DistEvalImpl<typename Op::result_type, Policy>
    {
    public:
      typedef BinaryEvalImpl<Left, Right, Op, Policy> BinaryEvalImpl_; ///< This object type
      typedef DistEvalImpl<typename Op::result_type, Policy> DistEvalImpl_; ///< The base class type
      typedef typename DistEvalImpl_::TensorImpl_ TensorImpl_; ///< The base, base class type
      typedef Left left_type; ///< The left-hand argument type
      typedef Right right_type; ///< The right-hand argument type
      typedef typename DistEvalImpl_::size_type size_type; ///< Size type
      typedef typename DistEvalImpl_::range_type range_type; ///< Range type
      typedef typename DistEvalImpl_::shape_type shape_type; ///< Shape type
      typedef typename DistEvalImpl_::pmap_interface pmap_interface; ///< Process map interface type
      typedef typename DistEvalImpl_::trange_type trange_type; ///< Tiled range type
      typedef typename DistEvalImpl_::value_type value_type; ///< Tile type
      typedef typename DistEvalImpl_::eval_type eval_type; ///< Tile evaluation type
      typedef Op op_type; ///< Tile evaluation operator type

    private:

      left_type left_; ///< Left argument
      right_type right_; ///< Right argument
      op_type op_; ///< binary element operator

    public:

      /// Construct a unary tensor op

      /// \param arg The argument
      /// \param op The element transform operation
      BinaryEvalImpl(const left_type& left, const right_type& right,
          madness::World& world, const shape_type& shape,
          const std::shared_ptr<pmap_interface>& pmap, const Permutation& perm,
          const op_type& op) :
        DistEvalImpl_(world, left.trange(), shape, pmap, perm),
        left_(left), right_(right), op_(op)
      {
        TA_ASSERT(left.trange() == right.trange());
      }

      virtual ~BinaryEvalImpl() { }

    private:

      /// Task function for evaluating tiles

      /// \param i The tile index
      /// \param left The left-hand tile
      /// \param right The right-hand tile
      template <typename L, typename R>
      void eval_tile(const size_type i, L left, R right) {
        DistEvalImpl_::set_tile(i, op_(left, right));
      }

      /// Function for evaluating this tensor's tiles

      /// This function is run inside a task, and will run after \c eval_children
      /// has completed. It should spawn additional tasks that evaluate the
      /// individual result tiles.
      virtual size_type internal_eval(const std::shared_ptr<DistEvalImpl_>& pimpl) {
        // Convert pimpl to this object type so it can be used in tasks
        std::shared_ptr<BinaryEvalImpl_> self =
            std::static_pointer_cast<BinaryEvalImpl_>(pimpl);

        // Evaluate child tensors
        left_.eval();
        right_.eval();

        // Task function argument types
        typedef typename madness::if_<std::is_const<typename op_type::first_argument_type>,
            const typename left_type::value_type,
                  typename left_type::value_type>::type &
                left_argument_type;
        typedef typename madness::if_<std::is_const<typename op_type::second_argument_type>,
            const typename right_type::value_type,
                  typename right_type::value_type>::type &
                right_argument_type;
        typedef ZeroTensor<typename left_type::value_type::value_type> zero_left_type;
        typedef ZeroTensor<typename right_type::value_type::value_type> zero_right_type;

        size_type task_count = 0ul;

        // Construct local iterator
        TA_ASSERT(left_.pmap() == right_.pmap());
        typename pmap_interface::const_iterator it = left_.pmap()->begin();
        const typename pmap_interface::const_iterator end = left_.pmap()->end();

        if(left_.is_dense() && right_.is_dense() && TensorImpl_::is_dense()) {
          // Evaluate tiles where both arguments and the result are dense
          for(; it != end; ++it) {
            // Get tile indices
            const size_type index = *it;
            const size_type target_index = DistEvalImpl_::perm_index(index);

            // Schedule tile evaluation task
            TensorImpl_::get_world().taskq.add(self,
                & BinaryEvalImpl_::template eval_tile<left_argument_type, right_argument_type>,
                target_index, left_.move(index), right_.move(index));

            ++task_count;
          }
        } else {
          // Evaluate tiles where the result or one of the arguments is sparse
          for(; it != end; ++it) {
            // Get tile indices
            const size_type index = *it;
            const size_type target_index = DistEvalImpl_::perm_index(index);

            if(! TensorImpl_::is_zero(target_index)) {
              // Schedule tile evaluation task
              if(left_.is_zero(index)) {
                TensorImpl_::get_world().taskq.add(self,
                  & BinaryEvalImpl_::template eval_tile<const zero_left_type, right_argument_type>,
                  target_index, zero_left_type(), right_.move(index));
              } else if(right_.is_zero(index)) {
                TensorImpl_::get_world().taskq.add(self,
                  & BinaryEvalImpl_::template eval_tile<left_argument_type, const zero_right_type>,
                  target_index, left_.move(index), zero_right_type());
              } else {
                TensorImpl_::get_world().taskq.add(self,
                  & BinaryEvalImpl_::template eval_tile<left_argument_type, right_argument_type>,
                  target_index, left_.move(index), right_.move(index));
              }

              ++task_count;
            } else {
              // Cleanup unused tiles
              if(! left_.is_zero(index))
                left_.move(index);
              if(! right_.is_zero(index))
                right_.move(index);
            }
          }
        }

        // Wait for child tensors to be evaluated, and process tasks while waiting.
        left_.wait();
        right_.wait();

        return task_count;
      }

    }; // class BinaryEvalImpl


    /// Distrubuted unary evaluator factory function

    /// Construct a distributed unary evaluator, which constructs a new tensor
    /// by applying \c op to tiles of \c arg .
    /// \tparam Tile Tile type of the argument
    /// \tparam Policy The policy type of the argument
    /// \tparam Op The unary tile operation
    /// \param arg Argument to be modified
    /// \param world The world where the argument will be evaluated
    /// \param shape The shape of the evaluated tensor
    /// \param pmap The process map for the evaluated tensor
    /// \param perm The permutation applied to the tensor
    /// \param op The unary tile operation
    template <typename LeftTile, typename RightTile, typename Policy, typename Op>
    DistEval<typename Op::result_type, Policy> make_binary_eval(
        const DistEval<LeftTile, Policy>& left,
        const DistEval<RightTile, Policy>& right,
        madness::World& world,
        const typename DistEval<typename Op::result_type, Policy>::shape_type& shape,
        const std::shared_ptr<typename DistEval<typename Op::result_type, Policy>::pmap_interface>& pmap,
        const Permutation& perm,
        const Op& op)
    {
      typedef BinaryEvalImpl<DistEval<LeftTile, Policy>, DistEval<RightTile,
          Policy>, Op, Policy> impl_type;
      typedef typename impl_type::DistEvalImpl_ impl_base_type;
      return DistEval<typename Op::result_type, Policy>(
          std::shared_ptr<impl_base_type>(new impl_type(left, right, world,
              shape, pmap, perm, op)));
    }

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_DIST_EVAL_BINARY_EVAL_H__INCLUDED
