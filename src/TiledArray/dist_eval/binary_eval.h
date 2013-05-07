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

#include <TiledArray/dist_eval.h>
#include <TiledArray/tensor.h>

namespace TiledArray {
  namespace detail {

    /// Binary, distributed tensor evaluator

    /// This object is used to evaluate the tiles of a distributed binary
    /// expression.
    /// \tparam Left The left argument type
    /// \tparam Right The right argument type
    /// \tparam Op The binary transform operator type.
    template <typename Left, typename Right, typename Op>
    class BinaryEvalImpl :
      public DistEvalImpl<typename Op::result_type>
    {
    public:
      typedef Op op_type;
      typedef BinaryTensorImpl<LExp, RExp, Op> BinaryEvalImpl_;
      typedef DistEvalImpl<typename op_type::result_type> DistEvalImpl_;
      typedef typename DistEvalImpl_::TensorImpl_ TensorImpl_;
      typedef Left left_type;
      typedef Right right_type;
      typedef typename DistEvalImpl_::size_type size_type;
      typedef typename DistEvalImpl_::range_type range_type;
      typedef typename DistEvalImpl_::shape_type shape_type;
      typedef typename DistEvalImpl_::pmap_interface pmap_interface;
      typedef typename DistEvalImpl_::trange_type trange_type;
      typedef typename DistEvalImpl_::value_type value_type;
      typedef typename DistEvalImpl_::const_reference const_reference;
      typedef typename DistEvalImpl_::const_iterator const_iterator;

    public:

      /// Construct a unary tensor op

      /// \param arg The argument
      /// \param op The element transform operation
      BinaryEvalImpl(const left_type& left, const right_type& right,
          madness::World& world, const Permutation& perm, const shape_type& shape,
          const std::shared_ptr<pmap_interface>& pmap, const op_type& op,
          const bool permute_tiles) :
            DistEvalImpl_(world, perm, left.trange(), shape, pmap, perm_tiles),
        op_(op), left_(left), right_(right)
      {
        TA_ASSERT(left.trange() == right.trange());
      }

      virtual ~BinaryEvalImpl() { }

    private:

      template <typename L, typename R>
      void eval_tile(const size_type i, const L& left, const R& right,
          madness::AtomicInt* const counter)
      {
        DistEvalImpl_::set(i, op_(left, right), op_.permute());
        (*counter)++;
      }

      /// Function for evaluating this tensor's tiles

      /// This function is run inside a task, and will run after \c eval_children
      /// has completed. It should spawn additional tasks that evaluate the
      /// individual result tiles.
      virtual void eval_tiles(const std::shared_ptr<DistEvalImpl>& pimpl,
          madness::AtomicInt& counter, int& task_count)
      {
        typedef ZeroTensor<typename left_type::value_type::value_type> zero_left_type;
        typedef ZeroTensor<typename right_type::value_type::value_type> zero_right_type;

        // Convert pimpl to this object type so it can be used in tasks
        TA_ASSERT(this == pimpl.get());
        std::shared_ptr<BinaryEvalImpl_> this_pimpl =
            std::static_pointer_cast<BinaryEvalImpl_>(pimpl);

        // Construct local iterator
        typename pmap_interface::const_iterator it = TensorImpl_::pmap()->begin();
        const typename pmap_interface::const_iterator end = TensorImpl_::pmap()->end();

        if(left_.is_dense() && right_.is_dense() && TensorImpl_::is_dense()) {
          // Evaluate tiles where both arguments and the result are dense
          for(; it != end; ++it) {
            const size_type i = *it;
            TensorImpl_::get_world().taskq.add(this_pimpl,
                & BinaryEvalImpl_::template eval_tile<typename left_type::value_type, typename right_type::value_type>,
                i, left_.move(i), right_.move(i), &counter);
            ++task_count;
          }
        } else {
          // Evaluate tiles where the result or one of the arguments is sparse
          for(; it != end; ++it) {
            const size_type i = *it;
            if(! TensorImpl_::is_zero(i)) {
              if(left_.is_zero(i)) {
                TensorImpl_::get_world().taskq.add(this_pimpl,
                  & BinaryEvalImpl_::template eval_tile<zero_left_type, typename right_type::value_type>,
                  i, zero_left_type(), right_.move(i), &counter);
              } else if(right_.is_zero(i)) {
                TensorImpl_::get_world().taskq.add(this_pimpl,
                  & BinaryEvalImpl_::template eval_tile<typename left_type::value_type, zero_right_type>,
                  i, left_.move(i), zero_right_type(), &counter);
              } else {
                TensorImpl_::get_world().taskq.add(this_pimpl,
                  & BinaryEvalImpl_::template eval_tile<typename left_type::value_type, typename right_type::value_type>,
                  i, left_.move(i), right_.move(i), &counter);
              }
              ++task_count;
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
      virtual void eval_children(madness::AtomicInt& counter, long& task_count) {
        left_.eval(counter, task_counter);
        right_.eval(counter, task_counter);
      }

      op_type op_; ///< binary element operator
      left_type left_; ///< Left argument
      right_type right_; ///< Right argument
    }; // class BinaryTensorImpl

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_DIST_EVAL_BINARY_EVAL_H__INCLUDED
