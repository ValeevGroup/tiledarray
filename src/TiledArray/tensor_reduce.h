/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
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

#ifndef TILEDARRAY_TENSOR_REDUCE_H__INCLUDED
#define TILEDARRAY_TENSOR_REDUCE_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/madness.h>

namespace TiledArray {
  namespace expressions {

    /// This task will reduce the tiles of a tensor expression

    /// \tparam Exp The tensor expression argument type
    /// \tparam Op The reduction operation type
    template <typename Exp, typename Op>
    class ReduceTensorExpression : public madness::TaskInterface {
    public:
      typedef typename madness::detail::result_of<Op>::type result_type;
    private:

      Exp arg_; ///< The tensor expression argument
      const Op op_; ///< The reduction operation
      madness::Future<result_type> result_; ///< The reduction result

    public:

      /// Constructor

      /// \param arg The tensor expression to be reduced
      /// \param op The reduction operation
      ReduceTensorExpression(const Exp& arg, const Op& op) :
          madness::TaskInterface(1, madness::TaskAttributes::hipri()),
          arg_(arg), op_(op), result_()
      {
        // Evaluate the expression
        std::shared_ptr<typename Exp::pmap_interface>
            pmap(new TiledArray::detail::BlockedPmap(arg_.get_world(), arg_.size()));
        madness::Future<bool> arg_eval = arg_.eval(arg_.vars(), pmap);

        // Add expression evaluation as a dependency for this task
        arg_eval.register_callback(this);
      }

      /// Result accessor

      /// \return A future for the result of this task
      const madness::Future<result_type>& result() const {
        return result_;
      }

      /// Task run function
      virtual void run(const madness::TaskThreadEnv&) {
        // Create reduce task object
        TiledArray::detail::ReduceTask<Op> reduce_task(arg_.get_world(), op_);

        // Spawn reduce tasks for each local tile.
        typename Exp::pmap_interface::const_iterator end = arg_.get_pmap()->end();
        typename Exp::pmap_interface::const_iterator it = arg_.get_pmap()->begin();
        if(arg_.is_dense()) {
          for(; it != end; ++it)
            reduce_task.add(arg_.move(*it));
        } else {
          for(; it != end; ++it)
            if(! arg_.is_zero(*it))
              reduce_task.add(arg_.move(*it));
        }

        // Set the result future
        result_.set(reduce_task.submit());
      }
    }; // class ReduceTiles

    /// This task will reduce the tiles of a pair of tensor expressions to a scalar value

    /// \tparam LExp The left tensor expression type
    /// \tparam RExp The left tensor expression type
    /// \tparam Op The reduction operation type
    template <typename LExp, typename RExp, typename Op>
    class ReduceTensorExpressionPair : public madness::TaskInterface {
    public:
      typedef typename Op::result_type result_type; ///< The result type

    private:
      LExp left_; ///< Left expression object
      RExp right_; ///< Right expression object
      Op op_; ///< The reduction operation
      madness::Future<result_type> result_; ///< The reduction result

    public:

      ReduceTensorExpressionPair(const LExp& left, const RExp& right, const Op& op) :
        madness::TaskInterface(2, madness::TaskAttributes::hipri()),
        left_(left), right_(right), op_(op), result_()
      {
        TA_ASSERT(left.trange() == right.trange());

        std::shared_ptr<typename LExp::pmap_interface>
          pmap(new TiledArray::detail::BlockedPmap(left_.get_world(), left_.size()));

        // Evaluate and wait the arguments
        madness::Future<bool> left_done = left_.eval(left_.vars(), pmap->clone());
        madness::Future<bool> right_done = right_.eval(left_.vars(), pmap);

        // Add expression evaluations as a dependencies for this task
        left_done.register_callback(this);
        right_done.register_callback(this);
      }

      /// Task result accessor

      /// \return A future to the reduction result
      madness::Future<result_type> result() {
        return result_;
      }

      /// Task run function
      virtual void run(const madness::TaskThreadEnv&) {
        TiledArray::detail::ReducePairTask<Op>
            reducer(left_.get_world(), op_);

        typename LExp::pmap_interface::const_iterator it = left_.get_pmap()->begin();
        const typename LExp::pmap_interface::const_iterator end = left_.get_pmap()->end();

        if(left_.is_dense() && right_.is_dense()) {
          // Evaluate tiles where both arguments and the result are dense
          for(; it != end; ++it) {
            const typename LExp::size_type i = *it;
            reducer.add(left_.move(i), right_.move(i));
          }
        } else {
          // Evaluate tiles where the result or one of the arguments is sparse
          for(; it != end; ++it) {
            const typename LExp::size_type i = *it;
            if(! left_.is_zero(i)) {
              madness::Future<typename LExp::value_type> left_tile = left_.move(i);
              if(! right_.is_zero(i))
                reducer.add(left_tile, right_.move(i));
            } else {
              if(! right_.is_zero(i))
                right_.move(i);
            }
          }
        }

        result_.set(reducer.submit());
      }
    }; // class ReduceTensorExpressionPair

  }  // namespace expressions
}  // namespace TieldArray

#endif // TILEDARRAY_TENSOR_REDUCE_H__INCLUDED
