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

#ifndef TILEDARRAY_UNARY_TENSOR_H__INCLUDED
#define TILEDARRAY_UNARY_TENSOR_H__INCLUDED

#include <TiledArray/dist_eval/dist_eval.h>

namespace TiledArray {
  namespace detail {

    /// Tensor that is composed from an argument tensor

    /// The tensor elements are constructed using a unary transformation
    /// operation.
    /// \tparam Arg The input distributed evaluator argument type
    /// \tparam Op The tile operation to be applied to the input tiles
    /// \tparam Policy The evaluator policy type
    template <typename Arg, typename Op, typename Policy>
    class UnaryEvalImpl : public DistEvalImpl<typename Arg::eval_type, Policy> {
    public:
      typedef UnaryEvalImpl<Arg, Op, Policy> UnaryEvalImpl_; ///< This object type
      typedef DistEvalImpl<typename Arg::eval_type, Policy> DistEvalImpl_; ///< The base class type
      typedef typename DistEvalImpl_::TensorImpl_ TensorImpl_; ///< The base, base class type
      typedef Arg arg_type; ///< The argument tensor type
      typedef typename DistEvalImpl_::size_type size_type; ///< Size type
      typedef typename DistEvalImpl_::range_type range_type; ///< Range type
      typedef typename DistEvalImpl_::shape_type shape_type; ///< Shape type
      typedef typename DistEvalImpl_::pmap_interface pmap_interface; ///< Process map interface type
      typedef typename DistEvalImpl_::trange_type trange_type; ///< Tiled range type
      typedef typename DistEvalImpl_::value_type value_type; ///< Tile type
      typedef typename DistEvalImpl_::eval_type eval_type; ///< Tile evaluation type
      typedef Op op_type; ///< Tile evaluation operator type

      /// Constructor

      /// \param arg The argument
      /// \param op The element transform operation
      UnaryEvalImpl(const arg_type& arg, const shape_type& shape,
          const std::shared_ptr<pmap_interface>& pmap, const Permutation& perm,
          const op_type& op) :
        DistEvalImpl_(arg.get_world(), arg.trange(), shape, pmap, perm),
        arg_(arg),
        op_(op)
      { }

      /// Virtual destructor
      virtual ~UnaryEvalImpl() { }

    private:

      /// Tile evaluation helper

      /// This function will:
      /// \li Apply the unary operation to the evaluated input tile
      /// \li Store the result tile
      /// \param i The tile index
      /// \param tile The tile to be evaluated
      void eval_tile_helper(const size_type i, typename op_type::argument_type tile) {
        // Apply unary operation to the evaluated input tile
        DistEvalImpl_::set_tile(i, op_(tile));
      }

      /// Tile evaluation helper

      /// This function will:
      /// \li Convert the input tile to the evaluation type
      /// \li Apply the unary operation to the evaluated input tile
      /// \li Store the result tile
      /// \tparam T The input tile type
      /// \param i The tile index
      /// \param tile The tile to be evaluated
      /// \note This function will only be instantiated when the input tile is
      /// a lazy tile that requires on-the-fly evaluation.
      template <typename T>
      typename madness::disable_if<std::is_same<T, typename arg_type::eval_type> >::type
      eval_tile_helper(const size_type i, const T& tile) {
        // Evaluate tile to an l-value so that it may be consumed by op_.
        typename arg_type::eval_type eval_tile = tile;

        // Apply unary operation to the evaluated input tile
        eval_tile_helper(i, eval_tile);
      }

      /// Input tile argument type

      /// The argument must be a non-const reference if the input tile is
      /// a consumable resource, otherwise a const reference is sufficient.
      typedef typename madness::if_<std::is_const<typename op_type::argument_type>,
          const typename arg_type::value_type&,
                typename arg_type::value_type&>::type
              tile_argument_type;

      /// Task function for evaluating tiles

      /// \param i The tile index
      /// \param tile The tile to be evaluated
      void eval_tile(const size_type i, tile_argument_type tile) {
        eval_tile_helper(i, tile);
      }

      /// Function for evaluating this tensor's tiles

      /// This function is run inside a task, and will run after \c eval_children
      /// has completed. It should spawn additional tasks that evaluate the
      /// individual result tiles.
      /// \return The number of local tiles
      virtual size_type eval_tiles(const std::shared_ptr<DistEvalImpl_>& pimpl) {
        // Counter for the number of tasks submitted by this object
        size_type task_count = 0ul;

        // Convert pimpl to this object type so it can be used in tasks
        TA_ASSERT(this == pimpl.get());
        std::shared_ptr<UnaryEvalImpl_> self =
            std::static_pointer_cast<UnaryEvalImpl_>(pimpl);

        // Make sure all local tiles are present.
        const typename pmap_interface::const_iterator end = TensorImpl_::pmap()->end();
        typename pmap_interface::const_iterator it = TensorImpl_::pmap()->begin();
        for(; it != end; ++it) {
          if(! arg_.is_zero(*it)) {
            TensorImpl_::get_world().taskq.add(self, & UnaryEvalImpl_::eval_tile,
                DistEvalImpl_::perm_index(*it), arg_.move(*it));
            ++task_count;
          }
        }

        return task_count;
      }

      /// Function for evaluating child tensors
      virtual void eval_children() { arg_.eval(); }

      /// Wait for tasks of children to finish
      virtual void wait_children() const { arg_.wait(); }

      arg_type arg_; ///< Argument
      op_type op_; ///< The unary tile operation
    }; // class UnaryEvalImpl

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_UNARY_TENSOR_H__INCLUDED
