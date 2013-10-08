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
    class UnaryEvalImpl : public DistEvalImpl<typename Op::result_type, Policy> {
    public:
      typedef UnaryEvalImpl<Arg, Op, Policy> UnaryEvalImpl_; ///< This object type
      typedef DistEvalImpl<typename Op::result_type, Policy> DistEvalImpl_; ///< The base class type
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
      UnaryEvalImpl(const arg_type& arg, madness::World& world,
          const shape_type& shape, const std::shared_ptr<pmap_interface>& pmap,
          const Permutation& perm, const op_type& op) :
        DistEvalImpl_(world, arg.trange(), shape, pmap, perm),
        arg_(arg),
        op_(op)
      { }

      /// Virtual destructor
      virtual ~UnaryEvalImpl() { }

    private:

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
        DistEvalImpl_::set_tile(i, op_(tile));
      }

      /// Function for evaluating this tensor's tiles

      /// This function is run inside a task, and will run after \c eval_children
      /// has completed. It should spawn additional tasks that evaluate the
      /// individual result tiles.
      /// \return The number of local tiles
      virtual size_type internal_eval(const std::shared_ptr<DistEvalImpl_>& pimpl) {
        // Convert pimpl to this object type so it can be used in tasks
        std::shared_ptr<UnaryEvalImpl_> self =
            std::static_pointer_cast<UnaryEvalImpl_>(pimpl);

        // Evaluate argument
        arg_.eval();

        // Counter for the number of tasks submitted by this object
        size_type task_count = 0ul;

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

        // Wait for local tiles of argument to be evaluated
        arg_.wait();

        return task_count;
      }

      arg_type arg_; ///< Argument
      op_type op_; ///< The unary tile operation
    }; // class UnaryEvalImpl

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
    template <typename Tile, typename Policy, typename Op>
    DistEval<typename Op::result_type, Policy> make_unary_eval(
        const DistEval<Tile, Policy>& arg,
        madness::World& world,
        const typename DistEval<Tile, Policy>::shape_type& shape,
        const std::shared_ptr<typename DistEval<Tile, Policy>::pmap_interface>& pmap,
        const Permutation& perm,
        const Op& op)
    {
      typedef UnaryEvalImpl<DistEval<Tile, Policy>, Op, Policy> impl_type;
      typedef typename impl_type::DistEvalImpl_ impl_base_type;
      return DistEval<typename Op::result_type, Policy>(
          std::shared_ptr<impl_base_type>(new impl_type(arg, world, shape, pmap, perm, op)));
    }

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_UNARY_TENSOR_H__INCLUDED
