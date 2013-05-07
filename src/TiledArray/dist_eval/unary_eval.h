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

#include <TiledArray/tensor_expression.h>
#include <TiledArray/tensor.h>

namespace TiledArray {
  namespace detail {


    /// Tensor that is composed from an argument tensor

    /// The tensor elements are constructed using a unary transformation
    /// operation.
    /// \tparam Arg The argument type
    /// \tparam Op The Unary transform operator type.
    template <typename Exp, typename Op>
    class UnaryEvalImpl : public DistEvalImpl<typename Op::result_type> {
    public:
      typedef UnaryEvalImpl<Exp, Op> UnaryEvalImpl_; ///< This object type
      typedef Exp arg_type; ///< The argument tensor type
      typedef DistEvalImpl<typename Op::result_type> DistEvalImpl_; ///< The base class type
      typedef typename DistEvalImpl_::TensorImpl_ TensorImpl_; ///< The base, base class type
      typedef typename DistEvalImpl_::size_type size_type; ///< Size type
      typedef typename DistEvalImpl_::range_type range_type; ///< Range type
      typedef typename DistEvalImpl_::shape_type shape_type; ///< Shape type
      typedef typename DistEvalImpl_::pmap_interface pmap_interface; ///< Process map interface type
      typedef typename DistEvalImpl_::trange_type trange_type; ///< tiled range type
      typedef typename DistEvalImpl_::value_type value_type; ///< value type
      typedef typename DistEvalImpl_::const_reference const_reference; ///< const reference type
      typedef typename DistEvalImpl_::const_iterator const_iterator; ///< const iterator type


    public:

      /// Constructor

      /// \param arg The argument
      /// \param op The element transform operation
      UnaryEvalImpl(const arg_type& arg, const Permutation& perm, const shape_type& shape,
          const std::shared_ptr<pmap_interface>& pmap, const Op& op) :
        DistEvalImpl_(arg.get_world(), perm, arg.trange(), shape, pmap),
        arg_(arg),
        op_(op)
      { }

      /// Virtual destructor
      virtual ~UnaryEvalImpl() { }

    private:

      void eval_tile(const size_type i, const typename arg_type::value_type& tile,
          madness::AtomicInt* const counter)
      {
        DistEvalImpl_::set(i, op_(tile));
        (*counter)++;
      }

      /// Function for evaluating this tensor's tiles

      /// This function is run inside a task, and will run after \c eval_children
      /// has completed. It should spawn additional tasks that evaluate the
      /// individual result tiles.
      virtual void eval_tiles(const std::shared_ptr<DistEvalImpl>& pimpl,
          madness::AtomicInt& counter, int& task_count)
      {
        // Convert pimpl to this object type so it can be used in tasks
        TA_ASSERT(this == pimpl.get());
        std::shared_ptr<UnaryEvalImpl_> this_pimpl =
            std::static_pointer_cast<UnaryEvalImpl_>(pimpl);

        // Make sure all local tiles are present.
        const typename pmap_interface::const_iterator end = TensorImpl_::pmap()->end();
        typename pmap_interface::const_iterator it = TensorImpl_::pmap()->begin();
        if(arg_.is_dense()) {
          for(; it != end; ++it) {
            TensorImpl_::get_world().taskq.add(this_pimpl,
                & UnaryEvalImpl_::eval_tile, *it, arg_.move(*it), counter);
            ++task_count;
          }
        } else {
          for(; it != end; ++it) {
            if(! arg_.is_zero(*it)) {
              TensorImpl_::get_world().taskq.add(this_pimpl,
                  & UnaryEvalImpl_::eval_tile, *it, arg_.move(*it), counter);
              ++task_count;
            }
          }
        }

        arg_.release();
      }

      /// Function for evaluating child tensors

      /// This function should return true when the child

      /// This function should evaluate all child tensors.
      /// \param vars The variable list for this tensor (may be different from
      /// the variable list used to initialize this tensor).
      /// \param pmap The process map for this tensor
      virtual void eval_children(madness::AtomicInt& counter, int& task_count) {
        arg_.eval(counter, task_count);
      }

      arg_type arg_; ///< Argument
      Op op_; ///< The unary tile operation
    }; // class UnaryEvalImpl

  }  // namespace detail
}  // namespace TiledArray

#endif // TILEDARRAY_UNARY_TENSOR_H__INCLUDED
