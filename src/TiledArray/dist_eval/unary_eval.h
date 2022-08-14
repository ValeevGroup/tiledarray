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

#ifndef TILEDARRAY_DIST_EVAL_UNARY_EVAL_H__INCLUDED
#define TILEDARRAY_DIST_EVAL_UNARY_EVAL_H__INCLUDED

#include <TiledArray/dist_eval/dist_eval.h>

#include <TiledArray/tensor/type_traits.h>

namespace TiledArray {
namespace detail {

/// Tensor that is composed from an argument tensor

/// The tensor elements are constructed using a unary transformation
/// operation.
/// \tparam Arg The input distributed evaluator argument type
/// \tparam Op The tile operation to be applied to the input tiles
/// \tparam Policy The evaluator policy type
template <typename Arg, typename Op, typename Policy>
class UnaryEvalImpl
    : public DistEvalImpl<typename Op::result_type, Policy>,
      public std::enable_shared_from_this<UnaryEvalImpl<Arg, Op, Policy>> {
 public:
  typedef UnaryEvalImpl<Arg, Op, Policy> UnaryEvalImpl_;  ///< This object type
  typedef DistEvalImpl<typename Op::result_type, Policy>
      DistEvalImpl_;  ///< The base class type
  typedef typename DistEvalImpl_::TensorImpl_
      TensorImpl_;       ///< The base, base class type
  typedef Arg arg_type;  ///< The argument tensor type
  typedef typename DistEvalImpl_::ordinal_type ordinal_type;  ///< Ordinal type
  typedef typename DistEvalImpl_::range_type range_type;      ///< Range type
  typedef typename DistEvalImpl_::shape_type shape_type;      ///< Shape type
  typedef typename DistEvalImpl_::pmap_interface
      pmap_interface;  ///< Process map interface type
  typedef
      typename DistEvalImpl_::trange_type trange_type;    ///< Tiled range type
  typedef typename DistEvalImpl_::value_type value_type;  ///< Tile type
  typedef
      typename DistEvalImpl_::eval_type eval_type;  ///< Tile evaluation type
  typedef Op op_type;  ///< Tile evaluation operator type

  /// Constructor

  /// \param arg The argument
  /// \param world The world where the tensor lives
  /// \param trange The tiled range object
  /// \param shape The tensor shape object
  /// \param pmap The tile-process map
  /// \param perm The permutation that is applied to tile indices
  /// \param op The tile transform operation
  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  UnaryEvalImpl(const arg_type& arg, World& world, const trange_type& trange,
                const shape_type& shape,
                const std::shared_ptr<const pmap_interface>& pmap,
                const Perm& perm, const op_type& op)
      : DistEvalImpl_(world, trange, shape, pmap, outer(perm)),
        arg_(arg),
        op_(op) {}

  /// Virtual destructor
  virtual ~UnaryEvalImpl() {}

  /// Get tile at index \c i

  /// \param i The index of the tile
  /// \return A \c Future to the tile at index i
  /// \throw TiledArray::Exception When tile \c i is owned by a remote node.
  /// \throw TiledArray::Exception When tile \c i a zero tile.
  virtual Future<value_type> get_tile(ordinal_type i) const {
    TA_ASSERT(TensorImpl_::is_local(i));
    TA_ASSERT(!TensorImpl_::is_zero(i));
    const auto source = arg_.owner(DistEvalImpl_::perm_index_to_source(i));
    const madness::DistributedID key(DistEvalImpl_::id(), i);
    return TensorImpl_::world().gop.template recv<value_type>(source, key);
  }

  /// Discard a tile that is not needed

  /// This function handles the cleanup for tiles that are not needed in
  /// subsequent computation.
  /// \param i The index of the tile
  virtual void discard_tile(ordinal_type i) const { get_tile(i); }

 private:
  /// Input tile argument type

  /// The argument must be a non-const reference if the input tile is
  /// a consumable resource, otherwise a const reference is sufficient.
  typedef typename std::conditional<
      op_type::is_consumable, typename arg_type::value_type&,
      const typename arg_type::value_type&>::type tile_argument_type;

  /// Task function for evaluating tiles

#ifdef TILEDARRAY_HAS_CUDA
  /// \param i The tile index
  /// \param tile The tile to be evaluated
  template <typename U = value_type>
  std::enable_if_t<detail::is_cuda_tile_v<U>, void> eval_tile(
      const ordinal_type i, tile_argument_type tile) {
    // TODO avoid copy Op object
    auto result_tile =
        madness::add_cuda_task(DistEvalImpl_::world(), op_, tile);
    DistEvalImpl_::set_tile(i, result_tile);
  }

  /// \param i The tile index
  /// \param tile The tile to be evaluated
  template <typename U = value_type>
  std::enable_if_t<!detail::is_cuda_tile_v<U>, void> eval_tile(
      const ordinal_type i, tile_argument_type tile) {
    DistEvalImpl_::set_tile(i, op_(tile));
  }
#else
  /// \param i The tile index
  /// \param tile The tile to be evaluated
  void eval_tile(const ordinal_type i, tile_argument_type tile) {
    DistEvalImpl_::set_tile(i, op_(tile));
  }
#endif
  /// Evaluate the tiles of this tensor

  /// This function will evaluate the children of this distributed evaluator
  /// and evaluate the tiles for this distributed evaluator. It will block
  /// until the tasks for the children are evaluated (not for the tasks of
  /// this object).
  /// \return The number of tiles that will be set by this process
  virtual int internal_eval() {
    // Convert pimpl to this object type so it can be used in tasks
    std::shared_ptr<UnaryEvalImpl_> self =
        std::enable_shared_from_this<UnaryEvalImpl_>::shared_from_this();

    // Evaluate argument
    arg_.eval();

    // Counter for the number of tasks submitted by this object
    ordinal_type task_count = 0ul;

    // Make sure all local tiles are present.
    const typename pmap_interface::const_iterator end = arg_.pmap()->end();
    typename pmap_interface::const_iterator it = arg_.pmap()->begin();
    for (; it != end; ++it) {
      // Get argument tile index
      const auto index = *it;

      if (!arg_.is_zero(index)) {
        // Get target tile index
        const auto target_index = DistEvalImpl_::perm_index_to_target(index);

        // Schedule tile evaluation task
#ifdef TILEDARRAY_HAS_CUDA
        TensorImpl_::world().taskq.add(self,
                                       &UnaryEvalImpl_::template eval_tile<>,
                                       target_index, arg_.get(index));
#else
        TensorImpl_::world().taskq.add(self, &UnaryEvalImpl_::eval_tile,
                                       target_index, arg_.get(index));
#endif

        ++task_count;
      }
    }

    // Wait for local tiles of argument to be evaluated
    arg_.wait();

    return task_count;
  }

  arg_type arg_;  ///< Argument
  op_type op_;    ///< The unary tile operation
};                // class UnaryEvalImpl

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_DIST_EVAL_UNARY_EVAL_H__INCLUDED
