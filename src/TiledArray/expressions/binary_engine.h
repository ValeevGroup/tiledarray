/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  binary_engine.h
 *  Mar 31, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_BINARY_ENGINE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_BINARY_ENGINE_H__INCLUDED

#include <TiledArray/dist_eval/binary_eval.h>
#include <TiledArray/expressions/expr_engine.h>
#include <TiledArray/expressions/permopt.h>

namespace TiledArray {
namespace expressions {

// Forward declarations
template <typename>
class BinaryExpr;
template <typename>
class BinaryEngine;

template <typename Derived>
class BinaryEngine : public ExprEngine<Derived> {
 public:
  // Class hierarchy typedefs
  typedef BinaryEngine<Derived> BinaryEngine_;  ///< This class type
  typedef ExprEngine<Derived> ExprEngine_;      ///< Base class type

  // Argument typedefs
  typedef typename EngineTrait<Derived>::left_type
      left_type;  ///< The left-hand expression type
  typedef typename EngineTrait<Derived>::right_type
      right_type;  ///< The right-hand expression type

  // Operational typedefs
  typedef typename EngineTrait<Derived>::value_type
      value_type;  ///< The result tile type
  typedef typename EngineTrait<Derived>::op_type
      op_type;  ///< The tile operation type
  typedef
      typename EngineTrait<Derived>::policy policy;  ///< The result policy type
  typedef typename EngineTrait<Derived>::dist_eval_type
      dist_eval_type;  ///< The distributed evaluator type

  // Meta data typedefs
  typedef typename EngineTrait<Derived>::size_type size_type;  ///< Size type
  typedef typename EngineTrait<Derived>::trange_type
      trange_type;  ///< Tiled range type
  typedef typename EngineTrait<Derived>::shape_type shape_type;  ///< Shape type
  typedef typename EngineTrait<Derived>::pmap_interface
      pmap_interface;  ///< Process map interface type

  static constexpr bool consumable = EngineTrait<Derived>::consumable;
  static constexpr unsigned int leaves = EngineTrait<Derived>::leaves;

 protected:
  // Import base class variables to this scope
  using ExprEngine_::indices_;
  using ExprEngine_::perm_;
  using ExprEngine_::permute_tiles_;
  using ExprEngine_::pmap_;
  using ExprEngine_::shape_;
  using ExprEngine_::trange_;
  using ExprEngine_::world_;

  left_type left_;                    ///< The left-hand argument
  right_type right_;                  ///< The right-hand argument
  BipartiteIndexList left_indices_;   ///< Target left-hand index list
  BipartiteIndexList right_indices_;  ///< Target right-hand index list
  PermutationType left_outer_permtype_ =
      PermutationType::general;  ///< Left-hand permutation type
  PermutationType right_outer_permtype_ =
      PermutationType::general;  ///< Right-hand permutation type
  PermutationType left_inner_permtype_ =
      PermutationType::general;  ///< Left-hand permutation type
  PermutationType right_inner_permtype_ =
      PermutationType::general;  ///< Right-hand permutation type

  template <TensorProduct ProductType>
  void init_indices_(const BipartiteIndexList& target_indices = {}) {
    static_assert(ProductType == TensorProduct::Contraction ||
                  ProductType == TensorProduct::Hadamard);
    // prefer to permute the arg with fewest leaves to try to minimize the
    // number of possible permutations
    using permopt_type =
        std::conditional_t<ProductType == TensorProduct::Contraction,
                           GEMMPermutationOptimizer,
                           HadamardPermutationOptimizer>;

    std::shared_ptr<BinaryOpPermutationOptimizer> outer_opt, inner_opt;
    if (!target_indices) {
      outer_opt = std::make_shared<permopt_type>(
          outer(left_.indices()), outer(right_.indices()),
          left_type::leaves <= right_type::leaves);
      inner_opt = make_permutation_optimizer(
          inner(left_.indices()), inner(right_.indices()),
          left_type::leaves <= right_type::leaves);
    } else {
      outer_opt = std::make_shared<permopt_type>(
          outer(target_indices), outer(left_.indices()),
          outer(right_.indices()), left_type::leaves <= right_type::leaves);
      inner_opt = make_permutation_optimizer(
          inner(target_indices), inner(left_.indices()),
          inner(right_.indices()), left_type::leaves <= right_type::leaves);
    }

    left_indices_ = BipartiteIndexList(outer_opt->target_left_indices(),
                                       inner_opt->target_left_indices());
    right_indices_ = BipartiteIndexList(outer_opt->target_right_indices(),
                                        inner_opt->target_right_indices());
    indices_ = BipartiteIndexList(outer_opt->target_result_indices(),
                                  inner_opt->target_result_indices());

    left_outer_permtype_ = outer_opt->left_permtype();
    right_outer_permtype_ = outer_opt->right_permtype();
    left_inner_permtype_ = inner_opt->left_permtype();
    right_inner_permtype_ = inner_opt->right_permtype();

    // Here we set the type of permutation that will be applied to the
    // argument tensors. If both arguments are plain tensors
    // (tensors-of-scalars) and their permutations can be fused into GEMM,
    // disable their permutation
    using left_tile_type = typename EngineTrait<left_type>::eval_type;
    using right_tile_type = typename EngineTrait<right_type>::eval_type;
    constexpr bool left_tile_is_tot =
        TiledArray::detail::is_tensor_of_tensor_v<left_tile_type>;
    constexpr bool right_tile_is_tot =
        TiledArray::detail::is_tensor_of_tensor_v<right_tile_type>;
    static_assert(!(left_tile_is_tot ^ right_tile_is_tot),
                  "ContEngine can only handle tensors of same nested-ness "
                  "(both plain or both ToT)");
    constexpr bool args_are_plain_tensors =
        !left_tile_is_tot && !right_tile_is_tot;
    if (args_are_plain_tensors &&
        (left_outer_permtype_ == PermutationType::matrix_transpose ||
         left_outer_permtype_ == PermutationType::identity)) {
      left_.permute_tiles(false);
    }
    if (!args_are_plain_tensors &&
        (left_inner_permtype_ == PermutationType::matrix_transpose ||
         left_inner_permtype_ == PermutationType::identity)) {
      left_.permute_tiles(false);
    }
    if (args_are_plain_tensors &&
        (right_outer_permtype_ == PermutationType::matrix_transpose ||
         right_outer_permtype_ == PermutationType::identity)) {
      right_.permute_tiles(false);
    }
    if (!args_are_plain_tensors &&
        (right_inner_permtype_ == PermutationType::matrix_transpose ||
         right_inner_permtype_ == PermutationType::identity)) {
      right_.permute_tiles(false);
    }
  }

 public:
  template <typename D>
  BinaryEngine(const BinaryExpr<D>& expr)
      : ExprEngine_(expr), left_(expr.left()), right_(expr.right()) {}

  /// Set the index list for this expression

  /// This function will set the index list for this expression and its
  /// children such that the number of permutations is minimized. The final
  /// index list may not be set to target, which indicates that the
  /// result of this expression will be permuted to match \c target_indices.
  /// \param target_indices The target index list for this expression
  void perm_indices(const BipartiteIndexList& target_indices) {
    if (permute_tiles_) {
      TA_ASSERT(left_.indices().size() == target_indices.size());
      TA_ASSERT(right_.indices().size() == target_indices.size());

      init_indices_<TensorProduct::Hadamard>(target_indices);

      TA_ASSERT(right_outer_permtype_ == PermutationType::general ||
                right_inner_permtype_ == PermutationType::general);

      if (left_.indices() != left_indices_) left_.init_indices(left_indices_);
      if (right_.indices() != right_indices_)
        right_.init_indices(right_indices_);
    }
  }

  /// Initialize the index list of this expression

  /// \param target_indices The target index list for this expression
  void init_indices(const BipartiteIndexList& target_indices) {
    left_.init_indices(target_indices);
    right_.init_indices(target_indices);
    perm_indices(target_indices);
  }

  /// Initialize the index list of this expression
  void init_indices(bool children_initialized = false) {
    if (!children_initialized) {
      left_.init_indices();
      right_.init_indices();
    }

    init_indices_<TensorProduct::Hadamard>();
    TA_ASSERT(right_outer_permtype_ == PermutationType::general ||
              right_inner_permtype_ == PermutationType::general);

    if (left_.indices() != left_indices_) left_.init_indices(left_indices_);
    if (right_.indices() != right_indices_) right_.init_indices(right_indices_);
  }

  /// Initialize result tensor structure

  /// This function will initialize the permutation, tiled range, and shape
  /// for the left-hand, right-hand, and result tensor.
  /// \param target_indices The target index list for the result tensor
  void init_struct(const BipartiteIndexList& target_indices) {
    left_.init_struct(left_indices_);
    right_.init_struct(right_indices_);
#ifndef NDEBUG
    if (left_.trange() != right_.trange()) {
      if (TiledArray::get_default_world().rank() == 0) {
        TA_USER_ERROR_MESSAGE(
            "The TiledRanges of the left- and right-hand arguments of the "
            "binary operation are not equal:"
            << "\n    left  = " << left_.trange()
            << "\n    right = " << right_.trange());
      }

      TA_EXCEPTION(
          "The TiledRanges of the left- and right-hand arguments "
          "of the binary operation are not equal.");
    }
#endif  // NDEBUG
    ExprEngine_::init_struct(target_indices);
  }

  /// Initialize result tensor distribution

  /// This function will initialize the world and process map for the result
  /// tensor.
  /// \param world The world were the result will be distributed
  /// \param pmap The process map for the result tensor tiles
  void init_distribution(World* world,
                         const std::shared_ptr<pmap_interface>& pmap) {
    left_.init_distribution(world, pmap);
    right_.init_distribution(world, left_.pmap());
    ExprEngine_::init_distribution(world, left_.pmap());
  }

  /// Non-permuting tiled range factory function

  /// \return The result tiled range
  trange_type make_trange() const { return left_.trange(); }

  /// Permuting tiled range factory function

  /// \param perm The permutation to be applied to the tiled range
  /// \return The result shape
  trange_type make_trange(const Permutation& perm) const {
    return perm * left_.trange();
  }

  /// Construct the distributed evaluator for this expression

  /// \return The distributed evaluator that will evaluate this expression
  dist_eval_type make_dist_eval() const {
    typedef TiledArray::detail::BinaryEvalImpl<
        typename left_type::dist_eval_type, typename right_type::dist_eval_type,
        op_type, policy>
        impl_type;

    // Construct left and right distributed evaluators
    const typename left_type::dist_eval_type left = left_.make_dist_eval();
    const typename right_type::dist_eval_type right = right_.make_dist_eval();

    // Construct the distributed evaluator type
    std::shared_ptr<impl_type> pimpl =
        std::make_shared<impl_type>(left, right, *world_, trange_, shape_,
                                    pmap_, perm_, this->derived().make_op());

    return dist_eval_type(pimpl);
  }

  /// Expression print

  /// \param os The output stream
  /// \param target_indices The target index list for this expression
  void print(ExprOStream os, const BipartiteIndexList& target_indices) const {
    ExprEngine_::print(os, target_indices);
    os.inc();
    left_.print(os, indices_);
    right_.print(os, indices_);
    os.dec();
  }
};  // class BinaryEngine

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_BINARY_ENGINE_H__INCLUDED
