/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
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
 *  dot_inner_engine.h
 *  Jun 15, 2026
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_DOT_INNER_ENGINE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_DOT_INNER_ENGINE_H__INCLUDED

#include <TiledArray/expressions/cont_engine.h>
#include <TiledArray/tile_op/binary_wrapper.h>
#include <TiledArray/tile_op/mult.h>

namespace TiledArray {
namespace expressions {

// Forward declarations
template <typename, typename>
class DotInnerExpr;
template <typename, typename, typename>
class DotInnerEngine;

template <typename Left, typename Right, typename Result>
struct EngineTrait<DotInnerEngine<Left, Right, Result>> {
  static_assert(
      std::is_same<typename EngineTrait<Left>::policy,
                   typename EngineTrait<Right>::policy>::value,
      "The left- and right-hand expressions must use the same policy class");

  // Argument typedefs
  typedef Left left_type;    ///< The left-hand expression type
  typedef Right right_type;  ///< The right-hand expression type

  // Operational typedefs
  // The result tile is a PLAIN tensor of scalars (denest): contracting the
  // (nested) inner modes of two ToT operands to a scalar per outer cell.
  // The outer regime decides which tile op is used: outer Hadamard rides the
  // Mult binary tile op below (via BinaryEngine), while outer Contraction /
  // General use ContEngine's own ContractReduce op_type. Both carry the inner
  // scalar dot via the element op installed in cont_engine.
  typedef Result value_type;  ///< The result tile type
  typedef TiledArray::detail::Mult<
      Result, typename EngineTrait<Left>::eval_type,
      typename EngineTrait<Right>::eval_type, EngineTrait<Left>::consumable,
      EngineTrait<Right>::consumable>
      op_base_type;  ///< The base tile operation type
  typedef TiledArray::detail::BinaryWrapper<op_base_type>
      op_type;  ///< The tile operation type
  typedef typename eval_trait<value_type>::type
      eval_type;  ///< Evaluation tile type
  typedef typename TiledArray::detail::numeric_type<value_type>::type
      scalar_type;                       ///< Tile scalar type
  typedef typename Left::policy policy;  ///< The result policy type
  typedef TiledArray::detail::DistEval<value_type, policy>
      dist_eval_type;  ///< The distributed evaluator type

  // Meta data typedefs
  typedef typename policy::ordinal_type size_type;   ///< Size type
  typedef typename policy::trange_type trange_type;  ///< Tiled range type
  typedef typename policy::shape_type shape_type;    ///< Shape type
  typedef typename policy::pmap_interface
      pmap_interface;  ///< Process map interface type

  static constexpr bool consumable = is_consumable_tile<eval_type>::value;
  static constexpr unsigned int leaves =
      EngineTrait<Left>::leaves + EngineTrait<Right>::leaves;
};

/// Inner-dot expression engine

/// Computes, for two tensor-of-tensor (ToT) operands, a per-outer-cell inner
/// dot product over a general outer product, producing a plain tensor (T)
/// result. The outer modes are combined as for an ordinary product (Hadamard,
/// contraction, or general); the inner modes are fully contracted (dotted) to a
/// scalar. Mirrors \c MultEngine but with a plain (denested) result tile type.
/// \tparam Left The left-hand engine type
/// \tparam Right The right-hand engine type
/// \tparam Result The (plain) result tile type
template <typename Left, typename Right, typename Result>
class DotInnerEngine : public ContEngine<DotInnerEngine<Left, Right, Result>> {
 public:
  // Class hierarchy typedefs
  typedef DotInnerEngine<Left, Right, Result>
      DotInnerEngine_;  ///< This class type
  typedef ContEngine<DotInnerEngine_>
      ContEngine_;  ///< Contraction engine base class
  typedef BinaryEngine<DotInnerEngine_>
      BinaryEngine_;  ///< Binary base class type
  typedef BinaryEngine<DotInnerEngine_>
      ExprEngine_;  ///< Expression engine base class type

  // Argument typedefs
  typedef typename EngineTrait<DotInnerEngine_>::left_type
      left_type;  ///< The left-hand expression type
  typedef typename EngineTrait<DotInnerEngine_>::right_type
      right_type;  ///< The right-hand expression type

  // Operational typedefs
  typedef typename EngineTrait<DotInnerEngine_>::value_type
      value_type;  ///< The result tile type
  typedef typename EngineTrait<DotInnerEngine_>::op_base_type
      op_base_type;  ///< The tile operation type
  typedef typename EngineTrait<DotInnerEngine_>::op_type
      op_type;  ///< The tile operation type
  typedef typename EngineTrait<DotInnerEngine_>::policy
      policy;  ///< The result policy type
  typedef typename EngineTrait<DotInnerEngine_>::dist_eval_type
      dist_eval_type;  ///< The distributed evaluator type
  typedef typename EngineTrait<DotInnerEngine_>::scalar_type
      scalar_type;  ///< Tile scalar type

  // Meta data typedefs
  typedef typename EngineTrait<DotInnerEngine_>::size_type
      size_type;  ///< Size type
  typedef typename EngineTrait<DotInnerEngine_>::trange_type
      trange_type;  ///< Tiled range type
  typedef typename EngineTrait<DotInnerEngine_>::shape_type
      shape_type;  ///< Shape type
  typedef typename EngineTrait<DotInnerEngine_>::pmap_interface
      pmap_interface;  ///< Process map interface type

  /// Tag consumed by ContEngine: this engine denests two nested operands to a
  /// scalar-element result (the inner modes are dotted away).
  static constexpr bool denest_to_scalar = true;

 public:
  /// Constructor

  /// \tparam L The left-hand argument expression type
  /// \tparam R The right-hand argument expression type
  /// \param expr The parent expression
  template <typename L, typename R>
  DotInnerEngine(const DotInnerExpr<L, R>& expr) : ContEngine_(expr) {}

  /// Set the index list for this expression

  /// \param target_indices The target index list for this expression
  void perm_indices(const BipartiteIndexList& target_indices) {
    if (this->product_type() == TensorProduct::Contraction)
      ContEngine_::perm_indices(target_indices);
    else if (this->product_type() == TensorProduct::General) {
      if (!this->implicit_permute()) {
        BinaryEngine_::template init_indices_<TensorProduct::General>(
            target_indices);
        if (BinaryEngine_::left_indices_ != BinaryEngine_::left_.indices())
          BinaryEngine_::left_.perm_indices(BinaryEngine_::left_indices_);
        if (BinaryEngine_::right_indices_ != BinaryEngine_::right_.indices())
          BinaryEngine_::right_.perm_indices(BinaryEngine_::right_indices_);
      }
    } else {
      BinaryEngine_::perm_indices(target_indices);
    }
  }

  /// Initialize the index list of this expression

  /// \param target_indices The target index list for this expression
  void init_indices(const BipartiteIndexList& target_indices) {
    BinaryEngine_::init_children_indices(target_indices);

    this->product_type_ = compute_product_type(
        outer(BinaryEngine_::left_.indices()),
        outer(BinaryEngine_::right_.indices()), outer(target_indices));
    // the inner modes are fully contracted (dotted) to a scalar; the result
    // carries no inner indices, so the inner product is a Contraction
    this->inner_product_type_ = compute_product_type(
        inner(BinaryEngine_::left_.indices()),
        inner(BinaryEngine_::right_.indices()), IndexList{});

    if (this->product_type() == TensorProduct::Hadamard) {
      BinaryEngine_::perm_indices(target_indices);
    } else if (this->product_type() == TensorProduct::General) {
      this->perm_indices(target_indices);
    } else {
      auto children_initialized = true;
      ContEngine_::init_indices(children_initialized);
      ContEngine_::perm_indices(target_indices);
    }
  }

  /// Initialize the index list of this expression
  void init_indices() {
    BinaryEngine_::left_.init_indices();
    BinaryEngine_::right_.init_indices();
    auto children_initialized = true;
    this->product_type_ =
        compute_product_type(outer(BinaryEngine_::left_.indices()),
                             outer(BinaryEngine_::right_.indices()));
    this->inner_product_type_ =
        compute_product_type(inner(BinaryEngine_::left_.indices()),
                             inner(BinaryEngine_::right_.indices()));

    if (this->product_type() == TensorProduct::Hadamard) {
      BinaryEngine_::init_indices(children_initialized);
    } else {
      ContEngine_::init_indices(children_initialized);
    }
  }

  /// Initialize result tensor structure

  /// \param target_indices The target index list for the result tensor
  void init_struct(const BipartiteIndexList& target_indices) {
    if (this->product_type() == TensorProduct::General) {
      this->init_inner_tile_op(inner(target_indices));
      ContEngine_::init_struct_general(target_indices);
      return;
    }

    this->init_perm(target_indices);

    // build the inner (scalar dot) element op before ContEngine_::init_struct,
    // which threads element_nonreturn_op_ into the outer ContractReduce
    this->init_inner_tile_op(inner(target_indices));
    if (this->product_type() == TensorProduct::Contraction)
      ContEngine_::init_struct(target_indices);
    else
      BinaryEngine_::init_struct(target_indices);
  }

  /// Initialize result tensor distribution

  /// \param world The world were the result will be distributed
  /// \param pmap The process map for the result tensor tiles
  void init_distribution(World* world,
                         std::shared_ptr<const pmap_interface> pmap) {
    if (this->product_type() == TensorProduct::Contraction)
      ContEngine_::init_distribution(world, pmap);
    else if (this->product_type() == TensorProduct::General)
      ContEngine_::init_distribution_general(world, pmap);
    else
      BinaryEngine_::init_distribution(world, pmap);
  }

  /// Non-permuting tiled range factory function

  /// \return The result tiled range object
  trange_type make_trange() const {
    if (this->product_type() == TensorProduct::Contraction)
      return ContEngine_::make_trange();
    else if (this->product_type() == TensorProduct::General)
      return ContEngine_::make_trange_general();
    else
      return BinaryEngine_::make_trange();
  }

  /// Permuting tiled range factory function

  /// \param perm The permutation to be applied to the array
  /// \return The result tiled range object
  trange_type make_trange(const Permutation& perm) const {
    if (this->product_type() == TensorProduct::Contraction)
      return ContEngine_::make_trange(perm);
    else
      return BinaryEngine_::make_trange(perm);
  }

  /// Non-permuting shape factory function

  /// \return The result shape
  shape_type make_shape() const {
    return BinaryEngine_::left_.shape().mult(BinaryEngine_::right_.shape());
  }

  /// Permuting shape factory function

  /// \param perm The permutation to be applied to the array
  /// \return The result shape
  shape_type make_shape(const Permutation& perm) const {
    return BinaryEngine_::left_.shape().mult(BinaryEngine_::right_.shape(),
                                             outer(perm));
  }

  /// Non-permuting tile operation factory function

  /// \return The tile operation
  /// \note Only used on the outer-Hadamard path (Contraction / General route
  /// through ContEngine's own ContractReduce op). The Mult binary tile op maps
  /// the value-returning inner dot (element_return_op_) over the outer cells.
  op_type make_tile_op() const {
    return op_type(op_base_type(this->element_return_op_));
  }

  /// Permuting tile operation factory function

  /// \param perm The permutation to be applied to the result
  /// \return The tile operation
  template <typename Perm,
            typename = std::enable_if_t<TiledArray::detail::is_permutation_v<
                std::remove_reference_t<Perm>>>>
  op_type make_tile_op(Perm&& perm) const {
    return op_type(op_base_type(this->element_return_op_),
                   std::forward<Perm>(perm));
  }

  /// Construct the distributed evaluator for this expression

  /// \return The distributed evaluator that will evaluate this expression
  dist_eval_type make_dist_eval() const {
    if (this->product_type() == TensorProduct::Contraction)
      return ContEngine_::make_dist_eval();
    else if (this->product_type() == TensorProduct::General)
      return ContEngine_::make_dist_eval_general();
    else
      return BinaryEngine_::make_dist_eval();
  }

  /// Expression identification tag

  /// \return An expression tag used to identify this expression
  const char* make_tag() const { return "[dot_inner] "; }

  /// Expression print

  /// \param os The output stream
  /// \param target_indices The target index list for this expression
  void print(ExprOStream os, const BipartiteIndexList& target_indices) const {
    if (this->product_type() == TensorProduct::Contraction)
      return ContEngine_::print(os, target_indices);
    else
      return BinaryEngine_::print(os, target_indices);
  }
};  // class DotInnerEngine

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_DOT_INNER_ENGINE_H__INCLUDED
