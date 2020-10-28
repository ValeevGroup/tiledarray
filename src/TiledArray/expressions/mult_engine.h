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
 *  mult_engine.h
 *  Mar 31, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_MULT_ENGINE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_MULT_ENGINE_H__INCLUDED

#include <TiledArray/expressions/cont_engine.h>
#include <TiledArray/tile_op/binary_wrapper.h>
#include <TiledArray/tile_op/mult.h>

namespace TiledArray {
namespace expressions {

// Forward declarations
template <typename, typename>
class MultExpr;
template <typename, typename, typename>
class ScalMultExpr;
template <typename, typename, typename>
class MultEngine;
template <typename, typename, typename, typename>
class ScalMultEngine;

template <typename Left, typename Right, typename Result>
struct EngineTrait<MultEngine<Left, Right, Result>> {
  static_assert(
      std::is_same<typename EngineTrait<Left>::policy,
                   typename EngineTrait<Right>::policy>::value,
      "The left- and right-hand expressions must use the same policy class");

  // Argument typedefs
  typedef Left left_type;    ///< The left-hand expression type
  typedef Right right_type;  ///< The right-hand expression type

  // Operational typedefs
  typedef TiledArray::detail::Mult<
      Result, typename EngineTrait<Left>::eval_type,
      typename EngineTrait<Right>::eval_type, EngineTrait<Left>::consumable,
      EngineTrait<Right>::consumable>
      op_base_type;  ///< The base tile operation type
  typedef TiledArray::detail::BinaryWrapper<op_base_type>
      op_type;  ///< The tile operation type
  typedef typename op_type::result_type value_type;  ///< The result tile type
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

template <typename Left, typename Right, typename Scalar, typename Result>
struct EngineTrait<ScalMultEngine<Left, Right, Scalar, Result>> {
  static_assert(
      std::is_same<typename EngineTrait<Left>::policy,
                   typename EngineTrait<Right>::policy>::value,
      "The left- and right-hand expressions must use the same policy class");

  // Argument typedefs
  typedef Left left_type;    ///< The left-hand expression type
  typedef Right right_type;  ///< The right-hand expression type

  // Operational typedefs
  typedef Scalar scalar_type;  ///< Tile scalar type
  typedef TiledArray::detail::ScalMult<
      Result, typename EngineTrait<Left>::eval_type,
      typename EngineTrait<Right>::eval_type, scalar_type,
      EngineTrait<Left>::consumable, EngineTrait<Right>::consumable>
      op_base_type;  ///< The base tile operation type
  typedef TiledArray::detail::BinaryWrapper<op_base_type>
      op_type;  ///< The tile operation type
  typedef typename op_type::result_type value_type;  ///< The result tile type
  typedef typename eval_trait<value_type>::type
      eval_type;                         ///< Evaluation tile type
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

/// Multiplication expression engine

/// This implements any expression encoded with the multiplication operator.
/// This includes Hadamard product, e.g. \code (c("i,j")=)a("i,j")*b("i,j")
/// \endcode , and pure contractions, e.g. \code (c("i,j")=)a("i,k")*b("k,j")
/// \endcode . \internal mixed Hadamard-contraction case, e.g. \code
/// c("i,j,l")=a("i,l,k")*b("j,l,k") \endcode , is not supported since
///   this requires that the result labels are assigned by user (currently they
///   are computed by this engine)
/// \tparam Left The left-hand engine type
/// \tparam Right The right-hand engine type
/// \tparam Result The result tile type
template <typename Left, typename Right, typename Result>
class MultEngine : public ContEngine<MultEngine<Left, Right, Result>> {
 public:
  // Class hierarchy typedefs
  typedef MultEngine<Left, Right, Result> MultEngine_;  ///< This class type
  typedef ContEngine<MultEngine_>
      ContEngine_;  ///< Contraction engine base class
  typedef BinaryEngine<MultEngine_> BinaryEngine_;  ///< Binary base class type
  typedef BinaryEngine<MultEngine_>
      ExprEngine_;  ///< Expression engine base class type

  // Argument typedefs
  typedef typename EngineTrait<MultEngine_>::left_type
      left_type;  ///< The left-hand expression type
  typedef typename EngineTrait<MultEngine_>::right_type
      right_type;  ///< The right-hand expression type

  // Operational typedefs
  typedef typename EngineTrait<MultEngine_>::value_type
      value_type;  ///< The result tile type
  typedef typename EngineTrait<MultEngine_>::op_base_type
      op_base_type;  ///< The tile operation type
  typedef typename EngineTrait<MultEngine_>::op_type
      op_type;  ///< The tile operation type
  typedef typename EngineTrait<MultEngine_>::policy
      policy;  ///< The result policy type
  typedef typename EngineTrait<MultEngine_>::dist_eval_type
      dist_eval_type;  ///< The distributed evaluator type

  // Meta data typedefs
  typedef
      typename EngineTrait<MultEngine_>::size_type size_type;  ///< Size type
  typedef typename EngineTrait<MultEngine_>::trange_type
      trange_type;  ///< Tiled range type
  typedef
      typename EngineTrait<MultEngine_>::shape_type shape_type;  ///< Shape type
  typedef typename EngineTrait<MultEngine_>::pmap_interface
      pmap_interface;  ///< Process map interface type

 private:
  bool contract_;  ///< Expression type flag (true == contraction, false ==
                   ///< coefficient-wise multiplication)

 public:
  /// Constructor

  /// \tparam L The left-hand argument expression type
  /// \tparam R The right-hand argument expression type
  /// \param expr The parent expression
  template <typename L, typename R>
  MultEngine(const MultExpr<L, R>& expr)
      : ContEngine_(expr), contract_(false) {}

  /// Set the index list for this expression

  /// This function will set the index list for this expression and its
  /// children such that the number of permutations is minimized. The final
  /// index list may not be set to target, which indicates that the
  /// result of this expression will be permuted to match \c target_vars.
  /// \param target_vars The target index list for this expression
  void perm_vars(const BipartiteIndexList& target_vars) {
    if (contract_)
      ContEngine_::perm_vars(target_vars);
    else {
      BinaryEngine_::perm_vars(target_vars);
    }
  }

  /// Initialize the index list of this expression

  /// \param target_vars The target index list for this expression
  void init_vars(const BipartiteIndexList& target_vars) {
    BinaryEngine_::left_.init_vars();
    BinaryEngine_::right_.init_vars();

    // it's either pure Hadamard (detect by checking that left arg's and
    // target's vars are the "same") or contraction
    // TODO add mixed Hadamard+contraction
    if (BinaryEngine_::left_.vars().is_permutation(target_vars)) {
      TA_ASSERT(BinaryEngine_::left_.vars().is_permutation(
          BinaryEngine_::right_.vars()));
      BinaryEngine_::perm_vars(target_vars);
    } else {
      contract_ = true;
      ContEngine_::init_vars();
      ContEngine_::perm_vars(target_vars);
    }
  }

  /// Initialize the index list of this expression
  void init_vars() {
    BinaryEngine_::left_.init_vars();
    BinaryEngine_::right_.init_vars();

    if (BinaryEngine_::left_.vars().is_permutation(
            BinaryEngine_::right_.vars())) {
      if (left_type::leaves <= right_type::leaves)
        ExprEngine_::vars_ = BinaryEngine_::left_.vars();
      else
        ExprEngine_::vars_ = BinaryEngine_::right_.vars();
    } else {
      contract_ = true;
      ContEngine_::init_vars();
    }
  }

  /// Initialize result tensor structure

  /// This function will initialize the permutation, tiled range, and shape
  /// for the result tensor.
  /// \param target_vars The target index list for the result tensor
  void init_struct(const BipartiteIndexList& target_vars) {
    if (contract_)
      ContEngine_::init_struct(target_vars);
    else
      BinaryEngine_::init_struct(target_vars);
  }

  /// Initialize result tensor distribution

  /// This function will initialize the world and process map for the result
  /// tensor.
  /// \param world The world were the result will be distributed
  /// \param pmap The process map for the result tensor tiles
  void init_distribution(World* world, std::shared_ptr<pmap_interface> pmap) {
    if (contract_)
      ContEngine_::init_distribution(world, pmap);
    else
      BinaryEngine_::init_distribution(world, pmap);
  }

  /// Non-permuting tiled range factory function

  /// \return The result tiled range object
  trange_type make_trange() const {
    if (contract_)
      return ContEngine_::make_trange();
    else
      return BinaryEngine_::make_trange();
  }

  /// Permuting tiled range factory function

  /// \param perm The permutation to be applied to the array
  /// \return The result tiled range object
  trange_type make_trange(const Permutation& perm) const {
    if (contract_)
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
  static op_type make_tile_op() { return op_type(op_base_type()); }

  /// Permuting tile operation factory function

  /// \param perm The permutation to be applied to tiles
  /// \return The tile operation
  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  static op_type make_tile_op(const Perm& perm) {
    return op_type(op_base_type(), perm);
  }

  /// Construct the distributed evaluator for this expression

  /// \return The distributed evaluator that will evaluate this expression
  dist_eval_type make_dist_eval() const {
    if (contract_)
      return ContEngine_::make_dist_eval();
    else
      return BinaryEngine_::make_dist_eval();
  }

  /// Expression identification tag

  /// \return An expression tag used to identify this expression
  const char* make_tag() const { return "[*] "; }

  /// Expression print

  /// \param os The output stream
  /// \param target_vars The target index list for this expression
  void print(ExprOStream os, const BipartiteIndexList& target_vars) const {
    if (contract_)
      return ContEngine_::print(os, target_vars);
    else
      return BinaryEngine_::print(os, target_vars);
  }

};  // class MultEngine

/// Scaled multiplication expression engine

/// Similar to MultEngine but implements the result of MultEngine scaled by a
/// constant. \tparam Left The left-hand engine type \tparam Right The
/// Right-hand engine type \tparam Scalar The scaling factor type \tparam Result
/// The result tile type
template <typename Left, typename Right, typename Scalar, typename Result>
class ScalMultEngine
    : public ContEngine<ScalMultEngine<Left, Right, Scalar, Result>> {
 public:
  // Class hierarchy typedefs
  typedef ScalMultEngine<Left, Right, Scalar, Result>
      ScalMultEngine_;  ///< This class type
  typedef ContEngine<ScalMultEngine_>
      ContEngine_;  ///< Contraction engine base class
  typedef BinaryEngine<ScalMultEngine_>
      BinaryEngine_;  ///< Binary base class type
  typedef BinaryEngine<ScalMultEngine_>
      ExprEngine_;  ///< Expression engine base class type

  // Argument typedefs
  typedef typename EngineTrait<ScalMultEngine_>::left_type
      left_type;  ///< The left-hand expression type
  typedef typename EngineTrait<ScalMultEngine_>::right_type
      right_type;  ///< The right-hand expression type

  // Operational typedefs
  typedef typename EngineTrait<ScalMultEngine_>::value_type
      value_type;  ///< The result tile type
  typedef typename EngineTrait<ScalMultEngine_>::scalar_type
      scalar_type;  ///< Tile scalar type
  typedef typename EngineTrait<ScalMultEngine_>::op_base_type
      op_base_type;  ///< The tile operation type
  typedef typename EngineTrait<ScalMultEngine_>::op_type
      op_type;  ///< The tile operation type
  typedef typename EngineTrait<ScalMultEngine_>::policy
      policy;  ///< The result policy type
  typedef typename EngineTrait<ScalMultEngine_>::dist_eval_type
      dist_eval_type;  ///< The distributed evaluator type

  // Meta data typedefs
  typedef typename EngineTrait<ScalMultEngine_>::size_type
      size_type;  ///< Size type
  typedef typename EngineTrait<ScalMultEngine_>::trange_type
      trange_type;  ///< Tiled range type
  typedef typename EngineTrait<ScalMultEngine_>::shape_type
      shape_type;  ///< Shape type
  typedef typename EngineTrait<ScalMultEngine_>::pmap_interface
      pmap_interface;  ///< Process map interface type

 private:
  bool contract_;  ///< Expression type flag (true == contraction, false ==
                   ///< coefficient-wise multiplication)

 public:
  /// Constructor

  /// \tparam L The left-hand argument expression type
  /// \tparam R The right-hand argument expression type
  /// \tparam S The expression scalar type
  /// \param expr The parent expression
  template <typename L, typename R, typename S>
  ScalMultEngine(const ScalMultExpr<L, R, S>& expr)
      : ContEngine_(expr), contract_(false) {}

  /// Set the index list for this expression

  /// This function will set the index list for this expression and its
  /// children such that the number of permutations is minimized. The final
  /// index list may not be set to target, which indicates that the
  /// result of this expression will be permuted to match \c target_vars.
  /// \param target_vars The target index list for this expression
  void perm_vars(const BipartiteIndexList& target_vars) {
    if (contract_)
      ContEngine_::perm_vars(target_vars);
    else {
      BinaryEngine_::perm_vars(target_vars);
    }
  }

  /// Initialize the index list of this expression

  /// \param target_vars The target index list for this expression
  void init_vars(const BipartiteIndexList& target_vars) {
    BinaryEngine_::left_.init_vars();
    BinaryEngine_::right_.init_vars();

    if (BinaryEngine_::left_.vars().is_permutation(
            BinaryEngine_::right_.vars())) {
      BinaryEngine_::perm_vars(target_vars);
    } else {
      contract_ = true;
      ContEngine_::init_vars();
      ContEngine_::perm_vars(target_vars);
    }
  }

  /// Initialize the index list of this expression
  void init_vars() {
    BinaryEngine_::left_.init_vars();
    BinaryEngine_::right_.init_vars();

    if (BinaryEngine_::left_.vars().is_permutation(
            BinaryEngine_::right_.vars())) {
      if (left_type::leaves <= right_type::leaves)
        ExprEngine_::vars_ = BinaryEngine_::left_.vars();
      else
        ExprEngine_::vars_ = BinaryEngine_::right_.vars();
    } else {
      contract_ = true;
      ContEngine_::init_vars();
    }
  }

  /// Initialize result tensor structure

  /// This function will initialize the permutation, tiled range, and shape
  /// for the result tensor.
  /// \param target_vars The target index list for the result tensor
  void init_struct(const BipartiteIndexList& target_vars) {
    if (contract_)
      ContEngine_::init_struct(target_vars);
    else
      BinaryEngine_::init_struct(target_vars);
  }

  /// Initialize result tensor distribution

  /// This function will initialize the world and process map for the result
  /// tensor.
  /// \param world The world were the result will be distributed
  /// \param pmap The process map for the result tensor tiles
  void init_distribution(World* world, std::shared_ptr<pmap_interface> pmap) {
    if (contract_)
      ContEngine_::init_distribution(world, pmap);
    else
      BinaryEngine_::init_distribution(world, pmap);
  }

  /// Construct the distributed evaluator for this expression

  /// \return The distributed evaluator that will evaluate this expression
  dist_eval_type make_dist_eval() const {
    if (contract_)
      return ContEngine_::make_dist_eval();
    else
      return BinaryEngine_::make_dist_eval();
  }

  /// Non-permuting tiled range factory function

  /// \return The result tiled range object
  trange_type make_trange() const {
    if (contract_)
      return ContEngine_::make_trange();
    else
      return BinaryEngine_::make_trange();
  }

  /// Permuting tiled range factory function

  /// \param perm The permutation to be applied to the array
  /// \return The result tiled range object
  trange_type make_trange(const Permutation& perm) const {
    if (contract_)
      return ContEngine_::make_trange(perm);
    else
      return BinaryEngine_::make_trange(perm);
  }

  /// Non-permuting shape factory function

  /// \return The result shape
  shape_type make_shape() const {
    return BinaryEngine_::left_.shape().mult(BinaryEngine_::right_.shape(),
                                             ContEngine_::factor_);
  }

  /// Permuting shape factory function

  /// \param perm The permutation to be applied to the array
  /// \return The result shape
  shape_type make_shape(const Permutation& perm) const {
    return BinaryEngine_::left_.shape().mult(BinaryEngine_::right_.shape(),
                                             ContEngine_::factor_, perm);
  }

  /// Non-permuting tile operation factory function

  /// \return The tile operation
  op_type make_tile_op() const {
    return op_type(op_base_type(ContEngine_::factor_));
  }

  /// Permuting tile operation factory function

  /// \param perm The permutation to be applied to tiles
  /// \return The tile operation
  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  op_type make_tile_op(const Perm& perm) const {
    return op_type(op_base_type(ContEngine_::factor_), perm);
  }

  /// Expression identification tag

  /// \return An expression tag used to identify this expression
  std::string make_tag() const {
    std::stringstream ss;
    ss << "[*] [" << ContEngine_::factor_ << "] ";
    return ss.str();
  }

  /// Expression print

  /// \param os The output stream
  /// \param target_vars The target index list for this expression
  void print(ExprOStream os, const BipartiteIndexList& target_vars) const {
    if (contract_)
      return ContEngine_::print(os, target_vars);
    else
      return BinaryEngine_::print(os, target_vars);
  }

};  // class ScalMultEngine

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_MULT_ENGINE_H__INCLUDED
