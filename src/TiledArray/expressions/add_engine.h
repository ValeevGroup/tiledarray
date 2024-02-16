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
 *  add_engine.h
 *  Mar 31, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_ADD_ENGINE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_ADD_ENGINE_H__INCLUDED

#include <TiledArray/expressions/binary_engine.h>
#include <TiledArray/tile_op/add.h>
#include <TiledArray/tile_op/binary_wrapper.h>

namespace TiledArray {
namespace expressions {

// Forward declarations
template <typename, typename>
class AddExpr;
template <typename, typename, typename>
class ScalAddExpr;
template <typename, typename, typename>
class AddEngine;
template <typename, typename, typename, typename>
class ScalAddEngine;

template <typename Left, typename Right, typename Result>
struct EngineTrait<AddEngine<Left, Right, Result>> {
  static_assert(
      std::is_same<typename EngineTrait<Left>::policy,
                   typename EngineTrait<Right>::policy>::value,
      "The left- and right-hand expressions must use the same policy class");

  // Argument typedefs
  typedef Left left_type;    ///< The left-hand expression type
  typedef Right right_type;  ///< The right-hand expression type

  // Operational typedefs
  typedef TiledArray::detail::Add<Result, typename EngineTrait<Left>::eval_type,
                                  typename EngineTrait<Right>::eval_type,
                                  EngineTrait<Left>::consumable,
                                  EngineTrait<Right>::consumable>
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

  static constexpr bool consumable = true;
  static constexpr unsigned int leaves =
      EngineTrait<Left>::leaves + EngineTrait<Right>::leaves;
};

template <typename Left, typename Right, typename Scalar, typename Result>
struct EngineTrait<ScalAddEngine<Left, Right, Scalar, Result>> {
  static_assert(
      std::is_same<typename EngineTrait<Left>::policy,
                   typename EngineTrait<Right>::policy>::value,
      "The left- and right-hand expressions must use the same policy class");

  // Argument typedefs
  typedef Left left_type;    ///< The left-hand expression type
  typedef Right right_type;  ///< The right-hand expression type

  // Operational typedefs
  typedef Scalar scalar_type;  ///< Tile scalar type
  typedef TiledArray::detail::ScalAdd<
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

/// Addition expression engine

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Result The result tile type
template <typename Left, typename Right, typename Result>
class AddEngine : public BinaryEngine<AddEngine<Left, Right, Result>> {
 public:
  // Class hierarchy typedefs
  typedef AddEngine<Left, Right, Result> AddEngine_;  ///< This class type
  typedef BinaryEngine<AddEngine_>
      BinaryEngine_;  ///< Binary expression engine base type
  typedef typename BinaryEngine_::ExprEngine_
      ExprEngine_;  ///< Expression engine base type

  // Argument typedefs
  typedef typename EngineTrait<AddEngine_>::left_type
      left_type;  ///< The left-hand expression type
  typedef typename EngineTrait<AddEngine_>::right_type
      right_type;  ///< The right-hand expression type

  // Operational typedefs
  typedef typename EngineTrait<AddEngine_>::value_type
      value_type;  ///< The result tile type
  typedef typename EngineTrait<AddEngine_>::op_base_type
      op_base_type;  ///< The tile operation type
  typedef typename EngineTrait<AddEngine_>::op_type
      op_type;  ///< The tile operation type
  typedef typename EngineTrait<AddEngine_>::policy
      policy;  ///< The result policy type
  typedef typename EngineTrait<AddEngine_>::dist_eval_type
      dist_eval_type;  ///< The distributed evaluator type

  // Meta data typedefs
  typedef typename EngineTrait<AddEngine_>::size_type size_type;  ///< Size type
  typedef typename EngineTrait<AddEngine_>::trange_type
      trange_type;  ///< Tiled range type
  typedef
      typename EngineTrait<AddEngine_>::shape_type shape_type;  ///< Shape type
  typedef typename EngineTrait<AddEngine_>::pmap_interface
      pmap_interface;  ///< Process map interface type

  /// Constructor

  /// \tparam L The left-hand argument expression type
  /// \tparam R The right-hand argument expression type
  /// \param expr The parent expression
  template <typename L, typename R>
  AddEngine(const AddExpr<L, R>& expr) : BinaryEngine_(expr) {}

  /// Non-permuting shape factory function

  /// \return The result shape
  shape_type make_shape() const {
    return BinaryEngine_::left_.shape().add(BinaryEngine_::right_.shape());
  }

  /// Permuting shape factory function

  /// \param perm The permutation to be applied to the array
  /// \return The result shape
  shape_type make_shape(const Permutation& perm) const {
    return BinaryEngine_::left_.shape().add(BinaryEngine_::right_.shape(),
                                            perm);
  }

  /// Non-permuting tile operation factory function

  /// \return The tile operation
  static op_type make_tile_op() { return op_type(op_base_type()); }

  /// Permuting tile operation factory function

  /// \param perm The permutation to be applied to tiles
  /// \return The tile operation
  template <typename Perm,
            typename = std::enable_if_t<TiledArray::detail::is_permutation_v<
                std::remove_reference_t<Perm>>>>
  static op_type make_tile_op(Perm&& perm) {
    return op_type(op_base_type(), std::forward<Perm>(perm));
  }

  /// Expression identification tag

  /// \return An expression tag used to identify this expression
  const char* make_tag() const { return "[+] "; }

};  // class AddEngine

/// Addition expression engine

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar The scaling factor type
/// \tparam Result The result tile type
template <typename Left, typename Right, typename Scalar, typename Result>
class ScalAddEngine
    : public BinaryEngine<ScalAddEngine<Left, Right, Scalar, Result>> {
 public:
  // Class hierarchy typedefs
  typedef ScalAddEngine<Left, Right, Scalar, Result>
      ScalAddEngine_;  ///< This class type
  typedef BinaryEngine<ScalAddEngine_>
      BinaryEngine_;  ///< Binary expression engine base type
  typedef ExprEngine<ScalAddEngine_>
      ExprEngine_;  ///< Expression engine base type

  // Argument typedefs
  typedef typename EngineTrait<ScalAddEngine_>::left_type
      left_type;  ///< The left-hand expression type
  typedef typename EngineTrait<ScalAddEngine_>::right_type
      right_type;  ///< The right-hand expression type

  // Operational typedefs
  typedef typename EngineTrait<ScalAddEngine_>::value_type
      value_type;  ///< The result tile type
  typedef typename EngineTrait<ScalAddEngine_>::scalar_type
      scalar_type;  ///< Tile scalar type
  typedef typename EngineTrait<ScalAddEngine_>::op_base_type
      op_base_type;  ///< The tile operation type
  typedef typename EngineTrait<ScalAddEngine_>::op_type
      op_type;  ///< The tile operation type
  typedef typename EngineTrait<ScalAddEngine_>::policy
      policy;  ///< The result policy type
  typedef typename EngineTrait<ScalAddEngine_>::dist_eval_type
      dist_eval_type;  ///< The distributed evaluator type

  // Meta data typedefs
  typedef
      typename EngineTrait<ScalAddEngine_>::size_type size_type;  ///< Size type
  typedef typename EngineTrait<ScalAddEngine_>::trange_type
      trange_type;  ///< Tiled range type
  typedef typename EngineTrait<ScalAddEngine_>::shape_type
      shape_type;  ///< Shape type
  typedef typename EngineTrait<ScalAddEngine_>::pmap_interface
      pmap_interface;  ///< Process map interface type

 private:
  scalar_type factor_;  ///< Scaling factor

 public:
  /// Constructor

  /// \tparam L The left-hand argument expression type
  /// \tparam R The right-hand argument expression type
  /// \tparam S The expression scalar type
  /// \param expr The parent expression
  template <typename L, typename R, typename S>
  ScalAddEngine(const ScalAddExpr<L, R, S>& expr)
      : BinaryEngine_(expr), factor_(expr.factor()) {}

  /// Non-permuting shape factory function

  /// \return The result shape
  shape_type make_shape() const {
    return BinaryEngine_::left_.shape().add(BinaryEngine_::right_.shape(),
                                            factor_);
  }

  /// Permuting shape factory function

  /// \param perm The permutation to be applied to the array
  /// \return The result shape
  shape_type make_shape(const Permutation& perm) const {
    return BinaryEngine_::left_.shape().add(BinaryEngine_::right_.shape(),
                                            factor_, perm);
  }

  /// Non-permuting tile operation factory function

  /// \return The tile operation
  op_type make_tile_op() const { return op_type(op_base_type(factor_)); }

  /// Permuting tile operation factory function

  /// \param perm The permutation to be applied to tiles
  /// \return The tile operation
  template <typename Perm,
            typename = std::enable_if_t<TiledArray::detail::is_permutation_v<
                std::remove_reference_t<Perm>>>>
  op_type make_tile_op(Perm&& perm) const {
    return op_type(op_base_type(factor_), std::forward<Perm>(perm));
  }

  /// Scaling factor accessor

  /// \return The scaling factor
  scalar_type factor() { return factor_; }

  /// Expression identification tag

  /// \return An expression tag used to identify this expression
  std::string make_tag() const {
    std::stringstream ss;
    ss << "[+] [" << factor_ << "] ";
    return ss.str();
  }

};  // class ScalAddEngine

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_ADD_ENGINE_H__INCLUDED
