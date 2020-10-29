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
 *  scal_tsr_engine.h
 *  Mar 31, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_SCAL_TSR_ENGINE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_SCAL_TSR_ENGINE_H__INCLUDED

#include <TiledArray/expressions/leaf_engine.h>
#include <TiledArray/tile_op/scal.h>
#include <TiledArray/tile_op/unary_wrapper.h>

namespace TiledArray {
namespace expressions {

template <typename, typename>
class ScalTsrExpr;
template <typename, typename, typename>
class ScalTsrEngine;

template <typename Tile, typename Policy, typename Scalar, typename Result>
struct EngineTrait<ScalTsrEngine<DistArray<Tile, Policy>, Scalar, Result>> {
  // Argument typedefs
  typedef DistArray<Tile, Policy> array_type;  ///< The array type

  // Operational typedefs
  typedef Scalar scalar_type;
  typedef TiledArray::detail::Scal<
      Result,
      typename TiledArray::eval_trait<typename array_type::value_type>::type,
      scalar_type,
      TiledArray::eval_trait<typename array_type::value_type>::is_consumable>
      op_base_type;  ///< The tile operation
  typedef TiledArray::detail::UnaryWrapper<op_base_type> op_type;
  typedef TiledArray::detail::LazyArrayTile<typename array_type::value_type,
                                            op_type>
      value_type;  ///< Tile type
  typedef typename eval_trait<value_type>::type
      eval_type;          ///< Evaluation tile type
  typedef Policy policy;  ///< Policy type
  typedef TiledArray::detail::DistEval<value_type, policy>
      dist_eval_type;  ///< The distributed evaluator type

  // Meta data typedefs
  typedef typename policy::ordinal_type size_type;   ///< Size type
  typedef typename policy::trange_type trange_type;  ///< Tiled range type
  typedef typename policy::shape_type shape_type;    ///< Shape type
  typedef typename policy::pmap_interface
      pmap_interface;  ///< Process map interface type

  static constexpr bool consumable = true;
  static constexpr unsigned int leaves = 1;
};

/// Scaled tensor expression engine

/// \tparam A The \c Array type
template <typename Array, typename Scalar, typename Result>
class ScalTsrEngine : public LeafEngine<ScalTsrEngine<Array, Scalar, Result>> {
 public:
  // Class hierarchy typedefs
  typedef ScalTsrEngine<Array, Scalar, Result>
      ScalTsrEngine_;                              ///< This class type
  typedef LeafEngine<ScalTsrEngine_> LeafEngine_;  ///< Leaf base class type
  typedef typename LeafEngine_::ExprEngine_
      ExprEngine_;  ///< Expression engine base class

  // Argument typedefs
  typedef typename EngineTrait<ScalTsrEngine_>::array_type
      array_type;  ///< The left-hand expression type

  // Operational typedefs
  typedef typename EngineTrait<ScalTsrEngine_>::value_type
      value_type;  ///< Tensor value type
  typedef typename EngineTrait<ScalTsrEngine_>::scalar_type
      scalar_type;  ///< Tile scalar type
  typedef typename EngineTrait<ScalTsrEngine_>::op_base_type
      op_base_type;  ///< Tile base operation type
  typedef typename EngineTrait<ScalTsrEngine_>::op_type
      op_type;  ///< Tile operation type
  typedef typename EngineTrait<ScalTsrEngine_>::policy
      policy;  ///< The result policy type
  typedef typename EngineTrait<ScalTsrEngine_>::dist_eval_type
      dist_eval_type;  ///< This expression's distributed evaluator type

  // Meta data typedefs
  typedef
      typename EngineTrait<ScalTsrEngine_>::size_type size_type;  ///< Size type
  typedef typename EngineTrait<ScalTsrEngine_>::trange_type
      trange_type;  ///< Tiled range type type
  typedef typename EngineTrait<ScalTsrEngine_>::shape_type
      shape_type;  ///< Tensor shape type
  typedef typename EngineTrait<ScalTsrEngine_>::pmap_interface
      pmap_interface;  ///< Process map interface type

 private:
  scalar_type factor_;  ///< The scaling factor

 public:
  template <typename A, typename S>
  ScalTsrEngine(const ScalTsrExpr<A, S>& expr)
      : LeafEngine_(expr), factor_(expr.factor()) {}

  /// Non-permuting shape factory function

  /// \return The result shape
  shape_type make_shape() { return LeafEngine_::array_.shape().scale(factor_); }

  /// Permuting shape factory function

  /// \param perm The permutation to be applied to the array
  /// \return The result shape
  shape_type make_shape(const Permutation& perm) {
    return LeafEngine_::array_.shape().scale(factor_, perm);
  }

  /// Non-permuting tile operation factory function

  /// \return The tile operation
  op_type make_tile_op() const { return op_type(op_base_type(factor_)); }

  /// Permuting tile operation factory function

  /// \param perm The permutation to be applied to tiles
  /// \return The tile operation
  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  op_type make_tile_op(const Perm& perm) const {
    return op_type(op_base_type(factor_), perm);
  }

  /// Expression identification tag

  /// \return An expression tag used to identify this expression
  std::string make_tag() const {
    std::stringstream ss;
    ss << "[" << factor_ << "] ";
    return ss.str();
  }

};  // class ScalTsrEngine

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_SCAL_TSR_ENGINE_H__INCLUDED
