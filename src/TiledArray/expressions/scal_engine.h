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
 *  scal_engine.h
 *  Apr 1, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_SCAL_ENGINE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_SCAL_ENGINE_H__INCLUDED

#include <TiledArray/expressions/unary_engine.h>
#include <TiledArray/tile_op/scal.h>
#include <TiledArray/tile_op/unary_wrapper.h>

namespace TiledArray {
namespace expressions {

// Forward declarations
template <typename, typename>
class ScalExpr;
template <typename, typename, typename>
class ScalEngine;

template <typename Arg, typename Scalar, typename Result>
struct EngineTrait<ScalEngine<Arg, Scalar, Result> > {
  // Argument typedefs
  typedef Arg argument_type;  ///< The argument expression engine type

  // Operational typedefs
  typedef Scalar scalar_type;  ///< Tile scalar type
  typedef TiledArray::detail::Scal<Result, typename EngineTrait<Arg>::eval_type,
                                   scalar_type,
                                   EngineTrait<Arg>::consumable>
      op_base_type;  ///< The tile base operation type
  typedef TiledArray::detail::UnaryWrapper<op_base_type>
      op_type;  ///< The tile operation type
  typedef typename op_type::result_type value_type;  ///< The result tile type
  typedef typename eval_trait<value_type>::type
      eval_type;                                  ///< Evaluation tile type
  typedef typename argument_type::policy policy;  ///< The result policy type
  typedef TiledArray::detail::DistEval<value_type, policy>
      dist_eval_type;  ///< The distributed evaluator type

  // Meta data typedefs
  typedef typename policy::ordinal_type size_type;   ///< Size type
  typedef typename policy::trange_type trange_type;  ///< Tiled range type
  typedef typename policy::shape_type shape_type;    ///< Shape type
  typedef typename policy::pmap_interface
      pmap_interface;  ///< Process map interface type

  static constexpr bool consumable = true;
  static constexpr unsigned int leaves = EngineTrait<Arg>::leaves;
};

/// Scaling expression engine

/// \tparam Arg The argument expression engine type
template <typename Arg, typename Scalar, typename Result>
class ScalEngine : public UnaryEngine<ScalEngine<Arg, Scalar, Result> > {
 public:
  // Class hierarchy typedefs
  typedef ScalEngine<Arg, Scalar, Result> ScalEngine_;  ///< This class type
  typedef UnaryEngine<ScalEngine_>
      UnaryEngine_;  ///< Unary expression engine base type
  typedef typename UnaryEngine_::ExprEngine_
      ExprEngine_;  ///< Expression engine base type

  // Argument typedefs
  typedef typename EngineTrait<ScalEngine_>::argument_type
      argument_type;  ///< The argument expression engine type

  // Operational typedefs
  typedef typename EngineTrait<ScalEngine_>::value_type
      value_type;  ///< The result tile type
  typedef typename EngineTrait<ScalEngine_>::scalar_type
      scalar_type;  ///< Tile scalar type
  typedef typename EngineTrait<ScalEngine_>::op_type
      op_type;  ///< The tile operation type
  typedef typename EngineTrait<ScalEngine_>::policy
      policy;  ///< The result policy type
  typedef typename EngineTrait<ScalEngine_>::dist_eval_type
      dist_eval_type;  ///< The distributed evaluator type

  // Meta data typedefs
  typedef
      typename EngineTrait<ScalEngine_>::size_type size_type;  ///< Size type
  typedef typename EngineTrait<ScalEngine_>::trange_type
      trange_type;  ///< Tiled range type
  typedef
      typename EngineTrait<ScalEngine_>::shape_type shape_type;  ///< Shape type
  typedef typename EngineTrait<ScalEngine_>::pmap_interface
      pmap_interface;  ///< Process map interface type

 private:
  scalar_type factor_;  ///< Scaling factor

 public:
  /// Constructor

  /// \tparam A The argument expression type
  /// \tparam S The expression scalar type
  /// \param expr The parent expression
  template <typename A, typename S>
  ScalEngine(const ScalExpr<A, S>& expr)
      : UnaryEngine_(expr), factor_(expr.factor()) {}

  /// Non-permuting shape factory function

  /// \return The result shape
  shape_type make_shape() const {
    return UnaryEngine_::arg_.shape().scale(factor_);
  }

  /// Permuting shape factory function

  /// \param perm The permutation to be applied to the array
  /// \return The result shape
  shape_type make_shape(const Permutation& perm) const {
    return UnaryEngine_::arg_.shape().scale(factor_, perm);
  }

  /// Non-permuting tile operation factory function

  /// \return The tile operation
  op_type make_tile_op() const { return op_type(factor_); }

  /// Permuting tile operation factory function

  /// \param perm The permutation to be applied to tiles
  /// \return The tile operation
  op_type make_tile_op(const Permutation& perm) const {
    return op_type(perm, factor_);
  }

  /// Expression identification tag

  /// \return An expression tag used to identify this expression
  std::string make_tag() const {
    std::stringstream ss;
    ss << "[" << factor_ << "] ";
    return ss.str();
  }

};  // class ScalEngine

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_SCAL_ENGINE_H__INCLUDED
