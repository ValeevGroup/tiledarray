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
 *  dot_inner_expr.h
 *  Jun 15, 2026
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_DOT_INNER_EXPR_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_DOT_INNER_EXPR_H__INCLUDED

#include <TiledArray/expressions/binary_expr.h>
#include <TiledArray/expressions/dot_inner_engine.h>
#include <TiledArray/tensor.h>

namespace TiledArray {
namespace expressions {

template <typename Left, typename Right>
struct ExprTrait<DotInnerExpr<Left, Right>> {
 private:
  // operand (post-evaluation) tile types -- both must be tensors-of-tensors
  typedef typename EngineTrait<typename ExprTrait<Left>::engine_type>::eval_type
      left_eval_type;
  typedef
      typename EngineTrait<typename ExprTrait<Right>::engine_type>::eval_type
          right_eval_type;
  static_assert(
      TiledArray::detail::is_tensor_of_tensor_v<left_eval_type> &&
          TiledArray::detail::is_tensor_of_tensor_v<right_eval_type>,
      "dot_inner requires both operands to be tensors-of-tensors (ToT)");

  /// inner numeric (scalar) type of the result: the type of the PRODUCT of the
  /// two operands' inner elements, so a mixed-precision inner dot (e.g.
  /// int-inner dot_inner double-inner) accumulates in the promoted type rather
  /// than silently narrowing to the left operand's numeric type.
  typedef std::decay_t<
      decltype(std::declval<TiledArray::detail::numeric_t<left_eval_type>>() *
               std::declval<TiledArray::detail::numeric_t<right_eval_type>>())>
      scalar_value_type;

 public:
  typedef Left left_type;    ///< The left-hand expression type
  typedef Right right_type;  ///< The right-hand expression type
  /// The result tile is a PLAIN tensor of scalars (the inner modes are dotted
  /// away to a scalar per outer cell)
  typedef TiledArray::Tensor<scalar_value_type> result_type;
  typedef DotInnerEngine<typename ExprTrait<Left>::engine_type,
                         typename ExprTrait<Right>::engine_type, result_type>
      engine_type;  ///< Expression engine type
  typedef TiledArray::detail::numeric_t<result_type>
      numeric_type;  ///< Inner-dot result numeric type
  typedef TiledArray::detail::scalar_t<result_type>
      scalar_type;  ///< Inner-dot result scalar type
};

/// Inner-dot expression

/// Computes, for two tensor-of-tensor operands, a per-outer-cell inner dot
/// product over a general outer product, producing a plain tensor-of-scalars
/// result. \tparam Left The left-hand expression type \tparam Right The
/// right-hand expression type
template <typename Left, typename Right>
class DotInnerExpr : public BinaryExpr<DotInnerExpr<Left, Right>> {
 public:
  typedef DotInnerExpr<Left, Right> DotInnerExpr_;  ///< This class type
  typedef BinaryExpr<DotInnerExpr_>
      BinaryExpr_;  ///< Binary expression base type
  typedef typename ExprTrait<DotInnerExpr_>::left_type
      left_type;  ///< The left-hand expression type
  typedef typename ExprTrait<DotInnerExpr_>::right_type
      right_type;  ///< The right-hand expression type
  typedef typename ExprTrait<DotInnerExpr_>::engine_type
      engine_type;  ///< Expression engine type

  // Compiler generated functions
  DotInnerExpr(const DotInnerExpr_&) = default;
  DotInnerExpr(DotInnerExpr_&&) = default;
  ~DotInnerExpr() = default;
  DotInnerExpr_& operator=(const DotInnerExpr_&) = delete;
  DotInnerExpr_& operator=(DotInnerExpr_&&) = delete;

  /// Expression constructor

  /// \param left The left-hand expression
  /// \param right The right-hand expression
  DotInnerExpr(const left_type& left, const right_type& right)
      : BinaryExpr_(left, right) {}

};  // class DotInnerExpr

}  // namespace expressions
}  // namespace TiledArray

#include <TiledArray/expressions/expr.h>

namespace TiledArray {
namespace expressions {

/// Inner-dot factory (out-of-line member of Expr)

/// Builds a DotInnerExpr node from two ToT operands; assignable to a plain
/// tensor-of-scalars array. Defined here (rather than in expr.h) so both Expr
/// and DotInnerExpr are complete.
template <typename Derived>
template <typename D>
auto Expr<Derived>::dot_inner(const Expr<D>& right) const {
  static_assert(
      is_aliased<Derived>::value && is_aliased<D>::value,
      "no_alias() expressions are not allowed as dot_inner operands.");
  return DotInnerExpr<Derived, D>(this->derived(), right.derived());
}

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_DOT_INNER_EXPR_H__INCLUDED
