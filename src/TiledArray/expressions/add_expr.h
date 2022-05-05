/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 *  add_expr.h
 *  Apr 1, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_ADD_EXPR_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_ADD_EXPR_H__INCLUDED

#include <TiledArray/expressions/add_engine.h>
#include <TiledArray/expressions/binary_expr.h>

namespace TiledArray {
namespace expressions {

template <typename Left, typename Right>
using ConjAddExpr =
    ScalAddExpr<Left, Right, TiledArray::detail::ComplexConjugate<void> >;

template <typename Left, typename Right, typename Scalar>
using ScalConjAddExpr =
    ScalAddExpr<Left, Right, TiledArray::detail::ComplexConjugate<Scalar> >;

using TiledArray::detail::conj_op;
using TiledArray::detail::mult_t;
using TiledArray::detail::numeric_t;
using TiledArray::detail::scalar_t;

template <typename Left, typename Right>
struct ExprTrait<AddExpr<Left, Right> > {
  typedef Left left_type;    ///< The left-hand expression type
  typedef Right right_type;  ///< The right-hand expression type
  typedef TiledArray::tile_interface::result_of_add_t<
      typename EngineTrait<typename ExprTrait<Left>::engine_type>::eval_type,
      typename EngineTrait<typename ExprTrait<Right>::engine_type>::eval_type>
      result_type;  ///< Result tile type
  typedef AddEngine<typename ExprTrait<Left>::engine_type,
                    typename ExprTrait<Right>::engine_type, result_type>
      engine_type;  ///< Expression engine type
  typedef numeric_t<typename EngineTrait<engine_type>::eval_type>
      numeric_type;  ///< Addition result numeric type
  typedef scalar_t<typename EngineTrait<engine_type>::eval_type>
      scalar_type;  ///< Addition result scalar type
};

template <typename Left, typename Right, typename Scalar>
struct ExprTrait<ScalAddExpr<Left, Right, Scalar> > {
  typedef Left left_type;      ///< The left-hand expression type
  typedef Right right_type;    ///< The right-hand expression type
  typedef Scalar scalar_type;  ///< Expression scalar type
  typedef TiledArray::tile_interface::result_of_add_t<
      typename EngineTrait<typename ExprTrait<Left>::engine_type>::eval_type,
      typename EngineTrait<typename ExprTrait<Right>::engine_type>::eval_type,
      scalar_type>
      result_type;  ///< Result tile type
  typedef ScalAddEngine<typename ExprTrait<Left>::engine_type,
                        typename ExprTrait<Right>::engine_type, Scalar,
                        result_type>
      engine_type;  ///< Expression engine type
  typedef numeric_t<typename EngineTrait<engine_type>::eval_type>
      numeric_type;  ///< Addition numeric type
};

/// Addition expression

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
template <typename Left, typename Right>
class AddExpr : public BinaryExpr<AddExpr<Left, Right> > {
 public:
  typedef AddExpr<Left, Right> AddExpr_;     ///< This class type
  typedef BinaryExpr<AddExpr_> BinaryExpr_;  ///< Binary base class type
  typedef typename ExprTrait<AddExpr_>::left_type
      left_type;  ///< The left-hand expression type
  typedef typename ExprTrait<AddExpr_>::right_type
      right_type;  ///< The right-hand expression type
  typedef typename ExprTrait<AddExpr_>::engine_type
      engine_type;  ///< Expression engine type

  // Compiler generated functions
  AddExpr(const AddExpr_&) = default;
  AddExpr(AddExpr_&&) = default;
  ~AddExpr() = default;
  AddExpr_& operator=(const AddExpr_&) = delete;
  AddExpr_& operator=(AddExpr_&&) = delete;

  /// Expression constructor

  /// \param left The left-hand expression
  /// \param right The right-hand expression
  AddExpr(const left_type& left, const right_type& right)
      : BinaryExpr_(left, right) {}

};  // class AddExpr

/// Add-then-scale expression

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
template <typename Left, typename Right, typename Scalar>
class ScalAddExpr : public BinaryExpr<ScalAddExpr<Left, Right, Scalar> > {
 public:
  typedef ScalAddExpr<Left, Right, Scalar> ScalAddExpr_;  ///< This class type
  typedef BinaryExpr<ScalAddExpr_> BinaryExpr_;  ///< Binary base class type
  typedef typename ExprTrait<ScalAddExpr_>::left_type
      left_type;  ///< The left-hand expression type
  typedef typename ExprTrait<ScalAddExpr_>::right_type
      right_type;  ///< The right-hand expression type
  typedef typename ExprTrait<ScalAddExpr_>::engine_type
      engine_type;  ///< Expression engine type
  typedef typename ExprTrait<ScalAddExpr_>::scalar_type
      scalar_type;  ///< Scalar type

 private:
  scalar_type factor_;  ///< The scaling factor

 public:
  // Compiler generated functions
  ScalAddExpr(const ScalAddExpr_&) = default;
  ScalAddExpr(ScalAddExpr_&&) = default;
  ~ScalAddExpr() = default;
  ScalAddExpr_& operator=(const ScalAddExpr_&) = delete;
  ScalAddExpr_& operator=(ScalAddExpr_&&) = delete;

  /// Expression constructor

  /// \param left The left argument expression
  /// \param right The right argument expression
  /// \param factor The scaling factor
  ScalAddExpr(const left_type& left, const right_type& right,
              const scalar_type factor)
      : BinaryExpr_(left, right), factor_(factor) {}

  /// Scaling factor accessor

  /// \return The scaling factor
  scalar_type factor() const { return factor_; }

};  // class ScalAddExpr

/// Addition expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \param left The left-hand expression object
/// \param right The right-hand expression object
/// \return An addition expression object
template <typename Left, typename Right>
inline AddExpr<Left, Right> operator+(const Expr<Left>& left,
                                      const Expr<Right>& right) {
  static_assert(
      TiledArray::expressions::is_aliased<Left>::value,
      "no_alias() expressions are not allowed on the right-hand side of "
      "the assignment operator.");
  static_assert(
      TiledArray::expressions::is_aliased<Right>::value,
      "no_alias() expressions are not allowed on the right-hand side of "
      "the assignment operator.");
  return AddExpr<Left, Right>(left.derived(), right.derived());
}

/// Scaled-addition expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar A scalar type
/// \param expr The addition expression object
/// \param factor The scaling factor
/// \return A scaled-addition expression object
template <typename Left, typename Right, typename Scalar>
inline typename std::enable_if<TiledArray::detail::is_numeric_v<Scalar>,
                               ScalAddExpr<Left, Right, Scalar> >::type
operator*(const AddExpr<Left, Right>& expr, const Scalar& factor) {
  return ScalAddExpr<Left, Right, Scalar>(expr.left(), expr.right(), factor);
}

/// Scaled-addition expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar A scalar type
/// \param factor The scaling factor
/// \param expr The addition expression object
/// \return A scaled-addition expression object
template <typename Left, typename Right, typename Scalar>
inline typename std::enable_if<TiledArray::detail::is_numeric_v<Scalar>,
                               ScalAddExpr<Left, Right, Scalar> >::type
operator*(const Scalar& factor, const AddExpr<Left, Right>& expr) {
  return ScalAddExpr<Left, Right, Scalar>(expr.left(), expr.right(), factor);
}

/// Scaled-addition expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar1 The expression scaling factor type
/// \tparam Scalar2 The scaling factor type
/// \param expr The addition expression object
/// \param factor The scaling factor
/// \return A scaled-addition expression object
template <typename Left, typename Right, typename Scalar1, typename Scalar2,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar2> >::type* = nullptr>
inline ScalAddExpr<Left, Right, mult_t<Scalar1, Scalar2> > operator*(
    const ScalAddExpr<Left, Right, Scalar1>& expr, const Scalar2& factor) {
  return ScalAddExpr<Left, Right, mult_t<Scalar1, Scalar2> >(
      expr.left(), expr.right(), expr.factor() * factor);
}

/// Scaled-addition expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar1 The scaling factor type
/// \tparam Scalar2 The expression scaling factor type
/// \param factor The scaling factor
/// \param expr The addition expression object
/// \return A scaled-addition expression object
template <typename Left, typename Right, typename Scalar1, typename Scalar2,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar1> >::type* = nullptr>
inline ScalAddExpr<Left, Right, mult_t<Scalar2, Scalar1> > operator*(
    const Scalar1& factor, const ScalAddExpr<Left, Right, Scalar2>& expr) {
  return ScalAddExpr<Left, Right, mult_t<Scalar2, Scalar1> >(
      expr.left(), expr.right(), expr.factor() * factor);
}

/// Negated addition expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \param expr The addition expression object
/// \return A scaled-addition expression object
template <typename Left, typename Right>
inline ScalAddExpr<Left, Right,
                   typename ExprTrait<AddExpr<Left, Right> >::numeric_type>
operator-(const AddExpr<Left, Right>& expr) {
  return ScalAddExpr<Left, Right,
                     typename ExprTrait<AddExpr<Left, Right> >::numeric_type>(
      expr.left(), expr.right(), -1);
}

/// Negated scaled-addition expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar A scalar type
/// \param expr The addition expression object
/// \return A scaled-addition expression object
template <typename Left, typename Right, typename Scalar>
inline ScalAddExpr<Left, Right, Scalar> operator-(
    const ScalAddExpr<Left, Right, Scalar>& expr) {
  return ScalAddExpr<Left, Right, Scalar>(expr.left(), expr.right(), -1);
}

/// Conjugated addition expression factory

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \param expr The addition expression object
/// \return A conjugated addition expression object
template <typename Left, typename Right>
inline ConjAddExpr<Left, Right> conj(const AddExpr<Left, Right>& expr) {
  return ConjAddExpr<Left, Right>(expr.left(), expr.right(), conj_op());
}

/// Conjugated-conjugate addition expression factory

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \param expr The addition expression object
/// \return A tensor expression object
template <typename Left, typename Right>
inline AddExpr<Left, Right> conj(const ConjAddExpr<Left, Right>& expr) {
  return AddExpr<Left, Right>(expr.left(), expr.right());
}

/// Conjugated addition expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar A scalar type
/// \param expr The addition expression object
/// \return A conjugated addition expression object
template <typename Left, typename Right, typename Scalar>
inline ScalConjAddExpr<Left, Right, Scalar> conj(
    const ScalAddExpr<Left, Right, Scalar>& expr) {
  return ScalConjAddExpr<Left, Right, Scalar>(
      expr.left(), expr.right(),
      conj_op(TiledArray::detail::conj(expr.factor())));
}

/// Conjugated-conjugate addition expression factory

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar A scalar type
/// \param expr The scaled conjugate tensor expression object
/// \return A conjugated expression object
template <typename Left, typename Right, typename Scalar>
inline ScalAddExpr<Left, Right, Scalar> conj(
    const ScalConjAddExpr<Left, Right, Scalar>& expr) {
  return ScalAddExpr<Left, Right, Scalar>(
      expr.left(), expr.right(),
      TiledArray::detail::conj(expr.factor().factor()));
}

/// Scaled-conjugated addition expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar A scalar type
/// \param expr The tensor expression object
/// \param factor The scaling factor
/// \return A scaled-tensor expression object
template <typename Left, typename Right, typename Scalar,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar> >::type* = nullptr>
inline ScalConjAddExpr<Left, Right, Scalar> operator*(
    const ConjAddExpr<Left, Right>& expr, const Scalar& factor) {
  return ScalConjAddExpr<Left, Right, Scalar>(expr.left(), expr.right(),
                                              conj_op(factor));
}

/// Scaled-conjugated addition expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar A scalar type
/// \param factor The scaling factor
/// \param expr The tensor expression object
/// \return A scaled-conjugated addition expression object
template <typename Left, typename Right, typename Scalar,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar> >::type* = nullptr>
inline ScalConjAddExpr<Left, Right, Scalar> operator*(
    const Scalar& factor, const ConjAddExpr<Left, Right>& expr) {
  return ScalConjAddExpr<Left, Right, Scalar>(expr.left(), expr.right(),
                                              conj_op(factor));
}

/// Scaled-conjugated addition expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar1 The expression scaling factor type
/// \tparam Scalar2 The scaling factor type
/// \param expr The scaled-tensor expression object
/// \param factor The scaling factor
/// \return A scaled-conjugated addition expression object
template <typename Left, typename Right, typename Scalar1, typename Scalar2,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar2> >::type* = nullptr>
inline ScalConjAddExpr<Left, Right, mult_t<Scalar1, Scalar2> > operator*(
    const ScalConjAddExpr<Left, Right, Scalar1>& expr, const Scalar2& factor) {
  return ScalConjAddExpr<Left, Right, mult_t<Scalar1, Scalar2> >(
      expr.left(), expr.right(), conj_op(expr.factor().factor() * factor));
}

/// Scaled-conjugated addition expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar1 The scaling factor type
/// \tparam Scalar2 The expression scaling factor type
/// \param factor The scaling factor
/// \param expr The scaled-conjugated addition expression object
/// \return A scaled-conjugated addition expression object
template <typename Left, typename Right, typename Scalar1, typename Scalar2,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar1> >::type* = nullptr>
inline ScalConjAddExpr<Left, Right, mult_t<Scalar2, Scalar1> > operator*(
    const Scalar1& factor, const ScalConjAddExpr<Left, Right, Scalar2>& expr) {
  return ScalConjAddExpr<Left, Right, mult_t<Scalar2, Scalar1> >(
      expr.left(), expr.right(), conj_op(expr.factor().factor() * factor));
}

/// Negated-conjugated addition expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \param expr The tensor expression object
/// \return A scaled-addition expression object
template <typename Left, typename Right>
inline ScalConjAddExpr<
    Left, Right, typename ExprTrait<ConjAddExpr<Left, Right> >::numeric_type>
operator-(const ConjAddExpr<Left, Right>& expr) {
  typedef
      typename ExprTrait<ConjAddExpr<Left, Right> >::numeric_type scalar_type;
  return ScalConjAddExpr<Left, Right, scalar_type>(expr.left(), expr.right(),
                                                   conj_op<scalar_type>(-1));
}

/// Negated-conjugated addition expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar A scalar type
/// \param expr The scaled-conjugated-tensor expression object
/// \return A scaled-conjugated addition expression object
template <typename Left, typename Right, typename Scalar>
inline ScalConjAddExpr<Left, Right, Scalar> operator-(
    const ScalConjAddExpr<Left, Right, Scalar>& expr) {
  return ScalConjAddExpr<Left, Right, Scalar>(expr.left(), expr.right(),
                                              conj_op(-expr.factor().factor()));
}

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_ADD_EXPR_H__INCLUDED
