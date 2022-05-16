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
 *  mult_expr.h
 *  Mar 31, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_MULT_EXPR_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_MULT_EXPR_H__INCLUDED

#include <TiledArray/expressions/binary_expr.h>
#include <TiledArray/expressions/mult_engine.h>

namespace TiledArray {
namespace expressions {

template <typename Left, typename Right>
using ConjMultExpr =
    ScalMultExpr<Left, Right, TiledArray::detail::ComplexConjugate<void> >;

template <typename Left, typename Right, typename Scalar>
using ScalConjMultExpr =
    ScalMultExpr<Left, Right, TiledArray::detail::ComplexConjugate<Scalar> >;

using TiledArray::detail::conj_op;
using TiledArray::detail::mult_t;
using TiledArray::detail::numeric_t;
using TiledArray::detail::scalar_t;

template <typename Left, typename Right>
struct ExprTrait<MultExpr<Left, Right> > {
  typedef Left left_type;    ///< The left-hand expression type
  typedef Right right_type;  ///< The right-hand expression type
  typedef result_of_mult_t<
      typename EngineTrait<typename ExprTrait<Left>::engine_type>::eval_type,
      typename EngineTrait<typename ExprTrait<Right>::engine_type>::eval_type>
      result_type;  ///< Result tile type
  typedef MultEngine<typename ExprTrait<Left>::engine_type,
                     typename ExprTrait<Right>::engine_type, result_type>
      engine_type;  ///< Expression engine type
  typedef numeric_t<typename EngineTrait<engine_type>::eval_type>
      numeric_type;  ///< Multiplication result numeric type
  typedef scalar_t<typename EngineTrait<engine_type>::eval_type>
      scalar_type;  ///< Multiplication result scalar type
};

template <typename Left, typename Right, typename Scalar>
struct ExprTrait<ScalMultExpr<Left, Right, Scalar> > {
  typedef Left left_type;      ///< The left-hand expression type
  typedef Right right_type;    ///< The right-hand expression type
  typedef Scalar scalar_type;  ///< Tile scalar type
  typedef result_of_mult_t<
      typename EngineTrait<typename ExprTrait<Left>::engine_type>::eval_type,
      typename EngineTrait<typename ExprTrait<Right>::engine_type>::eval_type,
      scalar_type>
      result_type;  ///< Result tile type
  typedef ScalMultEngine<typename ExprTrait<Left>::engine_type,
                         typename ExprTrait<Right>::engine_type, Scalar,
                         result_type>
      engine_type;  ///< Expression engine type
  typedef numeric_t<typename EngineTrait<engine_type>::eval_type>
      numeric_type;  ///< Multiplication result numeric type
};

/// Multiplication expression

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
template <typename Left, typename Right>
class MultExpr : public BinaryExpr<MultExpr<Left, Right> > {
 public:
  typedef MultExpr<Left, Right> MultExpr_;    ///< This class type
  typedef BinaryExpr<MultExpr_> BinaryExpr_;  ///< Binary expression base type
  typedef typename ExprTrait<MultExpr_>::left_type
      left_type;  ///< The left-hand expression type
  typedef typename ExprTrait<MultExpr_>::right_type
      right_type;  ///< The right-hand expression type
  typedef typename ExprTrait<MultExpr_>::engine_type
      engine_type;  ///< Expression engine type

  // Compiler generated functions
  MultExpr(const MultExpr_&) = default;
  MultExpr(MultExpr_&&) = default;
  ~MultExpr() = default;
  MultExpr_& operator=(const MultExpr_&) = delete;
  MultExpr_& operator=(MultExpr_&&) = delete;

  /// Expression constructor

  /// \param left The left-hand expression
  /// \param right The right-hand expression
  MultExpr(const left_type& left, const right_type& right)
      : BinaryExpr_(left, right) {}

  /// Dot product

  /// \tparam Numeric A numeric type
  /// \return The dot product of this expression.
  template <typename Numeric,
            typename std::enable_if<
                TiledArray::detail::is_numeric_v<Numeric> >::type* = nullptr>
  explicit operator Numeric() const {
    auto result = BinaryExpr_::left().dot(BinaryExpr_::right());
    return result.get();
  }

  /// Dot product

  /// \tparam Numeric A numeric type
  /// \return The dot product of this expression.
  template <typename Numeric,
            typename std::enable_if<
                TiledArray::detail::is_numeric_v<Numeric> >::type* = nullptr>
  explicit operator Future<Numeric>() const {
    return BinaryExpr_::left().dot(BinaryExpr_::right());
  }

};  // class MultExpr

/// Multiplication expression

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
template <typename Left, typename Right, typename Scalar>
class ScalMultExpr : public BinaryExpr<ScalMultExpr<Left, Right, Scalar> > {
 public:
  typedef ScalMultExpr<Left, Right, Scalar> ScalMultExpr_;  ///< This class type
  typedef BinaryExpr<ScalMultExpr_>
      BinaryExpr_;  ///< Binary expression base type
  typedef typename ExprTrait<ScalMultExpr_>::left_type
      left_type;  ///< The left-hand expression type
  typedef typename ExprTrait<ScalMultExpr_>::right_type
      right_type;  ///< The right-hand expression type
  typedef typename ExprTrait<ScalMultExpr_>::engine_type
      engine_type;  ///< Expression engine type
  typedef typename ExprTrait<ScalMultExpr_>::scalar_type
      scalar_type;  ///< Tile scalar type

 private:
  scalar_type factor_;  ///< The scaling factor

 public:
  // Compiler generated functions
  ScalMultExpr(const ScalMultExpr_&) = default;
  ScalMultExpr(ScalMultExpr_&&) = default;
  ~ScalMultExpr() = default;
  ScalMultExpr_& operator=(const ScalMultExpr_&) = delete;
  ScalMultExpr_& operator=(ScalMultExpr_&&) = delete;

  /// Expression constructor

  /// \param left The left-hand argument expression
  /// \param right The right-hand argument expression
  /// \param factor The scaling factor
  ScalMultExpr(const left_type& left, const right_type& right,
               const scalar_type factor)
      : BinaryExpr_(left, right), factor_(factor) {}

  /// Scaling factor accessor

  /// \return The scaling factor
  scalar_type factor() const { return factor_; }

};  // class ScalMultExpr

/// Multiplication expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \param left The left-hand expression object
/// \param right The right-hand expression object
/// \return An multiplication expression object
template <typename Left, typename Right>
inline MultExpr<Left, Right> operator*(const Expr<Left>& left,
                                       const Expr<Right>& right) {
  static_assert(
      TiledArray::expressions::is_aliased<Left>::value,
      "no_alias() expressions are not allowed on the right-hand side of the "
      "assignment operator.");
  static_assert(
      TiledArray::expressions::is_aliased<Right>::value,
      "no_alias() expressions are not allowed on the right-hand side of the "
      "assignment operator.");
  return MultExpr<Left, Right>(left.derived(), right.derived());
}

/// Scaled-multiplication expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar A scalar type
/// \param expr The multiplication expression object
/// \param factor The scaling factor
/// \return A scaled-multiplication expression object
template <typename Left, typename Right, typename Scalar,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar> >::type* = nullptr>
inline ScalMultExpr<Left, Right, Scalar> operator*(
    const MultExpr<Left, Right>& expr, const Scalar& factor) {
  return ScalMultExpr<Left, Right, Scalar>(expr.left(), expr.right(), factor);
}

/// Scaled-multiplication expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar A scalar type
/// \param factor The scaling factor
/// \param expr The multiplication expression object
/// \return A scaled-multiplication expression object
template <typename Left, typename Right, typename Scalar,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar> >::type* = nullptr>
inline ScalMultExpr<Left, Right, Scalar> operator*(
    const Scalar& factor, const MultExpr<Left, Right>& expr) {
  return ScalMultExpr<Left, Right, Scalar>(expr.left(), expr.right(), factor);
}

/// Scaled-multiplication expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar1 A scalar type
/// \tparam Scalar2 A scalar type
/// \param expr The multiplication expression object
/// \param factor The scaling factor
/// \return A scaled-multiplication expression object
template <typename Left, typename Right, typename Scalar1, typename Scalar2,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar2> >::type* = nullptr>
inline ScalMultExpr<Left, Right, mult_t<Scalar1, Scalar2> > operator*(
    const ScalMultExpr<Left, Right, Scalar1>& expr, const Scalar2& factor) {
  return ScalMultExpr<Left, Right, mult_t<Scalar1, Scalar2> >(
      expr.left(), expr.right(), expr.factor() * factor);
}

/// Scaled-multiplication expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar1 A scalar type
/// \tparam Scalar2 A scalar type
/// \param factor The scaling factor
/// \param expr The multiplication expression object
/// \return A scaled-multiplication expression object
template <typename Left, typename Right, typename Scalar1, typename Scalar2,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar1> >::type* = nullptr>
inline ScalMultExpr<Left, Right, mult_t<Scalar2, Scalar1> > operator*(
    const Scalar1& factor, const ScalMultExpr<Left, Right, Scalar2>& expr) {
  return ScalMultExpr<Left, Right, mult_t<Scalar2, Scalar1> >(
      expr.left(), expr.right(), expr.factor() * factor);
}

/// Negated multiplication expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \param expr The multiplication expression object
/// \return A scaled-multiplication expression object
template <typename Left, typename Right>
inline ScalMultExpr<Left, Right,
                    typename ExprTrait<MultExpr<Left, Right> >::numeric_type>
operator-(const MultExpr<Left, Right>& expr) {
  return ScalMultExpr<Left, Right,
                      typename ExprTrait<MultExpr<Left, Right> >::numeric_type>(
      expr.left(), expr.right(), -1);
}

/// Negated scaled-multiplication expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \param expr The multiplication expression object
/// \return A scaled-multiplication expression object
template <typename Left, typename Right, typename Scalar>
inline ScalMultExpr<Left, Right, Scalar> operator-(
    const ScalMultExpr<Left, Right, Scalar>& expr) {
  return ScalMultExpr<Left, Right, Scalar>(expr.left(), expr.right(),
                                           -expr.factor());
}

/// Conjugated multiplication expression factory

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \param expr The multiplication expression object
/// \return A conjugated multiplication expression object
template <typename Left, typename Right>
inline ConjMultExpr<Left, Right> conj(const MultExpr<Left, Right>& expr) {
  return ConjMultExpr<Left, Right>(expr.left(), expr.right(), conj_op());
}

/// Conjugated-conjugate multiplication expression factory

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \param expr The multiplication expression object
/// \return A multiplication expression object
template <typename Left, typename Right>
inline MultExpr<Left, Right> conj(const ConjMultExpr<Left, Right>& expr) {
  return MultExpr<Left, Right>(expr.left(), expr.right());
}

/// Conjugated multiplication expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar A scalar type
/// \param expr The multiplication expression object
/// \return A scaled-conjugated multiplication expression object
template <typename Left, typename Right, typename Scalar>
inline ScalConjMultExpr<Left, Right, Scalar> conj(
    const ScalMultExpr<Left, Right, Scalar>& expr) {
  return ScalConjMultExpr<Left, Right, Scalar>(
      expr.left(), expr.right(),
      conj_op(TiledArray::detail::conj(expr.factor())));
}

/// Conjugated-conjugate multiplication expression factory

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar A scalar type
/// \param expr The scaled conjugate tensor expression object
/// \return A scaled multiplication expression object
template <typename Left, typename Right, typename Scalar>
inline ScalMultExpr<Left, Right, Scalar> conj(
    const ScalConjMultExpr<Left, Right, Scalar>& expr) {
  return ScalMultExpr<Left, Right, Scalar>(
      expr.left(), expr.right(),
      TiledArray::detail::conj(expr.factor().factor()));
}

/// Scaled-conjugated multiplication expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar A scalar type
/// \param expr The tensor expression object
/// \param factor The scaling factor
/// \return A scaled-conjugated multiplication expression object
template <typename Left, typename Right, typename Scalar,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar> >::type* = nullptr>
inline ScalConjMultExpr<Left, Right, Scalar> operator*(
    const ConjMultExpr<Left, Right>& expr, const Scalar& factor) {
  return ScalConjMultExpr<Left, Right, Scalar>(expr.left(), expr.right(),
                                               conj_op(factor));
}

/// Scaled-conjugated multiplication expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar A scalar type
/// \param factor The scaling factor
/// \param expr The multiplication expression object
/// \return A scaled-conjugated multiplication expression object
template <typename Left, typename Right, typename Scalar,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar> >::type* = nullptr>
inline ScalConjMultExpr<Left, Right, Scalar> operator*(
    const Scalar& factor, const ConjMultExpr<Left, Right>& expr) {
  return ScalConjMultExpr<Left, Right, Scalar>(expr.left(), expr.right(),
                                               conj_op(factor));
}

/// Scaled-conjugated multiplication expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar1 The expression scaling factor type
/// \tparam Scalar2 The scaling factor type
/// \param expr The scaled-tensor expression object
/// \param factor The scaling factor
/// \return A scaled-conjugated multiplication expression object
template <typename Left, typename Right, typename Scalar1, typename Scalar2,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar2> >::type* = nullptr>
inline ScalConjMultExpr<Left, Right, mult_t<Scalar1, Scalar2> > operator*(
    const ScalConjMultExpr<Left, Right, Scalar1>& expr, const Scalar2& factor) {
  return ScalConjMultExpr<Left, Right, mult_t<Scalar1, Scalar2> >(
      expr.left(), expr.right(), conj_op(expr.factor().factor() * factor));
}

/// Scaled-conjugated multiplication expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar1 The scaling factor type
/// \tparam Scalar2 The expression scaling factor type
/// \param factor The scaling factor
/// \param expr The scaled-conjugated multiplication expression object
/// \return A scaled-conjugated multiplication expression object
template <typename Left, typename Right, typename Scalar1, typename Scalar2,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar1> >::type* = nullptr>
inline ScalConjMultExpr<Left, Right, mult_t<Scalar2, Scalar1> > operator*(
    const Scalar1& factor, const ScalConjMultExpr<Left, Right, Scalar2>& expr) {
  return ScalConjMultExpr<Left, Right, mult_t<Scalar2, Scalar1> >(
      expr.left(), expr.right(), conj_op(expr.factor().factor() * factor));
}

/// Negated-conjugated multiplication expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \param expr The tensor expression object
/// \return A scaled-multiplication expression object
template <typename Left, typename Right>
inline ScalConjMultExpr<
    Left, Right, typename ExprTrait<ConjMultExpr<Left, Right> >::numeric_type>
operator-(const ConjMultExpr<Left, Right>& expr) {
  typedef
      typename ExprTrait<ConjMultExpr<Left, Right> >::numeric_type scalar_type;
  return ScalConjMultExpr<Left, Right, scalar_type>(expr.left(), expr.right(),
                                                    conj_op<scalar_type>(-1));
}

/// Negated-conjugated multiplication expression factor

/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \tparam Scalar A scalar type
/// \param expr The scaled-conjugated-tensor expression object
/// \return A scaled-conjugated multiplication expression object
template <typename Left, typename Right, typename Scalar>
inline ScalConjMultExpr<Left, Right, Scalar> operator-(
    const ScalConjMultExpr<Left, Right, Scalar>& expr) {
  return ScalConjMultExpr<Left, Right, Scalar>(
      expr.left(), expr.right(), conj_op(-expr.factor().factor()));
}

/// Dot product add-to operator

/// \tparam Numeric The numeric result type
/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \param result The result that the dot product will be added to.
/// \param expr The multiply expression object
/// \return A reference to result
template <typename Numeric, typename Left, typename Right,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Numeric> >::type* = nullptr>
inline Numeric& operator+=(Numeric& result, const MultExpr<Left, Right>& expr) {
  result += expr.left().dot(expr.right()).get();
  return result;
}

/// Dot product subtract-to operator

/// \tparam Numeric The numeric result type
/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \param result The result that the dot product will be subtracted from.
/// \param expr The multiply expression object
/// \return A reference to result
template <typename Numeric, typename Left, typename Right,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Numeric> >::type* = nullptr>
inline Numeric& operator-=(Numeric& result, const MultExpr<Left, Right>& expr) {
  result -= expr.left().dot(expr.right()).get();
  return result;
}

/// Dot product multiply-to operator

/// \tparam Numeric The numeric result type
/// \tparam Left The left-hand expression type
/// \tparam Right The right-hand expression type
/// \param result The result that the dot product will be multiplied by.
/// \param expr The multiply expression object
/// \return A reference to result
template <typename Numeric, typename Left, typename Right,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Numeric> >::type* = nullptr>
inline Numeric& operator*=(Numeric& result, const MultExpr<Left, Right>& expr) {
  result *= expr.left().dot(expr.right()).get();
  return result;
}

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_MULT_EXPR_H__INCLUDED
