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
 *  scal_tsr_expr.h
 *  Apr 1, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_SCAL_TSR_EXPR_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_SCAL_TSR_EXPR_H__INCLUDED

#include <TiledArray/expressions/scal_tsr_engine.h>

namespace TiledArray {
namespace expressions {

template <typename Array>
using ConjTsrExpr =
    ScalTsrExpr<Array, TiledArray::detail::ComplexConjugate<void> >;

template <typename Array, typename Scalar>
using ScalConjTsrExpr =
    ScalTsrExpr<Array, TiledArray::detail::ComplexConjugate<Scalar> >;

using TiledArray::detail::conj_op;
using TiledArray::detail::mult_t;
using TiledArray::detail::numeric_t;

template <typename, typename>
class ScalTsrExpr;

template <typename Array, typename Scalar>
struct ExprTrait<ScalTsrExpr<Array, Scalar> > {
  typedef Array array_type;    ///< The \c Array type
  typedef Scalar scalar_type;  ///< Expression scalar type
  typedef TiledArray::tile_interface::result_of_scale_t<
      typename Array::eval_type, scalar_type>
      result_type;  ///< Result tile type
  typedef ScalTsrEngine<Array, scalar_type, result_type>
      engine_type;  ///< Expression engine type
  typedef TiledArray::detail::numeric_t<Array>
      numeric_type;  ///< Array base numeric type
};

/// Expression wrapper for scaled array objects

/// \tparam Array A `DistArray` type
/// \tparam Scalar The scaling factor type
template <typename Array, typename Scalar>
class ScalTsrExpr : public Expr<ScalTsrExpr<Array, Scalar> > {
 public:
  typedef ScalTsrExpr<Array, Scalar> ScalTsrExpr_;  ///< This class type
  typedef Expr<ScalTsrExpr_> Expr_;                 ///< Expression base type
  typedef typename ExprTrait<ScalTsrExpr_>::array_type
      array_type;  ///< The array type
  typedef typename ExprTrait<ScalTsrExpr_>::engine_type
      engine_type;  ///< Expression engine type
  typedef typename ExprTrait<ScalTsrExpr_>::scalar_type
      scalar_type;  ///< Scalar type

 private:
  const array_type& array_;  ///< The array that this expression is bound to
  std::string annotation_;   ///< The array annotation
  scalar_type factor_;       ///< The scaling factor

  // Not allowed
  ScalTsrExpr_& operator=(ScalTsrExpr_&);

 public:
  // Compiler generated functions
  ScalTsrExpr(const ScalTsrExpr_&) = default;
  ScalTsrExpr(ScalTsrExpr_&&) = default;
  ~ScalTsrExpr() = default;
  ScalTsrExpr_& operator=(const ScalTsrExpr_&) = delete;
  ScalTsrExpr_& operator=(ScalTsrExpr_&&) = delete;

  /// Construct a scaled tensor expression

  /// \param array The array object
  /// \param annotation The array annotation
  /// \param factor The scaling factor
  ScalTsrExpr(const array_type& array, const std::string& annotation,
              const scalar_type factor)
      : Expr_(), array_(array), annotation_(annotation), factor_(factor) {}

  /// Array accessor

  /// \return a const reference to this array
  const array_type& array() const { return array_; }

  /// Tensor annotation accessor

  /// \return A const reference to the annotation for this tensor
  const std::string& annotation() const { return annotation_; }

  /// Scaling factor accessor

  /// \return The expression scaling factor
  scalar_type factor() const { return factor_; }

};  // class ScalTsrExpr

/// Scaled-tensor expression factor

/// \tparam Array A `DistArray` type
/// \tparam Scalar A scalar type
/// \param expr The tensor expression object
/// \param factor The scaling factor
/// \return A scaled-tensor expression object
template <typename Array, typename Scalar,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar> >::type* = nullptr>
inline ScalTsrExpr<typename std::remove_const<Array>::type, Scalar> operator*(
    const TsrExpr<Array, true>& expr, const Scalar& factor) {
  return ScalTsrExpr<typename std::remove_const<Array>::type, Scalar>(
      expr.array(), expr.vars(), factor);
}

/// Scaled-tensor expression factor

/// \tparam Array A `DistArray` type
/// \tparam Scalar A scalar type
/// \param factor The scaling factor
/// \param expr The tensor expression object
/// \return A scaled-tensor expression object
template <typename Array, typename Scalar,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar> >::type* = nullptr>
inline ScalTsrExpr<typename std::remove_const<Array>::type, Scalar> operator*(
    const Scalar& factor, const TsrExpr<Array, true>& expr) {
  return ScalTsrExpr<typename std::remove_const<Array>::type, Scalar>(
      expr.array(), expr.annotation(), factor);
}

/// Scaled-tensor expression factor

/// \tparam Array A `DistArray` type
/// \tparam Scalar1 A scalar type
/// \tparam Scalar2 A scalar type
/// \param expr The scaled-tensor expression object
/// \param factor The scaling factor
/// \return A scaled-tensor expression object
template <typename Array, typename Scalar1, typename Scalar2,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar2> >::type* = nullptr>
inline ScalTsrExpr<Array, mult_t<Scalar1, Scalar2> > operator*(
    const ScalTsrExpr<Array, Scalar1>& expr, const Scalar2& factor) {
  return ScalTsrExpr<Array, mult_t<Scalar1, Scalar2> >(
      expr.array(), expr.annotation(), expr.factor() * factor);
}

/// Scaled-tensor expression factor

/// \tparam Array A `DistArray` type
/// \tparam Scalar1 A scalar type
/// \tparam Scalar2 A scalar type
/// \param factor The scaling factor
/// \param expr The scaled-tensor expression object
/// \return A scaled-tensor expression object
template <typename Array, typename Scalar1, typename Scalar2,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar1> >::type* = nullptr>
inline ScalTsrExpr<Array, mult_t<Scalar2, Scalar1> > operator*(
    const Scalar1& factor, const ScalTsrExpr<Array, Scalar2>& expr) {
  return ScalTsrExpr<Array, mult_t<Scalar2, Scalar1> >(
      expr.array(), expr.annotation(), expr.factor() * factor);
}

/// Negated-tensor expression factor

/// \tparam Array A `DistArray` type
/// \param expr The tensor expression object
/// \return A scaled-tensor expression object
template <typename Array>
inline ScalTsrExpr<typename std::remove_const<Array>::type,
                   typename ExprTrait<TsrExpr<Array, true> >::numeric_type>
operator-(const TsrExpr<Array, true>& expr) {
  return ScalTsrExpr<typename std::remove_const<Array>::type,
                     typename ExprTrait<TsrExpr<Array, true> >::numeric_type>(
      expr.array(), expr.annotation(), -1);
}

/// Negated-tensor expression factor

/// \tparam Array A `DistArray` type
/// \tparam Scalar A scalar type
/// \param expr The scaled-tensor expression object
/// \return A scaled-tensor expression object
template <typename Array, typename Scalar>
inline ScalTsrExpr<Array, Scalar> operator-(
    const ScalTsrExpr<Array, Scalar>& expr) {
  return ScalTsrExpr<Array, Scalar>(expr.array(), expr.annotation(),
                                    -expr.factor());
}

/// Conjugated tensor expression factory

/// \tparam Array A `DistArray` type
/// \param expr The tensor expression object
/// \return A conjugated expression object
template <typename Array>
inline ConjTsrExpr<typename std::remove_const<Array>::type> conj(
    const TsrExpr<Array, true>& expr) {
  return ConjTsrExpr<typename std::remove_const<Array>::type>(
      expr.array(), expr.annotation(), conj_op());
}

/// Conjugate-conjugate tensor expression factory

/// \tparam Array A `DistArray` type
/// \param expr The tensor expression object
/// \return A tensor expression object
template <typename Array>
inline TsrExpr<const Array, true> conj(const ConjTsrExpr<Array>& expr) {
  return TsrExpr<const Array, true>(expr.array(), expr.annotation());
}

/// Conjugated-tensor expression factor

/// \tparam Array A `DistArray` type
/// \tparam Scalar A scalar type
/// \param expr The tensor expression object
/// \return A conjugated expression object
template <typename Array, typename Scalar>
inline ScalConjTsrExpr<Array, Scalar> conj(
    const ScalTsrExpr<Array, Scalar>& expr) {
  return ScalConjTsrExpr<Array, Scalar>(
      expr.array(), expr.vars(),
      conj_op(TiledArray::detail::conj(expr.factor())));
}

/// Conjugate-conjugate tensor expression factory

/// \tparam Array A `DistArray` type
/// \tparam Scalar A scalar type
/// \param expr The scaled conjugate tensor expression object
/// \return A conjugated expression object
template <typename Array, typename Scalar>
inline ScalTsrExpr<Array, Scalar> conj(
    const ScalConjTsrExpr<Array, Scalar>& expr) {
  return ScalTsrExpr<Array, Scalar>(
      expr.array(), expr.annotation(),
      TiledArray::detail::conj(expr.factor().factor()));
}

/// Scaled-tensor expression factor

/// \tparam Array A `DistArray` type
/// \tparam Scalar A scalar type
/// \param expr The tensor expression object
/// \param factor The scaling factor
/// \return A scaled-tensor expression object
template <typename Array, typename Scalar,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar> >::type* = nullptr>
inline ScalConjTsrExpr<Array, Scalar> operator*(
    const ConjTsrExpr<const Array>& expr, const Scalar& factor) {
  return ScalConjTsrExpr<Array, Scalar>(expr.array(), expr.annotation(),
                                        conj_op(factor));
}

/// Scaled-tensor expression factor

/// \tparam Array A `DistArray` type
/// \tparam Scalar A scalar type
/// \param factor The scaling factor
/// \param expr The tensor expression object
/// \return A scaled-tensor expression object
template <typename Array, typename Scalar,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar> >::type* = nullptr>
inline ScalConjTsrExpr<Array, Scalar> operator*(
    const Scalar& factor, const ConjTsrExpr<Array>& expr) {
  return ScalConjTsrExpr<Array, Scalar>(expr.array(), expr.annotation(),
                                        conj_op(factor));
}

/// Scaled-tensor expression factor

/// \tparam Array A `DistArray` type
/// \tparam Scalar A scalar type
/// \param expr The scaled-tensor expression object
/// \param factor The scaling factor
/// \return A scaled-tensor expression object
template <typename Array, typename Scalar1, typename Scalar2,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar2> >::type* = nullptr>
inline ScalConjTsrExpr<Array, mult_t<Scalar1, Scalar2> > operator*(
    const ScalConjTsrExpr<Array, Scalar1>& expr, const Scalar2& factor) {
  return ScalConjTsrExpr<Array, mult_t<Scalar1, Scalar2> >(
      expr.array(), expr.annotation(),
      conj_op(expr.factor().factor() * factor));
}

/// Scaled-tensor expression factor

/// \tparam Array A `DistArray` type
/// \tparam Scalar A scalar type
/// \param factor The scaling factor
/// \param expr The scaled-tensor expression object
/// \return A scaled-tensor expression object
template <typename Array, typename Scalar1, typename Scalar2,
          typename std::enable_if<
              TiledArray::detail::is_numeric_v<Scalar1> >::type* = nullptr>
inline ScalConjTsrExpr<Array, mult_t<Scalar2, Scalar1> > operator*(
    const Scalar1& factor, const ScalConjTsrExpr<Array, Scalar2>& expr) {
  return ScalConjTsrExpr<Array, mult_t<Scalar2, Scalar1> >(
      expr.array(), expr.annotation(),
      conj_op(expr.factor().factor() * factor));
}

/// Negated-conjugated-tensor expression factor

/// \tparam Array A `DistArray` type
/// \param expr The tensor expression object
/// \return A scaled-tensor expression object
template <typename Array>
inline ScalConjTsrExpr<Array,
                       typename ExprTrait<ConjTsrExpr<Array> >::numeric_type>
operator-(const ConjTsrExpr<Array>& expr) {
  typedef typename ExprTrait<ConjTsrExpr<Array> >::numeric_type numeric_type;
  return ScalConjTsrExpr<Array, numeric_type>(expr.array(), expr.annotation(),
                                              conj_op<numeric_type>(-1));
}

/// Negated-conjugated-tensor expression factor

/// \tparam Array A `DistArray` type
/// \tparam Scalar A scalar type
/// \param expr The scaled-conjugated-tensor expression object
/// \return A scaled-tensor expression object
template <typename Array, typename Scalar>
inline ScalConjTsrExpr<Array, Scalar> operator-(
    const ScalConjTsrExpr<Array, Scalar>& expr) {
  return ScalConjTsrExpr<Array, Scalar>(expr.array(), expr.annotation(),
                                        conj_op(-expr.factor().factor()));
}

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_SCAL_TSR_EXPR_H__INCLUDED
