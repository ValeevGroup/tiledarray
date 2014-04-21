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
#include <TiledArray/expressions/mult_cont_engine.h>

namespace TiledArray {
  namespace expressions {

    template <typename Left, typename Right>
    struct ExprTrait<MultExpr<Left, Right> > : public BinaryExprTrait<Left, Right, MultContEngine>
    { };

    template <typename Left, typename Right>
    struct ExprTrait<ScalMultExpr<Left, Right> > : public BinaryExprTrait<Left, Right, ScalMultContEngine>
    { };


    /// Multiplication expression

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    template <typename Left, typename Right>
    class MultExpr : public BinaryExpr<MultExpr<Left, Right> > {
    public:
      typedef MultExpr<Left, Right> MultExpr_; ///< This class type
      typedef BinaryExpr<MultExpr_> BinaryExpr_; ///< Binary expression base type
      typedef typename ExprTrait<MultExpr_>::left_type left_type; ///< The left-hand expression type
      typedef typename ExprTrait<MultExpr_>::right_type right_type; ///< The right-hand expression type
      typedef typename ExprTrait<MultExpr_>::engine_type engine_type; ///< Expression engine type

    private:

      // Not allowed
      MultExpr_& operator=(const MultExpr_&);

    public:

      /// Expression constructor

      /// \param left The left-hand expression
      /// \param right The right-hand expression
      MultExpr(const left_type& left, const right_type& right) : BinaryExpr_(left, right) { }

      /// Copy constructor

      /// \param other The expression to be copied
      MultExpr(const MultExpr_& other) : BinaryExpr_(other) { }

    }; // class MultExpr


    /// Multiplication expression

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    template <typename Left, typename Right>
    class ScalMultExpr : public BinaryExpr<MultExpr<Left, Right> > {
    public:
      typedef ScalMultExpr<Left, Right> ScalMultExpr_; ///< This class type
      typedef MultExpr<Left, Right> MultExpr_; ///< Base expr type
      typedef BinaryExpr<MultExpr_> BinaryExpr_; ///< Binary expression base type
      typedef typename ExprTrait<ScalMultExpr_>::left_type left_type; ///< The left-hand expression type
      typedef typename ExprTrait<ScalMultExpr_>::right_type right_type; ///< The right-hand expression type
      typedef typename ExprTrait<ScalMultExpr_>::engine_type engine_type; ///< Expression engine type
      typedef typename ExprTrait<ScalMultExpr_>::scalar_type scalar_type; ///< Tile scalar type

    private:

      scalar_type factor_; ///< The scaling factor

      // Not allowed
      ScalMultExpr_& operator=(const ScalMultExpr_&);

    public:

      /// Expression constructor

      /// \param other The expression to be copied
      ScalMultExpr(const MultExpr<Left, Right>& arg, const scalar_type factor) :
        BinaryExpr_(arg.left(), arg.right()), factor_(factor)
      { }

      /// Expression constructor

      /// \param arg The scaled expression
      /// \param factor The scaling factor
      ScalMultExpr(const ScalMultExpr_& arg, const scalar_type factor) :
        BinaryExpr_(arg), factor_(factor * arg.factor_)
      { }

      /// Copy constructor

      /// \param other The expression to be copied
      ScalMultExpr(const ScalMultExpr_& other) :
        BinaryExpr_(other), factor_(other.factor_)
      { }

      /// Scaling factor accessor

      /// \return The scaling factor
      scalar_type factor() const { return factor_; }

    }; // class ScalMultExpr


    /// Multiplication expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \param left The left-hand expression object
    /// \param right The right-hand expression object
    /// \return An multiplication expression object
    template <typename Left, typename Right>
    inline MultExpr<Left, Right> operator*(const Expr<Left>& left, const Expr<Right>& right) {
      return MultExpr<Left, Right>(left.derived(), right.derived());
    }

    /// Scaled-multiplication expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \tparam Scalar A scalar type
    /// \param expr The multiplication expression object
    /// \param factor The scaling factor
    /// \return A scaled-multiplication expression object
    template <typename Left, typename Right, typename Scalar>
    inline typename madness::enable_if<TiledArray::detail::is_numeric<Scalar>, ScalMultExpr<Left, Right> >::type
    operator*(const MultExpr<Left, Right>& expr, const Scalar& factor) {
      return ScalMultExpr<Left, Right>(expr, factor);
    }

    /// Scaled-multiplication expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param expr The multiplication expression object
    /// \return A scaled-multiplication expression object
    template <typename Left, typename Right, typename Scalar>
    inline typename madness::enable_if<TiledArray::detail::is_numeric<Scalar>, ScalMultExpr<Left, Right> >::type
    operator*(const Scalar& factor, const MultExpr<Left, Right>& expr) {
      return ScalMultExpr<Left, Right>(expr, factor);
    }

    /// Scaled-multiplication expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \tparam Scalar A scalar type
    /// \param expr The multiplication expression object
    /// \param factor The scaling factor
    /// \return A scaled-multiplication expression object
    template <typename Left, typename Right, typename Scalar>
    inline typename madness::enable_if<TiledArray::detail::is_numeric<Scalar>, ScalMultExpr<Left, Right> >::type
    operator*(const ScalMultExpr<Left, Right>& expr, const Scalar& factor) {
      return ScalMultExpr<Left, Right>(expr, factor);
    }

    /// Scaled-multiplication expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param expr The multiplication expression object
    /// \return A scaled-multiplication expression object
    template <typename Left, typename Right, typename Scalar>
    inline typename madness::enable_if<TiledArray::detail::is_numeric<Scalar>, ScalMultExpr<Left, Right> >::type
    operator*(const Scalar& factor, const ScalMultExpr<Left, Right>& expr) {
      return ScalMultExpr<Left, Right>(expr, factor);
    }

    /// Negated multiplication expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \param expr The multiplication expression object
    /// \return A scaled-multiplication expression object
    template <typename Left, typename Right>
    inline ScalMultExpr<Left, Right> operator-(const MultExpr<Left, Right>& expr) {
      return ScalMultExpr<Left, Right>(expr, -1);
    }

    /// Negated scaled-multiplication expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \param expr The multiplication expression object
    /// \return A scaled-multiplication expression object
    template <typename Left, typename Right, typename Scalar>
    inline ScalMultExpr<Left, Right> operator-(const ScalMultExpr<Left, Right>& expr) {
      return ScalMultExpr<Left, Right>(expr, -1);
    }

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_MULT_EXPR_H__INCLUDED
