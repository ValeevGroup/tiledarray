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
 *  subt_expr.h
 *  Apr 1, 2014
 *
 */


#ifndef TILEDARRAY_EXPRESSIONS_SUBT_EXPR_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_SUBT_EXPR_H__INCLUDED

#include <TiledArray/expressions/binary_expr.h>
#include <TiledArray/expressions/subt_engine.h>

namespace TiledArray {
  namespace expressions {

    using TiledArray::detail::numeric_t;
    using TiledArray::detail::scalar_t;

    template <typename Left, typename Right>
    struct ExprTrait<SubtExpr<Left, Right> > {
      typedef Left left_type; ///< The left-hand expression type
      typedef Right right_type; ///< The right-hand expression type
      typedef SubtEngine<typename ExprTrait<Left>::engine_type,
          typename ExprTrait<Right>::engine_type> engine_type; ///< Expression engine type
      typedef numeric_t<typename EngineTrait<engine_type>::eval_type>
          numeric_type; ///< Subtraction result numeric type
      typedef scalar_t<typename EngineTrait<engine_type>::eval_type>
          scalar_type; ///< Subtraction result scalar type
    };

    template <typename Left, typename Right, typename Scalar>
    struct ExprTrait<ScalSubtExpr<Left, Right, Scalar> > {
      typedef Left left_type; ///< The left-hand expression type
      typedef Right right_type; ///< The right-hand expression type
      typedef ScalSubtEngine<typename ExprTrait<Left>::engine_type,
          typename ExprTrait<Right>::engine_type, Scalar> engine_type; ///< Expression engine type
      typedef numeric_t<typename EngineTrait<engine_type>::eval_type>
          numeric_type; ///< Subtraction result numeric type
      typedef Scalar scalar_type;  ///< Tile scalar type
    };


    /// Subtraction expression

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    template <typename Left, typename Right>
    class SubtExpr : public BinaryExpr<SubtExpr<Left, Right> > {
    public:
      typedef SubtExpr<Left, Right> SubtExpr_;
      typedef BinaryExpr<SubtExpr_> BinaryExpr_; ///< Binary base class type
      typedef typename ExprTrait<SubtExpr_>::left_type left_type; ///< The left-hand expression type
      typedef typename ExprTrait<SubtExpr_>::right_type right_type; ///< The right-hand expression type
      typedef typename ExprTrait<SubtExpr_>::engine_type engine_type; ///< Expression engine type

    private:

      // Not allowed
      SubtExpr_& operator=(const SubtExpr_&);

    public:

      /// Expression constructor

      /// \param left The left-hand expression
      /// \param right The right-hand expression
      SubtExpr(const left_type& left, const right_type& right) : BinaryExpr_(left, right) { }

      /// Copy constructor

      /// \param other The expression to be copied
      SubtExpr(const SubtExpr_& other) : BinaryExpr_(other) { }

    }; // class SubtExpr


    /// Subtraction expression

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    template <typename Left, typename Right, typename Scalar>
    class ScalSubtExpr : public BinaryExpr<ScalSubtExpr<Left, Right, Scalar> > {
    public:
      typedef ScalSubtExpr<Left, Right, Scalar> ScalSubtExpr_; ///< This class type
      typedef BinaryExpr<ScalSubtExpr_> BinaryExpr_; ///< Binary base class type
      typedef typename ExprTrait<ScalSubtExpr_>::left_type left_type; ///< The left-hand expression type
      typedef typename ExprTrait<ScalSubtExpr_>::right_type right_type; ///< The right-hand expression type
      typedef typename ExprTrait<ScalSubtExpr_>::engine_type engine_type; ///< Expression engine type
      typedef typename ExprTrait<ScalSubtExpr_>::scalar_type scalar_type; ///< Scalar type

    private:

      scalar_type factor_; ///< The scaling factor

      // Not allowed
      ScalSubtExpr_& operator=(const ScalSubtExpr_&);

    public:

      /// Expression constructor

      /// \param arg The argument expression
      /// \param factor The scaling factor
      ScalSubtExpr(const SubtExpr<Left, Right>& arg, const scalar_type factor) :
        BinaryExpr_(arg.left(), arg.right()), factor_(factor)
      { }

      /// Expression constructor

      /// \param arg The scaled expression
      /// \param factor The scaling factor
      ScalSubtExpr(const ScalSubtExpr_& arg, const scalar_type factor) :
        BinaryExpr_(arg), factor_(arg.factor_ * factor)
      { }

      /// Copy constructor

      /// \param other The expression to be copied
      ScalSubtExpr(const ScalSubtExpr_& other) :
        BinaryExpr_(other), factor_(other.factor_)
      { }

      /// Scaling factor accessor

      /// \return The scaling factor
      scalar_type factor() const { return factor_; }

    }; // class ScalSubtExpr


    using TiledArray::detail::mult_t;

    /// Subtraction expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \param left The left-hand expression object
    /// \param right The right-hand expression object
    /// \return A subtraction expression object
    template <typename Left, typename Right>
    inline SubtExpr<Left, Right> operator-(const Expr<Left>& left, const Expr<Right>& right) {
      return SubtExpr<Left, Right>(left.derived(), right.derived());
    }

    /// Scaled-subtraction expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \tparam Scalar A scalar type
    /// \param expr The subtraction expression object
    /// \param factor The scaling factor
    /// \return A scaled-subtraction expression object
    template <typename Left, typename Right, typename Scalar,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar>::value
        >::type* = nullptr>
    inline ScalSubtExpr<Left, Right, Scalar>
    operator*(const SubtExpr<Left, Right>& expr, const Scalar& factor) {
      return ScalSubtExpr<Left, Right, Scalar>(expr, factor);
    }

    /// Scaled-subtraction expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param expr The subtraction expression object
    /// \return A scaled-subtraction expression object
    template <typename Left, typename Right, typename Scalar,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar>::value
        >::type* = nullptr>
    inline ScalSubtExpr<Left, Right, Scalar>
    operator*(const Scalar& factor, const SubtExpr<Left, Right>& expr) {
      return ScalSubtExpr<Left, Right, Scalar>(expr, factor);
    }

    /// Scaled-subtraction expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \tparam Scalar A scalar type
    /// \param expr The scaled-subtraction expression object
    /// \param factor The scaling factor
    /// \return A scaled-subtraction expression object
    template <typename Left, typename Right, typename Scalar1, typename Scalar2,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar2>::value
        >::type* = nullptr>
    inline ScalSubtExpr<Left, Right, mult_t<Scalar1, Scalar2> >
    operator*(const ScalSubtExpr<Left, Right, Scalar1>& expr, const Scalar2& factor) {
      return ScalSubtExpr<Left, Right, mult_t<Scalar1, Scalar2> >(expr, factor);
    }

    /// Scaled-subtraction expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param expr The scaled-subtraction expression object
    /// \return A scaled-subtraction expression object
    template <typename Left, typename Right, typename Scalar1, typename Scalar2,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar1>::value
        >::type* = nullptr>
    inline ScalSubtExpr<Left, Right, mult_t<Scalar2, Scalar1> >
    operator*(const Scalar1& factor, const ScalSubtExpr<Left, Right, Scalar2>& expr) {
      return ScalSubtExpr<Left, Right, mult_t<Scalar2, Scalar1> >(expr, factor);
    }


    /// Negated addition expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \param expr The addition expression object
    /// \return A scaled-addition expression object
    template <typename Left, typename Right>
    inline ScalSubtExpr<Left, Right, typename ExprTrait<SubtExpr<Left, Right> >::scalar_type>
    operator-(const SubtExpr<Left, Right>& expr) {
      return ScalSubtExpr<Left, Right,
          typename SubtExpr<Left, Right>::scalar_type>(expr, -1);
    }

    /// Negated scaled-addition expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \param expr The addition expression object
    /// \return A scaled-addition expression object
    template <typename Left, typename Right, typename Scalar>
    inline ScalSubtExpr<Left, Right, Scalar>
    operator-(const ScalSubtExpr<Left, Right, Scalar>& expr) {
      return ScalSubtExpr<Left, Right, Scalar>(expr, -1);
    }

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_SUBT_EXPR_H__INCLUDED
