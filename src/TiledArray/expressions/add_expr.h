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

    using TiledArray::detail::mult_t;
    using TiledArray::detail::numeric_t;
    using TiledArray::detail::scalar_t;

    template <typename Left, typename Right>
    struct ExprTrait<AddExpr<Left, Right> > {
      typedef Left left_type; ///< The left-hand expression type
      typedef Right right_type; ///< The right-hand expression type
      typedef AddEngine<typename ExprTrait<Left>::engine_type,
          typename ExprTrait<Right>::engine_type>
          engine_type; ///< Expression engine type
      typedef numeric_t<typename EngineTrait<engine_type>::eval_type>
          numeric_type; ///< Addition result numeric type
      typedef scalar_t<typename EngineTrait<engine_type>::eval_type>
          scalar_type; ///< Addition result scalar type
    };

    template <typename Left, typename Right, typename Scalar>
    struct ExprTrait<ScalAddExpr<Left, Right, Scalar> > {
      typedef Left left_type; ///< The left-hand expression type
      typedef Right right_type; ///< The right-hand expression type
      typedef ScalAddEngine<typename ExprTrait<Left>::engine_type,
          typename ExprTrait<Right>::engine_type, Scalar>
          engine_type; ///< Expression engine type
      typedef numeric_t<typename EngineTrait<engine_type>::eval_type>
          numeric_type; ///< Addition numeric type
      typedef Scalar scalar_type;  ///< Expression scalar type
    };


    /// Addition expression

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    template <typename Left, typename Right>
    class AddExpr : public BinaryExpr<AddExpr<Left, Right> > {
    public:
      typedef AddExpr<Left, Right> AddExpr_; ///< This class type
      typedef BinaryExpr<AddExpr<Left, Right> >
          BinaryExpr_; ///< Binary base class type
      typedef typename ExprTrait<AddExpr_>::left_type
          left_type; ///< The left-hand expression type
      typedef typename ExprTrait<AddExpr_>::right_type
          right_type; ///< The right-hand expression type
      typedef typename ExprTrait<AddExpr_>::engine_type
          engine_type; ///< Expression engine type


      /// Expression constructor

      /// \param left The left-hand expression
      /// \param right The right-hand expression
      AddExpr(const left_type& left, const right_type& right) : BinaryExpr_(left, right) { }

      /// Copy constructor

      /// \param other The expression to be copied
      AddExpr(const AddExpr_& other) : BinaryExpr_(other) { }

    }; // class AddExpr


    /// Addition expression

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    template <typename Left, typename Right, typename Scalar>
    class ScalAddExpr : public BinaryExpr<ScalAddExpr<Left, Right, Scalar> > {
    public:
      typedef ScalAddExpr<Left, Right, Scalar>
          ScalAddExpr_; ///< This class type
      typedef BinaryExpr<ScalAddExpr_>
          BinaryExpr_; ///< Binary base class type
      typedef typename ExprTrait<ScalAddExpr_>::left_type
          left_type; ///< The left-hand expression type
      typedef typename ExprTrait<ScalAddExpr_>::right_type
          right_type; ///< The right-hand expression type
      typedef typename ExprTrait<ScalAddExpr_>::engine_type
          engine_type; ///< Expression engine type
      typedef typename ExprTrait<ScalAddExpr_>::scalar_type
          scalar_type; ///< Scalar type


    private:

      scalar_type factor_; ///< The scaling factor

    public:

      /// Expression constructor

      /// \param arg The argument expression
      /// \param factor The scaling factor
      ScalAddExpr(const AddExpr<Left, Right>& arg, const scalar_type factor) :
        BinaryExpr_(arg.left(), arg.right()), factor_(factor)
      { }

      /// Expression constructor

      /// \param arg The scaled expression
      /// \param factor The scaling factor
      ScalAddExpr(const ScalAddExpr_& arg, const scalar_type factor) :
        BinaryExpr_(arg), factor_(arg.factor_ * factor)
      { }

      /// Copy constructor

      /// \param other The expression to be copied
      ScalAddExpr(const ScalAddExpr_& other) :
        BinaryExpr_(other), factor_(other.factor_)
      { }

      /// Scaling factor accessor

      /// \return The scaling factor
      scalar_type factor() const { return factor_; }

    }; // class ScalAddExpr


    /// Addition expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \param left The left-hand expression object
    /// \param right The right-hand expression object
    /// \return An addition expression object
    template <typename Left, typename Right>
    inline AddExpr<Left, Right>
    operator+(const Expr<Left>& left, const Expr<Right>& right) {
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
    inline typename std::enable_if<TiledArray::detail::is_numeric<Scalar>::value,
        ScalAddExpr<Left, Right, Scalar> >::type
    operator*(const AddExpr<Left, Right>& expr, const Scalar& factor) {
      return ScalAddExpr<Left, Right, Scalar>(expr, factor);
    }

    /// Scaled-addition expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param expr The addition expression object
    /// \return A scaled-addition expression object
    template <typename Left, typename Right, typename Scalar>
    inline typename std::enable_if<TiledArray::detail::is_numeric<Scalar>::value,
        ScalAddExpr<Left, Right, Scalar> >::type
    operator*(const Scalar& factor, const AddExpr<Left, Right>& expr) {
      return ScalAddExpr<Left, Right, Scalar>(expr, factor);
    }

    /// Scaled-addition expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \tparam Scalar A scalar type
    /// \param expr The addition expression object
    /// \param factor The scaling factor
    /// \return A scaled-addition expression object
    template <typename Left, typename Right, typename Scalar1, typename Scalar2,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar2>::value
        >::type* = nullptr>
    inline ScalAddExpr<Left, Right, mult_t<Scalar1, Scalar2> >
    operator*(const ScalAddExpr<Left, Right, Scalar1>& expr, const Scalar2& factor) {
      return ScalAddExpr<Left, Right, mult_t<Scalar1, Scalar2> >(expr, factor);
    }

    /// Scaled-addition expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param expr The addition expression object
    /// \return A scaled-addition expression object
    template <typename Left, typename Right, typename Scalar1, typename Scalar2,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar1>::value
        >::type* = nullptr>
    inline ScalAddExpr<Left, Right, mult_t<Scalar2, Scalar1> >
    operator*(const Scalar1& factor, const ScalAddExpr<Left, Right, Scalar2>& expr) {
      return ScalAddExpr<Left, Right, mult_t<Scalar2, Scalar1> >(expr, factor);
    }

    /// Negated addition expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \param expr The addition expression object
    /// \return A scaled-addition expression object
    template <typename Left, typename Right>
    inline ScalAddExpr<Left, Right, typename ExprTrait<AddExpr<Left, Right> >::scalar_type>
    operator-(const AddExpr<Left, Right>& expr) {
      return ScalAddExpr<Left, Right,
          typename ExprTrait<AddExpr<Left, Right> >::scalar_type>(expr, -1);
    }

    /// Negated scaled-addition expression factor

    /// \tparam Left The left-hand expression type
    /// \tparam Right The right-hand expression type
    /// \param expr The addition expression object
    /// \return A scaled-addition expression object
    template <typename Left, typename Right, typename Scalar>
    inline ScalAddExpr<Left, Right, Scalar>
    operator-(const ScalAddExpr<Left, Right, Scalar>& expr) {
      return ScalAddExpr<Left, Right, Scalar>(expr, -1);
    }

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_ADD_EXPR_H__INCLUDED
