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
 *  scal_expr.h
 *  Mar 16, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_SCAL_EXPR_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_SCAL_EXPR_H__INCLUDED

#include <TiledArray/expressions/unary_expr.h>
#include <TiledArray/expressions/scal_engine.h>

namespace TiledArray {
  namespace expressions {

    using TiledArray::detail::mult_t;
    using TiledArray::detail::numeric_t;
    using TiledArray::detail::scalar_t;

    template <typename Arg, typename Scalar>
    struct ExprTrait<ScalExpr<Arg, Scalar> > {
      typedef Arg argument_type; ///< The argument expression type
      typedef Scalar scalar_type; ///< Addition result scalar type
      typedef TiledArray::tile_interface::result_of_scale_t<
          typename EngineTrait<typename ExprTrait<Arg>::engine_type>::eval_type,
          scalar_type> result_type; ///< Result tile type
      typedef ScalEngine<typename ExprTrait<Arg>::engine_type, Scalar,
          result_type> engine_type; ///< Expression engine type
      typedef numeric_t<typename EngineTrait<engine_type>::eval_type>
          numeric_type; ///< Addition result numeric type
    };

    /// Scaling expression

    /// \tparam Arg The argument expression type
    template <typename Arg, typename Scalar>
    class ScalExpr : public UnaryExpr<ScalExpr<Arg, Scalar> > {
    public:
      typedef ScalExpr<Arg, Scalar> ScalExpr_; ///< This class type
      typedef UnaryExpr<ScalExpr_> UnaryExpr_; ///< Unary base class type
      typedef typename ExprTrait<ScalExpr_>::argument_type argument_type; ///< The argument expression type
      typedef typename ExprTrait<ScalExpr_>::engine_type engine_type; ///< Expression engine type
      typedef typename ExprTrait<ScalExpr_>::scalar_type scalar_type; ///< Scalar type

    private:

      scalar_type factor_; ///< The scaling factor

    public:

      // Compiler generated functions
      ScalExpr(const ScalExpr_&) = default;
      ScalExpr(ScalExpr_&&) = default;
      ~ScalExpr() = default;
      ScalExpr_& operator=(const ScalExpr_&) = delete;
      ScalExpr_& operator=(ScalExpr_&&) = delete;

      /// Scaled expression constructor

      /// \param arg The argument expression
      /// \param factor The scalar type
      ScalExpr(const argument_type& arg, const scalar_type factor) :
        UnaryExpr_(arg), factor_(factor)
      { }

      /// Rescale expression constructor

      /// \param other The expression to be copied
      /// \param factor The scaling factor applied to the new expression
      ScalExpr(const ScalExpr_& other, const scalar_type factor) :
        UnaryExpr_(other), factor_(other.factor_ * factor)
      { }


      /// Scaling factor accessor

      /// \return The scaling factor
      scalar_type factor() const { return factor_; }

    }; // class ScalExpr


    using TiledArray::detail::mult_t;

    /// Scaled expression factor

    /// \tparam Arg The expression type
    /// \tparam Scalar A scalar type
    /// \param expr The expression object
    /// \param factor The scaling factor
    /// \return A scaled expression object
    template <typename Arg, typename Scalar,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar>::value
        >::type* = nullptr>
    inline ScalExpr<Arg, Scalar>
    operator*(const Expr<Arg>& expr, const Scalar& factor) {
      static_assert(TiledArray::expressions::is_aliased<Arg>::value,
          "no_alias() expressions are not allowed on the right-hand side of "
          "the assignment operator.");
      return ScalExpr<Arg, Scalar>(expr.derived(), factor);
    }

    /// Scaled expression factor

    /// \tparam Arg The expression type
    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param expr The expression object
    /// \return A scaled expression object
    template <typename Arg, typename Scalar,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar>::value
        >::type* = nullptr>
    inline ScalExpr<Arg, Scalar>
    operator*(const Scalar& factor, const Expr<Arg>& expr) {
      static_assert(TiledArray::expressions::is_aliased<Arg>::value,
          "no_alias() expressions are not allowed on the right-hand side of "
          "the assignment operator.");
      return ScalExpr<Arg, Scalar>(expr.derived(), factor);
    }

    /// Scaled expression factor

    /// \tparam Arg The argument expression type
    /// \tparam Scalar A scalar type
    /// \param expr The scaled expression object
    /// \param factor The scaling factor
    /// \return A scaled expression object
    template <typename Arg, typename Scalar1, typename Scalar2,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar2>::value
        >::type* = nullptr>
    inline ScalExpr<Arg, mult_t<Scalar1, Scalar2> >
    operator*(const ScalExpr<Arg, Scalar1>& expr, const Scalar2& factor) {
      return ScalExpr<Arg, mult_t<Scalar1, Scalar2> >(expr, factor);
    }

    /// Scaled expression factor

    /// \tparam Arg The argument expression type
    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param expr The scaled expression object
    /// \return A scaled expression object
    template <typename Arg, typename Scalar1, typename Scalar2,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar1>::value
        >::type* = nullptr>
    inline ScalExpr<Arg, mult_t<Scalar2, Scalar1> >
    operator*(const Scalar1& factor, const ScalExpr<Arg, Scalar2>& expr) {
      return ScalExpr<Arg, mult_t<Scalar2, Scalar1> >(expr, factor);
    }

    /// Negated expression factor

    /// \tparam Arg The expression type
    /// \param expr The expression object
    /// \return A scaled expression object
    template <typename Arg>
    inline ScalExpr<Arg, typename ExprTrait<Arg>::scalar_type>
    operator-(const Expr<Arg>& expr) {
      static_assert(TiledArray::expressions::is_aliased<Arg>::value,
          "no_alias() expressions are not allowed on the right-hand side of "
          "the assignment operator.");
      return ScalExpr<Arg, typename ExprTrait<Arg>::scalar_type>(expr.derived(), -1);
    }

    /// Negated expression factor

    /// \tparam Arg The argument expression type
    /// \param expr The scaled expression object
    /// \return A scaled expression object
    template <typename Arg, typename Scalar>
    inline ScalExpr<Arg, Scalar> operator-(const ScalExpr<Arg, Scalar>& expr) {
      return ScalExpr<Arg, Scalar>(expr, -1);
    }

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_SCAL_EXPR_H__INCLUDED
