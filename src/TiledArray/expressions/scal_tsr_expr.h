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

    template <typename, typename> class ScalTsrExpr;

    template <typename A, typename Scalar>
    struct ExprTrait<ScalTsrExpr<A, Scalar> > {
      typedef A array_type; ///< The \c Array type
      typedef ScalTsrEngine<A> engine_type; ///< Expression engine type
      typedef Scalar scalar_type;  ///< Tile scalar type
    };

    /// Expression wrapper for scaled array objects

    /// \tparam A The \c TiledArray::Array type
    /// \tparam Scalar The scaling factor type
    template <typename Array, typename Scalar>
    class ScalTsrExpr : public Expr<ScalTsrExpr<Array, Scalar> > {
    public:
      typedef ScalTsrExpr<Array, Scalar> ScalTsrExpr_; ///< This class type
      typedef Expr<ScalTsrExpr_> Expr_; ///< Expression base type
      typedef typename ExprTrait<ScalTsrExpr_>::array_type array_type; ///< The array type
      typedef typename ExprTrait<ScalTsrExpr_>::engine_type engine_type; ///< Expression engine type
      typedef typename ExprTrait<ScalTsrExpr_>::scalar_type scalar_type; ///< Scalar type

    private:

      const array_type& array_; ///< The array that this expression
      std::string vars_; ///< The tensor variable list
      scalar_type factor_; ///< The scaling factor

      // Not allowed
      ScalTsrExpr_& operator=(ScalTsrExpr_&);

    public:

      /// Construct a scaled tensor expression from a tensor expression

      /// \param tsr_expr The tensor expression
      /// \param factor The scaling factor
      template <typename A, typename S>
      ScalTsrExpr(const TsrExpr<A>& tsr_expr, const S factor) :
        Expr_(), array_(tsr_expr.array()), vars_(tsr_expr.vars()), factor_(factor)
      { }
//
//      /// Construct a scaled tensor expression from a const tensor expression
//
//      /// \param tsr_expr The const tensor expression
//      /// \param factor The scaling factor
//      template <typename S>
//      ScalTsrExpr(const TsrExpr<const array_type>& tsr_expr, const S factor) :
//        Expr_(), array_(tsr_expr.array()), vars_(tsr_expr.vars()), factor_(factor)
//      { }

      /// Copy constructor

      /// \param other The expression to be copied
      /// \param factor The scaling factor applied to the new expression
      template <typename A, typename S1, typename S2>
      ScalTsrExpr(const ScalTsrExpr<A, S1>& other, const S2 factor) :
        Expr_(), array_(other.array()), vars_(other.vars()), factor_(other.factor() * factor)
      { }

      /// Copy constructor

      /// \param other The expression to be copied
      ScalTsrExpr(const ScalTsrExpr_& other) :
        Expr_(), array_(other.array_), vars_(other.vars_), factor_(other.factor_)
      { }

      /// Array accessor

      /// \return a const reference to this array
      const array_type& array() const { return array_; }

      /// Tensor variable string accessor

      /// \return A const reference to the variable string for this tensor
      const std::string& vars() const { return vars_; }


      /// Scaling factor accessor

      /// \return The expression scaling factor
      scalar_type factor() const { return factor_; }

    }; // class ScalTsrExpr


    using TiledArray::detail::mult_t;

    /// Scaled-tensor expression factor

    /// \tparam A An array type
    /// \tparam Scalar A scalar type
    /// \param expr The tensor expression object
    /// \param factor The scaling factor
    /// \return A scaled-tensor expression object
    template <typename A, typename Scalar,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar>::value
        >::type* = nullptr>
    inline ScalTsrExpr<A, Scalar>
    operator*(const TsrExpr<A>& expr, const Scalar& factor) {
      return ScalTsrExpr<A, Scalar>(expr, factor);
    }

    /// Scaled-tensor expression factor

    /// \tparam A An array type
    /// \tparam Scalar A scalar type
    /// \param expr The tensor expression object
    /// \param factor The scaling factor
    /// \return A scaled-tensor expression object
    template <typename A, typename Scalar,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar>::value
        >::type* = nullptr>
    inline ScalTsrExpr<A, Scalar>
    operator*(const TsrExpr<const A>& expr, const Scalar& factor) {
      return ScalTsrExpr<A, Scalar>(expr, factor);
    }

    /// Scaled-tensor expression factor

    /// \tparam A An array type
    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param expr The tensor expression object
    /// \return A scaled-tensor expression object
    template <typename A, typename Scalar,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar>::value
        >::type* = nullptr>
    inline ScalTsrExpr<A, Scalar>
    operator*(const Scalar& factor, const TsrExpr<A>& expr) {
      return ScalTsrExpr<A, Scalar>(expr, factor);
    }

    /// Scaled-tensor expression factor

    /// \tparam A An array type
    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param expr The tensor expression object
    /// \return A scaled-tensor expression object
    template <typename A, typename Scalar,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar>::value
        >::type* = nullptr>
    inline ScalTsrExpr<A, Scalar>
    operator*(const Scalar& factor, const TsrExpr<const A>& expr) {
      return ScalTsrExpr<A, Scalar>(expr, factor);
    }

    /// Scaled-tensor expression factor

    /// \tparam A An array type
    /// \tparam Scalar A scalar type
    /// \param expr The scaled-tensor expression object
    /// \param factor The scaling factor
    /// \return A scaled-tensor expression object
    template <typename A, typename Scalar1, typename Scalar2,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar2>::value
        >::type* = nullptr>
    inline ScalTsrExpr<A, mult_t<Scalar1, Scalar2> >
    operator*(const ScalTsrExpr<A, Scalar1>& expr, const Scalar2& factor) {
      return ScalTsrExpr<A, mult_t<Scalar1, Scalar2> >(expr, factor);
    }

    /// Scaled-tensor expression factor

    /// \tparam A An array type
    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param expr The scaled-tensor expression object
    /// \return A scaled-tensor expression object
    template <typename A, typename Scalar1, typename Scalar2,
        typename std::enable_if<
            TiledArray::detail::is_numeric<Scalar1>::value
        >::type* = nullptr>
    inline ScalTsrExpr<A, mult_t<Scalar2, Scalar1> >
    operator*(const Scalar1& factor, const ScalTsrExpr<A, Scalar2>& expr) {
      return ScalTsrExpr<A, mult_t<Scalar2, Scalar1> >(expr, factor);
    }

    /// Negated-tensor expression factor

    /// \tparam A An array type
    /// \param expr The tensor expression object
    /// \return A scaled-tensor expression object
    template <typename A>
    inline ScalTsrExpr<A, typename ExprTrait<TsrExpr<A> >::scalar_type>
    operator-(const TsrExpr<A>& expr) {
      return ScalTsrExpr<A, typename ExprTrait<TsrExpr<A> >::scalar_type>(expr, -1);
    }

    /// Negated-tensor expression factor

    /// \tparam A An array type
    /// \param expr The tensor expression object
    /// \return A scaled-tensor expression object
    template <typename A>
    inline ScalTsrExpr<A, typename ExprTrait<TsrExpr<const A> >::scalar_type>
    operator-(const TsrExpr<const A>& expr) {
      return ScalTsrExpr<A, typename ExprTrait<TsrExpr<const A> >::scalar_type>(expr, -1);
    }

    /// Negated-tensor expression factor

    /// \tparam A An array type
    /// \param expr The scaled-tensor expression object
    /// \return A scaled-tensor expression object
    template <typename A, typename Scalar>
    inline ScalTsrExpr<A, Scalar> operator-(const ScalTsrExpr<A, Scalar>& expr) {
      return ScalTsrExpr<A, Scalar>(expr, -1);
    }

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_SCAL_TSR_EXPR_H__INCLUDED
