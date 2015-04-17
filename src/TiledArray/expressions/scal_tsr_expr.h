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

    template <typename> class ScalTsrExpr;

    template <typename A>
    struct ExprTrait<ScalTsrExpr<A> > {
      typedef A array_type; ///< The \c Array type
      typedef ScalTsrEngine<A> engine_type; ///< Expression engine type
      typedef typename TiledArray::detail::scalar_type<A>::type scalar_type;  ///< Tile scalar type
    };

    /// Expression wrapper for scaled array objects

    /// \tparam A The \c TiledArray::Array type
    template <typename A>
    class ScalTsrExpr : public Expr<ScalTsrExpr<A> > {
    public:
      typedef ScalTsrExpr<A> ScalTsrExpr_; ///< This class type
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
      ScalTsrExpr(const TsrExpr<array_type>& tsr_expr, const scalar_type factor) :
        Expr_(), array_(tsr_expr.array()), vars_(tsr_expr.vars()), factor_(factor)
      { }

      /// Construct a scaled tensor expression from a const tensor expression

      /// \param tsr_expr The const tensor expression
      /// \param factor The scaling factor
      ScalTsrExpr(const TsrExpr<const array_type>& tsr_expr, const scalar_type factor) :
        Expr_(), array_(tsr_expr.array()), vars_(tsr_expr.vars()), factor_(factor)
      { }

      /// Copy constructor

      /// \param other The expression to be copied
      ScalTsrExpr(const ScalTsrExpr_& other, const scalar_type factor) :
        Expr_(other), array_(other.array_), vars_(other.vars_), factor_(other.factor_ * factor)
      { }

      /// Copy constructor

      /// \param other The expression to be copied
      ScalTsrExpr(const ScalTsrExpr_& other) :
        Expr_(other), array_(other.array_), vars_(other.vars_), factor_(other.factor_)
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


    /// Scaled-tensor expression factor

    /// \tparam A An array type
    /// \tparam Scalar A scalar type
    /// \param expr The tensor expression object
    /// \param factor The scaling factor
    /// \return A scaled-tensor expression object
    template <typename A, typename Scalar>
    inline typename madness::enable_if<TiledArray::detail::is_numeric<Scalar>, ScalTsrExpr<A> >::type
    operator*(const TsrExpr<A>& expr, const Scalar& factor) {
      return ScalTsrExpr<A>(expr, factor);
    }

    /// Scaled-tensor expression factor

    /// \tparam A An array type
    /// \tparam Scalar A scalar type
    /// \param expr The tensor expression object
    /// \param factor The scaling factor
    /// \return A scaled-tensor expression object
    template <typename A, typename Scalar>
    inline typename madness::enable_if<TiledArray::detail::is_numeric<Scalar>, ScalTsrExpr<A> >::type
    operator*(const TsrExpr<const A>& expr, const Scalar& factor) {
      return ScalTsrExpr<A>(expr, factor);
    }

    /// Scaled-tensor expression factor

    /// \tparam A An array type
    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param expr The tensor expression object
    /// \return A scaled-tensor expression object
    template <typename A, typename Scalar>
    inline typename madness::enable_if<TiledArray::detail::is_numeric<Scalar>, ScalTsrExpr<A> >::type
    operator*(const Scalar& factor, const TsrExpr<A>& expr) {
      return ScalTsrExpr<A>(expr, factor);
    }

    /// Scaled-tensor expression factor

    /// \tparam A An array type
    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param expr The tensor expression object
    /// \return A scaled-tensor expression object
    template <typename A, typename Scalar>
    inline typename madness::enable_if<TiledArray::detail::is_numeric<Scalar>, ScalTsrExpr<A> >::type
    operator*(const Scalar& factor, const TsrExpr<const A>& expr) {
      return ScalTsrExpr<A>(expr, factor);
    }

    /// Scaled-tensor expression factor

    /// \tparam A An array type
    /// \tparam Scalar A scalar type
    /// \param expr The scaled-tensor expression object
    /// \param factor The scaling factor
    /// \return A scaled-tensor expression object
    template <typename A, typename Scalar>
    inline typename madness::enable_if<TiledArray::detail::is_numeric<Scalar>, ScalTsrExpr<A> >::type
    operator*(const ScalTsrExpr<A>& expr, const Scalar& factor) {
      return ScalTsrExpr<A>(expr, factor);
    }

    /// Scaled-tensor expression factor

    /// \tparam A An array type
    /// \tparam Scalar A scalar type
    /// \param factor The scaling factor
    /// \param expr The scaled-tensor expression object
    /// \return A scaled-tensor expression object
    template <typename A, typename Scalar>
    inline typename madness::enable_if<TiledArray::detail::is_numeric<Scalar>, ScalTsrExpr<A> >::type
    operator*(const Scalar& factor, const ScalTsrExpr<A>& expr) {
      return ScalTsrExpr<A>(expr, factor);
    }

    /// Negated-tensor expression factor

    /// \tparam A An array type
    /// \param expr The tensor expression object
    /// \return A scaled-tensor expression object
    template <typename A>
    inline ScalTsrExpr<A> operator-(const TsrExpr<A>& expr) {
      return ScalTsrExpr<A>(expr, -1);
    }

    /// Negated-tensor expression factor

    /// \tparam A An array type
    /// \param expr The tensor expression object
    /// \return A scaled-tensor expression object
    template <typename A>
    inline ScalTsrExpr<A> operator-(const TsrExpr<const A>& expr) {
      return ScalTsrExpr<A>(expr, -1);
    }

    /// Negated-tensor expression factor

    /// \tparam A An array type
    /// \param expr The scaled-tensor expression object
    /// \return A scaled-tensor expression object
    template <typename A>
    inline ScalTsrExpr<A> operator-(const ScalTsrExpr<A>& expr) {
      return ScalTsrExpr<A>(expr, -1);
    }

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_SCAL_TSR_EXPR_H__INCLUDED
