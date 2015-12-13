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
 *  tsr_expr.h
 *  Apr 1, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_TSR_EXPR_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_TSR_EXPR_H__INCLUDED

#include <TiledArray/expressions/add_expr.h>
#include <TiledArray/expressions/subt_expr.h>
#include <TiledArray/expressions/mult_expr.h>
#include <TiledArray/expressions/tsr_engine.h>
#include <TiledArray/expressions/blk_tsr_expr.h>
#include <TiledArray/expressions/scal_tsr_expr.h>

namespace TiledArray {
  namespace expressions {

    using TiledArray::detail::numeric_t;
    using TiledArray::detail::scalar_t;

    template <typename Array>
    struct ExprTrait<TsrExpr<Array> > {
      typedef Array array_type; ///< The \c Array type
      typedef TiledArray::detail::numeric_t<Array>
          numeric_type; ///< Array base numeric type
      typedef TiledArray::detail::scalar_t<Array>
          scalar_type; ///< Array base scalar type
      typedef TsrEngine<Array> engine_type; ///< Expression engine type
    };

    template <typename Array>
    struct ExprTrait<TsrExpr<const Array> > {
      typedef Array array_type; ///< The \c Array type
      typedef TiledArray::detail::numeric_t<Array>
          numeric_type; ///< Array base numeric type
      typedef TiledArray::detail::scalar_t<Array>
          scalar_type; ///< Array base scalar type
      typedef TsrEngine<Array> engine_type; ///< Expression engine type
    };

    /// Expression wrapper for array objects

    /// \tparam Array The \c TiledArray::Array type
    template <typename Array>
    class TsrExpr : public Expr<TsrExpr<Array> > {
    public:
      typedef TsrExpr<Array> TsrExpr_; ///< This class type
      typedef Expr<TsrExpr_> Expr_; ///< Base class type
      typedef typename ExprTrait<TsrExpr_>::array_type array_type; ///< The array type
      typedef typename ExprTrait<TsrExpr_>::engine_type engine_type; ///< Expression engine type

    private:

      array_type& array_; ///< The array that this expression
      std::string vars_; ///< The tensor variable list

    public:

      /// Constructor

      /// \param array The array object
      /// \param vars The variable list that is associated with this expression
      TsrExpr(array_type& array, const std::string& vars) :
        array_(array), vars_(vars)
      { }

      /// Copy constructor

      /// \param other The expression to be copied
      TsrExpr(const TsrExpr_& other) :
        array_(other.array_), vars_(other.vars_)
      { }

      /// Expression assignment operator

      /// \param other The expression that will be assigned to this array
      TsrExpr_& operator=(TsrExpr_& other) {
        other.eval_to(*this);
        return *this;
      }

      /// Expression assignment operator

      /// \tparam D The derived expression type
      /// \param other The expression that will be assigned to this array
      template <typename D>
      TsrExpr_& operator=(const Expr<D>& other) {
        other.derived().eval_to(*this);
        return *this;
      }

      /// Expression plus-assignment operator

      /// \tparam D The derived expression type
      /// \param other The expression that will be added to this array
      template <typename D>
      TsrExpr_& operator+=(const Expr<D>& other) {
        return operator=(AddExpr<TsrExpr_, D>(*this, other.derived()));
      }

      /// Expression minus-assignment operator

      /// \tparam D The derived expression type
      /// \param other The expression that will be subtracted from this array
      template <typename D>
      TsrExpr_& operator-=(const Expr<D>& other) {
        return operator=(SubtExpr<TsrExpr_, D>(*this, other.derived()));
      }

      /// Expression multiply-assignment operator

      /// \tparam D The derived expression type
      /// \param other The expression that will scale this array
      template <typename D>
      TsrExpr_& operator*=(const Expr<D>& other) {
        return operator=(MultExpr<TsrExpr_, D>(*this, other.derived()));
      }

      /// Array accessor

      /// \return a const reference to this array
      array_type& array() const { return array_; }

      /// Block expression

      /// \tparam Index The bound index types
      /// \param lower_bound The lower_bound of the block
      /// \param upper_bound The upper_bound of the block
      template <typename Index>
      BlkTsrExpr<const Array>
      block(const Index& lower_bound, const Index& upper_bound) const {
        return BlkTsrExpr<const Array>(*this, lower_bound, upper_bound);
      }

      /// Block expression

      /// \param lower_bound The lower_bound of the block
      /// \param upper_bound The upper_bound of the block
      BlkTsrExpr<const Array>
      block(const std::initializer_list<std::size_t>& lower_bound,
          const std::initializer_list<std::size_t>& upper_bound) const {
        return BlkTsrExpr<const Array>(*this, lower_bound, upper_bound);
      }

      /// Block expression

      /// \tparam Index The bound index types
      /// \param lower_bound The lower_bound of the block
      /// \param upper_bound The upper_bound of the block
      template <typename Index>
      BlkTsrExpr<Array>
      block(const Index& lower_bound, const Index& upper_bound) {
        return BlkTsrExpr<Array>(array_, vars_, lower_bound, upper_bound);
      }

      /// Block expression

      /// \param lower_bound The lower_bound of the block
      /// \param upper_bound The upper_bound of the block
      BlkTsrExpr<Array>
      block(const std::initializer_list<std::size_t>& lower_bound,
          const std::initializer_list<std::size_t>& upper_bound) {
        return BlkTsrExpr<Array>(array_, vars_, lower_bound, upper_bound);
      }

      /// Conjugated-tensor expression factor

      /// \return A conjugated expression object
      ConjTsrExpr<Array> conj() const {
        return ConjTsrExpr<Array>(array_, vars_, conj_op());
      }

      /// Tensor variable string accessor

      /// \return A const reference to the variable string for this tensor
      const std::string& vars() const { return vars_; }

    }; // class TsrExpr


    /// Expression wrapper for const array objects

    /// \tparam A The \c TiledArray::Array type
    template <typename Array>
    class TsrExpr<const Array> :
        public Expr<TsrExpr<const Array> >
    {
    public:
      typedef TsrExpr<const Array> TsrExpr_; ///< This class type
      typedef Expr<TsrExpr_> Expr_; ///< Expression base type
      typedef typename ExprTrait<TsrExpr_>::array_type array_type; ///< The array type
      typedef typename ExprTrait<TsrExpr_>::engine_type engine_type; ///< Expression engine type

    private:

      const array_type& array_; ///< The array that this expression
      std::string vars_; ///< The tensor variable string

      // Not allowed
      TsrExpr_& operator=(TsrExpr_&);

    public:

      /// Constructor

      /// \param array The array object
      /// \param vars The variable list that is associated with this expression
      TsrExpr(const array_type& array, const std::string& vars) :
        Expr_(), array_(array), vars_(vars) { }

      /// Copy constructor

      /// \param other The expression to be copied
      TsrExpr(const TsrExpr_& other) : array_(other.array_), vars_(other.vars_) { }

      /// Copy conversion

      /// \param other The expression to be copied
      explicit TsrExpr(const TsrExpr<array_type>& other) :
        array_(other.array()), vars_(other.vars_)
      { }

      /// Array accessor

      /// \return a const reference to this array
      const array_type& array() const { return array_; }

      /// Block expression

      /// \tparam Index The bound index types
      /// \param lower_bound The lower_bound of the block
      /// \param upper_bound The upper_bound of the block
      template <typename Index>
      BlkTsrExpr<const Array>
      block(const Index& lower_bound, const Index& upper_bound) const {
        return BlkTsrExpr<const Array>(array_, vars_, lower_bound, upper_bound);
      }

      /// Block expression

      /// \tparam Index The bound index types
      /// \param lower_bound The lower_bound of the block
      /// \param upper_bound The upper_bound of the block
      template <typename Index>
      BlkTsrExpr<const Array>
      block(const std::initializer_list<Index>& lower_bound,
          const std::initializer_list<Index>& upper_bound) const {
        return BlkTsrExpr<const Array>(array_, vars_, lower_bound, upper_bound);
      }

      /// Conjugated-tensor expression factor

      /// \return A conjugated expression object
      ConjTsrExpr<Array> conj() const {
        return ConjTsrExpr<Array>(array_, vars_, conj_op());
      }


      /// Tensor variable string accessor

      /// \return A const reference to the variable string for this tensor
      const std::string& vars() const { return vars_; }

    }; // class TsrExpr<const A>

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_TSR_EXPR_H__INCLUDED
