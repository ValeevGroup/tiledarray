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

    template <typename E>
    struct is_aliased : public std::true_type { };

    template <typename Array, bool Alias>
    struct is_aliased<TsrExpr<Array, Alias> > :
        public std::integral_constant<bool, Alias> { };

    template <typename Array, bool Alias>
    struct ExprTrait<TsrExpr<Array, Alias> > {
      typedef Array array_type; ///< The \c Array type
      typedef TiledArray::detail::numeric_t<Array>
          numeric_type; ///< Array base numeric type
      typedef TiledArray::detail::scalar_t<Array>
          scalar_type; ///< Array base scalar type
      typedef TsrEngine<Array, typename Array::eval_type, Alias>
          engine_type; ///< Expression engine type
    };

    template <typename Array>
    struct ExprTrait<TsrExpr<const Array, true> > {
      typedef Array array_type; ///< The \c Array type
      typedef TiledArray::detail::numeric_t<Array>
          numeric_type; ///< Array base numeric type
      typedef TiledArray::detail::scalar_t<Array>
          scalar_type; ///< Array base scalar type
      typedef TsrEngine<Array, typename Array::eval_type, true>
          engine_type; ///< Expression engine type
    };


    // This is here to catch errors in expression types. It should not be
    // possible to construct this type.
    template <typename Array>
    struct ExprTrait<TsrExpr<const Array, false> >; // <----- This should never happen!

    /// Expression wrapper for array objects

    /// \tparam Array The \c TiledArray::Array type
    /// \tparam Alias Indicates the array tiles should be computed as a
    /// temporary before assignment
    template <typename Array, bool Alias>
    class TsrExpr : public Expr<TsrExpr<Array, Alias> > {
    public:
      typedef TsrExpr<Array, Alias> TsrExpr_; ///< This class type
      typedef Expr<TsrExpr_> Expr_; ///< Base class type
      typedef typename ExprTrait<TsrExpr_>::array_type
          array_type; ///< The array type
      typedef typename ExprTrait<TsrExpr_>::engine_type
          engine_type; ///< Expression engine type

    private:

      array_type& array_; ///< The array that this expression
      std::string vars_; ///< The tensor variable list

    public:

      // Compiler generated functions
      TsrExpr() = default;
      TsrExpr(const TsrExpr_&) = default;
      TsrExpr(TsrExpr_&&) = default;
      ~TsrExpr() = default;

      /// Constructor

      /// \param array The array object
      /// \param vars The variable list that is associated with this expression
      TsrExpr(array_type& array, const std::string& vars) :
        array_(array), vars_(vars)
      { }

      /// Expression assignment operator

      /// \param other The expression that will be assigned to this array
      array_type& operator=(TsrExpr_& other) {
        other.eval_to(*this);
        return array_;
      }

      /// Expression assignment operator

      /// \tparam D The derived expression type
      /// \param other The expression that will be assigned to this array
      template <typename D>
      array_type& operator=(const Expr<D>& other) {
        static_assert(TiledArray::expressions::is_aliased<D>::value,
            "no_alias() expressions are not allowed on the right-hand side of "
            "the assignment operator.");
        other.derived().eval_to(*this);
        return array_;
      }

      /// Expression plus-assignment operator

      /// \tparam D The derived expression type
      /// \param other The expression that will be added to this array
      template <typename D>
      array_type& operator+=(const Expr<D>& other) {
        static_assert(TiledArray::expressions::is_aliased<D>::value,
            "no_alias() expressions are not allowed on the right-hand side of "
            "the assignment operator.");
        return operator=(AddExpr<TsrExpr_, D>(*this, other.derived()));
      }

      /// Expression minus-assignment operator

      /// \tparam D The derived expression type
      /// \param other The expression that will be subtracted from this array
      template <typename D>
      array_type& operator-=(const Expr<D>& other) {
        static_assert(TiledArray::expressions::is_aliased<D>::value,
            "no_alias() expressions are not allowed on the right-hand side of "
            "the assignment operator.");
        return operator=(SubtExpr<TsrExpr_, D>(*this, other.derived()));
      }

      /// Expression multiply-assignment operator

      /// \tparam D The derived expression type
      /// \param other The expression that will scale this array
      template <typename D>
      array_type& operator*=(const Expr<D>& other) {
        static_assert(TiledArray::expressions::is_aliased<D>::value,
            "no_alias() expressions are not allowed on the right-hand side of "
            "the assignment operator.");
        return operator=(MultExpr<TsrExpr_, D>(*this, other.derived()));
      }

      /// Array accessor

      /// \return a const reference to this array
      array_type& array() const { return array_; }

      /// Flag this tensor expression for a non-aliasing assignment

      /// \return A non-aliased tensor expression
      TsrExpr<Array, false>
      no_alias() const {
        return TsrExpr<Array, false>(array_, vars_);
      }

      /// Block expression

      /// \tparam Index The bound index types
      /// \param lower_bound The lower_bound of the block
      /// \param upper_bound The upper_bound of the block
      template <typename Index>
      BlkTsrExpr<const Array, Alias>
      block(const Index& lower_bound, const Index& upper_bound) const {
        return BlkTsrExpr<const Array, Alias>(*this, lower_bound,
            upper_bound);
      }

      /// Block expression

      /// \param lower_bound The lower_bound of the block
      /// \param upper_bound The upper_bound of the block
      BlkTsrExpr<const Array, Alias>
      block(const std::initializer_list<std::size_t>& lower_bound,
          const std::initializer_list<std::size_t>& upper_bound) const {
        return BlkTsrExpr<const Array, Alias>(*this, lower_bound,
            upper_bound);
      }

      /// Block expression

      /// \tparam Index The bound index types
      /// \param lower_bound The lower_bound of the block
      /// \param upper_bound The upper_bound of the block
      template <typename Index>
      BlkTsrExpr<Array, Alias>
      block(const Index& lower_bound, const Index& upper_bound) {
        return BlkTsrExpr<Array, Alias>(array_, vars_, lower_bound,
            upper_bound);
      }

      /// Block expression

      /// \param lower_bound The lower_bound of the block
      /// \param upper_bound The upper_bound of the block
      BlkTsrExpr<Array, Alias>
      block(const std::initializer_list<std::size_t>& lower_bound,
          const std::initializer_list<std::size_t>& upper_bound) {
        return BlkTsrExpr<Array, Alias>(array_, vars_, lower_bound,
            upper_bound);
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
    class TsrExpr<const Array, true> :
        public Expr<TsrExpr<const Array, true> >
    {
    public:
      typedef TsrExpr<const Array, true> TsrExpr_; ///< This class type
      typedef Expr<TsrExpr_> Expr_; ///< Expression base type
      typedef typename ExprTrait<TsrExpr_>::array_type array_type; ///< The array type
      typedef typename ExprTrait<TsrExpr_>::engine_type engine_type; ///< Expression engine type

    private:

      const array_type& array_; ///< The array that this expression
      std::string vars_; ///< The tensor variable string

      // Not allowed
      TsrExpr_& operator=(TsrExpr_&);

    public:

      // Compiler generated functions
      TsrExpr(const TsrExpr_&) = default;
      TsrExpr(TsrExpr_&&) = default;
      ~TsrExpr() = default;
      TsrExpr_& operator=(const TsrExpr_&) = delete;
      TsrExpr_& operator=(TsrExpr_&&) = delete;

      /// Constructor

      /// \param array The array object
      /// \param vars The variable list that is associated with this expression
      TsrExpr(const array_type& array, const std::string& vars) :
        Expr_(), array_(array), vars_(vars)
      { }

      /// Array accessor

      /// \return a const reference to this array
      const array_type& array() const { return array_; }

      /// Block expression

      /// \tparam Index The bound index types
      /// \param lower_bound The lower_bound of the block
      /// \param upper_bound The upper_bound of the block
      template <typename Index>
      BlkTsrExpr<const Array, true>
      block(const Index& lower_bound, const Index& upper_bound) const {
        return BlkTsrExpr<const Array, true>(array_, vars_, lower_bound,
            upper_bound);
      }

      /// Block expression

      /// \tparam Index The bound index types
      /// \param lower_bound The lower_bound of the block
      /// \param upper_bound The upper_bound of the block
      template <typename Index>
      BlkTsrExpr<const Array, true>
      block(const std::initializer_list<Index>& lower_bound,
          const std::initializer_list<Index>& upper_bound) const {
        return BlkTsrExpr<const Array, true>(array_, vars_, lower_bound,
            upper_bound);
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
