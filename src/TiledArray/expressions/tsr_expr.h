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
#include <TiledArray/expressions/blk_tsr_expr.h>
#include <TiledArray/expressions/mult_expr.h>
#include <TiledArray/expressions/scal_tsr_expr.h>
#include <TiledArray/expressions/subt_expr.h>
#include <TiledArray/expressions/tsr_engine.h>

namespace TiledArray {
namespace expressions {

using TiledArray::detail::numeric_t;
using TiledArray::detail::scalar_t;

template <typename Array, bool Alias>
struct ExprTrait<TsrExpr<Array, Alias>> {
  typedef Array array_type;  ///< The \c Array type
  typedef TiledArray::detail::numeric_t<Array>
      numeric_type;  ///< Array base numeric type
  typedef TiledArray::detail::scalar_t<Array>
      scalar_type;  ///< Array base scalar type
  typedef TsrEngine<Array, typename Array::eval_type, Alias>
      engine_type;  ///< Expression engine type
};

template <typename Array>
struct ExprTrait<TsrExpr<const Array, true>> {
  typedef Array array_type;  ///< The \c Array type
  typedef TiledArray::detail::numeric_t<Array>
      numeric_type;  ///< Array base numeric type
  typedef TiledArray::detail::scalar_t<Array>
      scalar_type;  ///< Array base scalar type
  typedef TsrEngine<Array, typename Array::eval_type, true>
      engine_type;  ///< Expression engine type
};

// This is here to catch errors in expression types. It should not be
// possible to construct this type.
template <typename Array>
struct ExprTrait<TsrExpr<const Array, false>>;  // <----- This should never
                                                // happen!

/// Expression wrapper for array objects

/// \tparam Array The \c TiledArray::Array type
/// \tparam Alias If true, the array tiles should be evaluated as
/// temporaries before assignment; if false, can reuse the result tiles
template <typename Array, bool Alias>
class TsrExpr : public Expr<TsrExpr<Array, Alias>> {
 public:
  typedef TsrExpr<Array, Alias> TsrExpr_;  ///< This class type
  typedef Expr<TsrExpr_> Expr_;            ///< Base class type
  typedef
      typename ExprTrait<TsrExpr_>::array_type array_type;  ///< The array type
  typedef typename ExprTrait<TsrExpr_>::engine_type
      engine_type;  ///< Expression engine type
  using index1_type = TA_1INDEX_TYPE;

 private:
  array_type& array_;       ///< The array that this expression is bound to
  std::string annotation_;  ///< The array annotation

 public:
  // Compiler generated functions
  TsrExpr() = default;
  TsrExpr(const TsrExpr_&) = default;
  TsrExpr(TsrExpr_&&) = default;
  ~TsrExpr() = default;

  /// Constructor

  /// \param array The array object
  /// \param annotation The annotation for \p array
  TsrExpr(array_type& array, const std::string& annotation)
      : array_(array), annotation_(annotation) {}

  operator TsrExpr<const Array>() const {
    return TsrExpr<const Array>(this->array(), this->annotation());
  }

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
    static_assert(
        TiledArray::expressions::is_aliased<D>::value,
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
    static_assert(
        TiledArray::expressions::is_aliased<D>::value,
        "no_alias() expressions are not allowed on the right-hand side of "
        "the assignment operator.");
    return operator=(AddExpr<TsrExpr_, D>(*this, other.derived()));
  }

  /// Expression minus-assignment operator

  /// \tparam D The derived expression type
  /// \param other The expression that will be subtracted from this array
  template <typename D>
  array_type& operator-=(const Expr<D>& other) {
    static_assert(
        TiledArray::expressions::is_aliased<D>::value,
        "no_alias() expressions are not allowed on the right-hand side of "
        "the assignment operator.");
    return operator=(SubtExpr<TsrExpr_, D>(*this, other.derived()));
  }

  /// Expression multiply-assignment operator

  /// \tparam D The derived expression type
  /// \param other The expression that will scale this array
  template <typename D>
  array_type& operator*=(const Expr<D>& other) {
    static_assert(
        TiledArray::expressions::is_aliased<D>::value,
        "no_alias() expressions are not allowed on the right-hand side of "
        "the assignment operator.");
    return operator=(MultExpr<TsrExpr_, D>(*this, other.derived()));
  }

  /// Array accessor

  /// \return a const reference to this array
  array_type& array() const { return array_; }

  /// Flag this tensor expression for a non-aliasing assignment

  /// \return A non-aliased tensor expression
  TsrExpr<Array, false> no_alias() const {
    return TsrExpr<Array, false>(array_, annotation_);
  }

  /// immutable Block expression factory

  /// \tparam Index1 An integral range type
  /// \tparam Index2 An integral range type
  /// \param lower_bound The lower_bound of the block
  /// \param upper_bound The upper_bound of the block
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<
                TiledArray::detail::is_integral_range_v<Index1> &&
                TiledArray::detail::is_integral_range_v<Index2>>>
  BlkTsrExpr<const Array, Alias> block(const Index1& lower_bound,
                                       const Index2& upper_bound) const {
    return BlkTsrExpr<const Array, Alias>(array_, annotation_, lower_bound,
                                          upper_bound);
  }

  /// immutable Block expression factory

  /// \tparam Index1 An integral type
  /// \tparam Index2 An integral type
  /// \param lower_bound The lower_bound of the block
  /// \param upper_bound The upper_bound of the block
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2>>>
  BlkTsrExpr<const Array, Alias> block(
      const std::initializer_list<Index1>& lower_bound,
      const std::initializer_list<Index2>& upper_bound) const {
    return BlkTsrExpr<const Array, Alias>(array_, annotation_, lower_bound,
                                          upper_bound);
  }

  /// immutable Block expression factory

  /// \tparam PairRange Type representing a range of generalized pairs (see
  /// TiledArray::detail::is_gpair_v ) \param bounds The {lower,upper} bounds of
  /// the block
  template <typename PairRange,
            typename = std::enable_if_t<
                TiledArray::detail::is_gpair_range_v<PairRange>>>
  BlkTsrExpr<const Array, Alias> block(const PairRange& bounds) const {
    return BlkTsrExpr<const Array, Alias>(array_, annotation_, bounds);
  }

  /// immutable Block expression factory

  /// \tparam Index An integral type
  /// \param bounds The {lower,upper} bounds of the block
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  BlkTsrExpr<const Array, Alias> block(
      const std::initializer_list<std::initializer_list<Index>>& bounds) const {
    return BlkTsrExpr<const Array, Alias>(array_, annotation_, bounds);
  }

  /// mutable Block expression factory

  /// \tparam Index1 An integral range type
  /// \tparam Index2 An integral range type
  /// \param lower_bound The lower_bound of the block
  /// \param upper_bound The upper_bound of the block
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<
                TiledArray::detail::is_integral_range_v<Index1> &&
                TiledArray::detail::is_integral_range_v<Index2>>>
  BlkTsrExpr<Array, Alias> block(const Index1& lower_bound,
                                 const Index2& upper_bound) {
    return BlkTsrExpr<Array, Alias>(array_, annotation_, lower_bound,
                                    upper_bound);
  }

  /// mutable Block expression factory

  /// \tparam Index1 An integral type
  /// \tparam Index2 An integral type
  /// \param lower_bound The lower_bound of the block
  /// \param upper_bound The upper_bound of the block
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2>>>
  BlkTsrExpr<Array, Alias> block(
      const std::initializer_list<Index1>& lower_bound,
      const std::initializer_list<Index2>& upper_bound) {
    return BlkTsrExpr<Array, Alias>(array_, annotation_, lower_bound,
                                    upper_bound);
  }

  /// mutable Block expression factory

  /// \tparam PairRange Type representing a range of generalized pairs (see
  /// TiledArray::detail::is_gpair_v ) \param bounds The {lower,upper} bounds of
  /// the block
  template <typename PairRange,
            typename = std::enable_if_t<
                TiledArray::detail::is_gpair_range_v<PairRange>>>
  BlkTsrExpr<Array, Alias> block(const PairRange& bounds) {
    return BlkTsrExpr<Array, Alias>(array_, annotation_, bounds);
  }

  /// mutable Block expression factory

  /// \tparam Index An integral type
  /// \param bounds The {lower,upper} bounds of the block
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  BlkTsrExpr<Array, Alias> block(
      const std::initializer_list<std::initializer_list<Index>>& bounds) {
    return BlkTsrExpr<Array, Alias>(array_, annotation_, bounds);
  }

  /// Conjugated-tensor expression factor

  /// \return A conjugated expression object
  ConjTsrExpr<Array> conj() const {
    return ConjTsrExpr<Array>(array_, annotation_, conj_op());
  }

  /// Tensor annotation accessor

  /// \return A const reference to the annotation for this tensor
  const std::string& annotation() const { return annotation_; }

};  // class TsrExpr

/// Expression wrapper for const array objects

/// \tparam A The \c TiledArray::Array type
template <typename Array>
class TsrExpr<const Array, true> : public Expr<TsrExpr<const Array, true>> {
 public:
  typedef TsrExpr<const Array, true> TsrExpr_;  ///< This class type
  typedef Expr<TsrExpr_> Expr_;                 ///< Expression base type
  typedef
      typename ExprTrait<TsrExpr_>::array_type array_type;  ///< The array type
  typedef typename ExprTrait<TsrExpr_>::engine_type
      engine_type;  ///< Expression engine type
  using index1_type = TA_1INDEX_TYPE;

 private:
  const array_type& array_;  ///< The array that this expression is bound to
  std::string annotation_;   ///< The array annotation

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
  /// \param annotation The annotation for \p array
  TsrExpr(const array_type& array, const std::string& annotation)
      : Expr_(), array_(array), annotation_(annotation) {}

  /// Array accessor

  /// \return a const reference to this array
  const array_type& array() const { return array_; }

  /// Block expression

  /// \tparam Index1 An integral range type
  /// \tparam Index2 An integral range type
  /// \param lower_bound The lower_bound of the block
  /// \param upper_bound The upper_bound of the block
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<
                TiledArray::detail::is_integral_range_v<Index1> &&
                TiledArray::detail::is_integral_range_v<Index2>>>
  BlkTsrExpr<const Array, true> block(const Index1& lower_bound,
                                      const Index2& upper_bound) const {
    return BlkTsrExpr<const Array, true>(array_, annotation_, lower_bound,
                                         upper_bound);
  }

  /// Block expression

  /// \tparam Index1 An integral type
  /// \tparam Index2 An integral type
  /// \tparam Index The bound index types
  /// \param lower_bound The lower_bound of the block
  /// \param upper_bound The upper_bound of the block
  template <typename Index1, typename Index2,
            typename = std::enable_if_t<std::is_integral_v<Index1> &&
                                        std::is_integral_v<Index2>>>
  BlkTsrExpr<const Array, true> block(
      const std::initializer_list<Index1>& lower_bound,
      const std::initializer_list<Index2>& upper_bound) const {
    return BlkTsrExpr<const Array, true>(array_, annotation_, lower_bound,
                                         upper_bound);
  }

  /// Block expression

  /// \tparam PairRange Type representing a range of generalized pairs (see
  /// TiledArray::detail::is_gpair_v ) \param bounds The {lower,upper} bounds of
  /// the block
  template <typename PairRange,
            typename = std::enable_if_t<
                TiledArray::detail::is_gpair_range_v<PairRange>>>
  BlkTsrExpr<const Array, true> block(const PairRange& bounds) const {
    return BlkTsrExpr<const Array, true>(array_, annotation_, bounds);
  }

  /// Block expression

  /// \tparam Index An integral type
  /// \param bounds The {lower,upper} bounds of the block
  template <typename Index,
            typename = std::enable_if_t<std::is_integral_v<Index>>>
  BlkTsrExpr<const Array, true> block(
      const std::initializer_list<std::initializer_list<Index>>& bounds) const {
    return BlkTsrExpr<const Array, true>(array_, annotation_, bounds);
  }

  /// Conjugated-tensor expression factor

  /// \return A conjugated expression object
  ConjTsrExpr<Array> conj() const {
    return ConjTsrExpr<Array>(array_, annotation_, conj_op());
  }

  /// Tensor annotation accessor

  /// \return A const reference to the annotation for this tensor
  const std::string& annotation() const { return annotation_; }

};  // class TsrExpr<const A>

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_TSR_EXPR_H__INCLUDED
