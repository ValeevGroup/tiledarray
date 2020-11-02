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
 *  binary_expr.h
 *  Apr 1, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_BINARY_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_BINARY_H__INCLUDED

#include <TiledArray/expressions/expr.h>
#include <TiledArray/expressions/permopt.h>

namespace TiledArray {
namespace expressions {

/// Binary expression object

/// \tparam Derived The derived class type
template <typename Derived>
class BinaryExpr : public Expr<Derived> {
 public:
  typedef BinaryExpr<Derived> BinaryExpr_;  ///< This class type
  typedef typename ExprTrait<Derived>::left_type
      left_type;  ///< The left-hand expression type
  typedef typename ExprTrait<Derived>::right_type
      right_type;  ///< The right-hand expression type

 private:
  left_type left_;    ///< The left-hand argument
  right_type right_;  ///< The right-hand argument

 public:
  // Compiler generated functions
  BinaryExpr(const BinaryExpr_&) = default;
  BinaryExpr(BinaryExpr_&&) = default;
  ~BinaryExpr() = default;
  BinaryExpr_& operator=(const BinaryExpr_&) = delete;
  BinaryExpr_& operator=(BinaryExpr_&&) = delete;

  /// Binary expression constructor
  BinaryExpr(const left_type& left, const right_type& right)
      : left_(left), right_(right) {}

  /// Left-hand expression argument accessor

  /// \return A const reference to the left-hand expression object
  const left_type& left() const { return left_; }

  /// Right-hand expression argument accessor

  /// \return A const reference to the right-hand expression object
  const right_type& right() const { return right_; }

};  // class BinaryExpr

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_BINARY_BASE_H__INCLUDED
