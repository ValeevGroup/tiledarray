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

namespace TiledArray {
  namespace expressions {

    template <typename Left, typename Right, template <typename, typename> class Engine>
    struct BinaryExprTrait {
      typedef Left left_type; ///< The left-hand expression type
      typedef Right right_type; ///< The right-hand expression type
      typedef Engine<typename Left::engine_type, typename Right::engine_type> engine_type; ///< Expression engine type
      typedef typename ExprTrait<Left>::scalar_type scalar_type;  ///< Tile scalar type
    };

    /// Binary expression object

    /// \tparam Derived The derived class type
    template <typename Derived>
    class BinaryExpr : public Expr<Derived> {
    public:
      typedef typename ExprTrait<Derived>::left_type left_type; ///< The left-hand expression type
      typedef typename ExprTrait<Derived>::right_type right_type; ///< The right-hand expression type

    private:

      left_type left_; ///< The left-hand argument
      right_type right_; ///< The right-hand argument

      // Not allowed
      BinaryExpr<Derived>& operator=(const BinaryExpr<Derived>&);

    public:

      /// Binary expression constructor
      BinaryExpr(const left_type& left, const right_type& right) :
        left_(left), right_(right)
      { }

      /// Copy constructor
      BinaryExpr(const BinaryExpr<Derived>& other) :
        left_(other.left_), right_(other.right_)
      { }

      /// Left-hand expression argument accessor

      /// \return A const reference to the left-hand expression object
      const left_type& left() const { return left_; }

      /// Right-hand expression argument accessor

      /// \return A const reference to the right-hand expression object
      const right_type& right() const { return right_; }

    }; // class BinaryExpr

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_BINARY_BASE_H__INCLUDED
