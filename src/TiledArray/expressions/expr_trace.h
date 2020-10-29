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
 *  expr_trace.h
 *  Mar 16, 2014
 *
 */

#ifndef TILEDARRAY_EXPR_TRACE_H__INCLUDED
#define TILEDARRAY_EXPR_TRACE_H__INCLUDED

#include <TiledArray/expressions/index_list.h>
#include <iostream>

namespace TiledArray {
namespace expressions {

template <typename>
class Expr;
template <typename, bool>
class TsrExpr;

/// Expression output stream
class ExprOStream {
  std::ostream& os_;  ///< output stream
  unsigned int tab_;  ///< Number of leading tabs

 public:
  /// Constructor

  /// \param os The output stream
  ExprOStream(std::ostream& os) : os_(os), tab_(0u) {}

  /// Copy constructor

  /// \param other The stream object to be copied
  ExprOStream(const ExprOStream& other) : os_(other.os_), tab_(other.tab_) {}

  /// Output operator

  /// \tparam T The object type
  /// \param t The object to be added to the stream
  template <typename T>
  std::ostream& operator<<(const T& t) {
    for (unsigned int i = 0u; i < tab_; ++i) {
      os_ << "  ";
    }

    os_ << t;
    return os_;
  }

  /// Increment the number of tabs
  void inc() { ++tab_; }

  /// Decrement the number of tabs
  void dec() { --tab_; }

  /// Output stream accessor
  std::ostream& get_stream() const { return os_; }

};  // class ExprOStream

/// Expression trace target

/// Wrapper object that helps start the expression
class ExprTraceTarget {
  std::ostream& os_;                   ///< Output stream
  BipartiteIndexList target_indices_;  ///< Target index list for an expression

 public:
  /// Constructor

  /// \param os Output stream
  /// \param target_annotation The target annotation for an expression
  ExprTraceTarget(std::ostream& os, const std::string& target_annotation)
      : os_(os), target_indices_(target_annotation) {}

  /// Copy constructor

  /// \param other The object to be copied
  ExprTraceTarget(const ExprTraceTarget& other)
      : os_(other.os_), target_indices_(other.target_indices_) {}

  /// Start the expression trace

  /// \tparam D the Expression type
  /// \param expr The expression to be printed
  /// \return The output stream
  template <typename D>
  std::ostream& operator<<(const Expr<D>& expr) const {
    if (TiledArray::get_default_world().rank() == 0) {
      os_ << target_indices_ << " =\n";

      ExprOStream expr_stream(os_);
      expr_stream.inc();
      expr.derived().print(expr_stream, target_indices_);
    }

    return os_;
  }

};  // class ExprTraceTarget

/// Expression trace factory function

/// \tparam A An \c Array object
/// \tparam Alias Tiles alias flag
/// \param os The output stream for the expression trace
/// \param tsr The tensor that will be the target of the expression
/// \return The expression trace object
template <typename A, bool Alias>
inline ExprTraceTarget operator<<(std::ostream& os,
                                  const TsrExpr<A, Alias>& tsr) {
  return ExprTraceTarget(os, tsr.annotation());
}

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPR_TRACE_H__INCLUDED
