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
 *  expr.h
 *  Apr 1, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_FWD_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_FWD_H__INCLUDED

#include <type_traits>

namespace TiledArray::expressions {

template <typename>
class Expr;

template <typename, bool Alias = true>
class TsrExpr;

template <typename, bool>
class BlkTsrExpr;

template <typename, typename>
class ScalBlkTsrExpr;

template <typename>
struct is_aliased : std::true_type {};

template <typename Array>
struct is_aliased<TsrExpr<Array, false>> : std::false_type {};

// Forward declaration
template <typename>
struct ExprTrait;

template <typename, typename, bool>
class TsrEngine;

template <typename, typename, bool>
class BlkTsrEngine;

template <typename, typename, typename>
class ScalBlkTsrEngine;

template <typename, typename>
class ScalTsrExpr;

template <typename, typename, typename>
class ScalTsrEngine;

}  // namespace TiledArray::expressions

#endif  // TILEDARRAY_EXPRESSIONS_FWD_H__INCLUDED
