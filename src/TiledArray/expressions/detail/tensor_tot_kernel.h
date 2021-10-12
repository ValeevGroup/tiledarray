/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_DETAIL_TENSOR_TOT_KERNEL_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_DETAIL_TENSOR_TOT_KERNEL_H__INCLUDED

#include <tuple>                         // std::tuple
#include "TiledArray/expressions/fwd.h"  // TsrExpr<T>
#include "TiledArray/util/index.h"       // Index

namespace TiledArray::expressions::detail {

template <typename ArrayA, typename ArrayB, typename... Indices>
auto tensor_tot_kernel(TsrExpr<ArrayA> A, TsrExpr<ArrayB> B,
                       std::tuple<Index, Indices...> cs, World &world) {
  return B.array();
}

}  // namespace TiledArray::expressions::detail

#endif
