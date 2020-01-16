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
 */

#ifndef TILEDARRAY_SHAPE_H__INCLUDED
#define TILEDARRAY_SHAPE_H__INCLUDED

#include <TiledArray/dense_shape.h>
#include <TiledArray/sparse_shape.h>

namespace TiledArray {

template <typename, typename>
class DistArray;
class DensePolicy;

/// Type trait to detect dense shape types
template <typename S>
struct is_dense : public std::false_type {};

template <>
struct is_dense<DenseShape> : public std::true_type {};

template <>
struct is_dense<DensePolicy> : public std::true_type {};

template <typename Tile, typename Policy>
struct is_dense<DistArray<Tile, Policy> >
    : public is_dense<typename DistArray<Tile, Policy>::shape_type> {};

template <typename T>
constexpr const bool is_dense_v = is_dense<T>::value;

}  // namespace TiledArray

#endif  // TILEDARRAY_SHAPE_H__INCLUDED
