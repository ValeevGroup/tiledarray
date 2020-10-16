/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2020 Virginia Tech
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
 *  David Williams-Young
 *  Computational Research Division, Lawrence Berkeley National Laboratory
 *
 *  svd.h
 *  Created:    12 June, 2020
 *
 */
#ifndef TILEDARRAY_ALGEBRA_SVD_UTILS_H__INCLUDED
#define TILEDARRAY_ALGEBRA_SVD_UTILS_H__INCLUDED

#include <TiledArray/config.h>
#include <type_traits>

namespace TiledArray {

struct SVDReturnType {};
struct SVDValuesOnly : public SVDReturnType {};
struct SVDLeftVectors : public SVDReturnType {};
struct SVDRightVectors : public SVDReturnType {};
struct SVDAllVectors : public SVDReturnType {};

namespace detail {

template <typename SVDType, typename = void>
struct is_svd_return_type : public std::false_type {};

template <typename SVDType>
struct is_svd_return_type<
    SVDType, std::enable_if_t<std::is_base_of_v<SVDReturnType, SVDType>>>
    : public std::true_type {};

template <typename SVDType>
inline constexpr bool is_svd_return_type_v = is_svd_return_type<SVDType>::value;

template <typename SVDType, typename U = void>
struct enable_if_svd_return_type
    : public std::enable_if<is_svd_return_type_v<SVDType>, U> {};

template <typename SVDType, typename U = void>
using enable_if_svd_return_type_t =
    typename enable_if_svd_return_type<SVDType, U>::type;

}  // namespace detail

}  // namespace TiledArray

#endif  // TILEDARRAY_ALGEBRA_SVD_UTILS_H__INCLUDED
