/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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

#ifndef TILEDARRAY_INITIALIZER_LIST_UTILS_H__INCLUDED
#define TILEDARRAY_INITIALIZER_LIST_UTILS_H__INCLUDED

#include <initializer_list>  // std::initializer_list
#include <type_traits>       // false_type, true_type

/** @file initializer_list_utils.h
 *
 *  initializer_list_utils.h contains routines and template meta-programming
 *  utilities for manipulating std::initializer_list instances.
 */

namespace TiledArray {

//------------------------------------------------------------------------------
// IsInitializerList Struct
//------------------------------------------------------------------------------

/** @brief Primary template for detecting if @p T is an std::initializer_list
 *
 *  This is the primary template for detecting if a type @p T is an
 *  std::initializer_list it will be selected when @p T is **NOT** an
 *  std::initializer_list and will contain a constexpr static member `value`,
 *  which is always false.
 *
 *  @tparam T The type we are testing for its initializer_list-ness
 */
template<typename T>
struct IsInitializerList : std::false_type {};

/** @brief Specialization of IsInitializerList for an std::initializer_list
 *
 *  This specialization is selected if the template type parameter to
 *  IsInitializerList is of the form `std::initializer_list<T>` and will contain
 *  a constexpr static member `value` which is always true.
 *
 *  @tparam T The type we are testing for its initializer_list-ness
 */
template<typename T>
struct IsInitializerList<std::initializer_list<T>> : std::true_type{};

/** @brief Helper variable template for the IsInitializerList struct.
 *
 *  This helper variable conforms to the STL's practice of declaring a helper
 *  variable to retrieve the static member `value` of a struct. The value of
 *  `is_initializer_list_v<T>` will be the same as `IsInitializerList<T>::value`
 *
 *  @tparam T The type we want to know the initializer_list-ness of.
 */
template<typename T>
static constexpr bool is_initializer_list_v = IsInitializerList<T>::value;

//------------------------------------------------------------------------------
// InitializerListRank Struct
//------------------------------------------------------------------------------

/** @brief Primary template for determining how many nested
 *         std::initializer_list's are in a type.
 *
 *  This is the primary template for determining how many`std::initializer_list`
 *  are in @p T. It is selected when @p T is **NOT** an `std::initializer_list`
 *  and will contain a static constexpr member `value` of type @p SizeType equal
 *  to 0.
 *
 *  @tparam T The type we are analyzing for its
 *          std::initializer_list-nested-ness
 *  @tparam SizeType the type to use for the `value` member. Defaults to
 *          `std::size_t`.
 */
template<typename T, typename SizeType = std::size_t>
struct InitializerListRank : std::integral_constant<SizeType, 0> {};

/** @brief Helper variable for retrieving the degree of nesting for an
 *         std::initializer_list.
 *
 *  This helper variable creates a global variable which contains the value
 *  of InitializerListRank<T, SizeType> and is intended to be used as a (more)
 *  convenient means of retrieving the value.
 *
 *  @tparam T The type we are analyzing for its
 *          std::initializer_list-nested-ness
 *  @tparam SizeType the type to use for the `value` member. Defaults to
 *          `std::size_t`.
 */
template<typename T, typename SizeType = std::size_t>
constexpr auto initializer_list_rank_v = InitializerListRank<T, SizeType>::value;

/** @brief Specialization of InitializerListRank used when the template type
 *         parameter is a std::initializer_list type
 *
 *  This specialization is selected when InitializerListRank is parameterized
 *  with an `std::initializer_list` and will contain a static constexpr member
 *  `value` of type @p SizeType equal to 1 plus the number of
 *  `std::initializer_list`s in @p T.
 *
 *  @tparam T The type we are analyzing for its
 *          std::initializer_list-nested-ness
 *  @tparam SizeType the type to use for the `value` member. Defaults to
 *          `std::size_t`.
 */
template<typename T, typename SizeType>
struct InitializerListRank<std::initializer_list<T>, SizeType> :
std::integral_constant<SizeType, initializer_list_rank_v<T, SizeType> + 1> {};

}  // namespace TiledArray

#endif // TILEDARRAY_INITIALIZER_LIST_UTILS_H__INCLUDED