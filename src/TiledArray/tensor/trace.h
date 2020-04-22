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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  kernels.h
 *  Jun 1, 2015
 *
 */

#ifndef TILEDARRAY_TRACE_H__INCLUDED
#define TILEDARRAY_TRACE_H__INCLUDED

#include <type_traits> // enable_if, true_type, false_type

namespace TiledArray {
namespace detail {

/// Struct for determining if the trace of a tile of type \c T is defined.
///
/// This struct allows us to determine at compile-time whether or not we can
/// take the trace of a particular tile type \c T. This struct contains a static
/// bool member `value`, which will be true if we have a routine for taking the
/// trace of a tile of type \c T and false otherwise. When adding a new tile
/// type to TiledArray you are responsible for specializing TraceIsDefined for
/// your tile type.
///
/// \tparam T The type of the tile we are trying to take the trace of.
/// \tparam Enabler An extra template parameter, which can be used in
///                 specializations for SFINAE.
template <typename T, typename Enabler = void>
struct TraceIsDefined : std::false_type {};

/// Helper variable for determining if the trace operation is defined for a tile
/// of type \c T
///
/// This global variable is provided as a convenience for checking the value of
/// `TraceIsDefined<T>::value`.
///
/// \tparam T The type of the tile we are trying to take the trace of.
template <typename T>
constexpr auto trace_is_defined_v = TraceIsDefined<T>::value;

/// SFINAE type for enabling code when the trace operation is defined.
template <typename T>
using enable_if_trace_is_defined_t = std::enable_if<trace_is_defined_v<T>>;

/// Specialization of this class registers the routine for taking the trace of a
/// particular tile type.
///
/// \tparam TileType The type of the tile we are taking the trace of.
/// \tparam Enabler A template type parameter which can be used to selectively
///                 enable a specialization. Defaults to void.
template <typename TileType, typename Enabler = void>
struct Trace;

} // namespace detail

/// Helper function for taking the trace of a tensor
template<typename T, typename = detail::enable_if_trace_is_defined_t<T>>
decltype(auto) trace(const T& t){
  detail::Trace<T> tracer;
  return tracer(t);
}

/// Helper type for determining the result of taking the trace of a tensor
template<typename T>
using result_of_trace_t = decltype(trace(std::declval<T>()));

} // namespace TiledArray
#endif // TILEDARRAY_TRACE_H__INCLUDED
