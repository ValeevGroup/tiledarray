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
 *  utility.h
 *  Oct 18, 2013
 *
 */

#ifndef TILEDARRAY_UTILITY_H__INCLUDED
#define TILEDARRAY_UTILITY_H__INCLUDED

#include <array>
#include <atomic>
#include <initializer_list>
#include <iosfwd>
#include <iterator>
#include <vector>

#include <TiledArray/type_traits.h>
#include <TiledArray/util/container.h>

namespace TiledArray {
namespace detail {

/// Print the content of an array like object

/// \tparam A The array container type
/// \param out A standard output stream
/// \param a The array-like container to be printed
/// \param n The number of elements in the array.
template <typename A>
inline void print_array(std::ostream& out, const A& a, const std::size_t n) {
  out << "[";
  for (std::size_t i = 0; i < n; ++i) {
    out << a[i];
    if (i != (n - 1)) out << ",";
  }
  out << "]";
}

/// Print the content of an array like object

/// \tparam A The array container type
/// \param out A standard output stream
/// \param a The array-like container to be printed
template <typename A>
inline void print_array(std::ostream& out, const A& a) {
  using std::size;
  print_array(out, a, size(a));
}

inline std::atomic<bool>& ignore_tile_position_accessor() {
  static std::atomic<bool> val{false};
  return val;
}
}  // namespace detail

/// Controls whether tile positions are checked in binary array operations.
/// These checks are disabled if preprocessor symbol \c NDEBUG is defined.
/// By default, tile positions are checked.
/// \param[in] b if true, tile positions will be ignored in binary array
///            operations.
/// \warning this function should be called following a fence
///          from the main thread only.
inline void ignore_tile_position(bool b) {
  detail::ignore_tile_position_accessor() = b;
}

/// Reports whether tile positions are checked in binary array operations.
/// These checks are disabled if preprocessor symbol \c NDEBUG is defined.
/// By default, tile positions are checked.
/// \return if true, tile positions will be ignored in binary array
///         operations.
inline bool ignore_tile_position() {
  return detail::ignore_tile_position_accessor();
}

}  // namespace TiledArray

namespace std {

/// Vector output stream operator
template <typename T, typename A>
inline std::ostream& operator<<(std::ostream& os,
                                const std::vector<T, A>& vec) {
  TiledArray::detail::print_array(os, vec);
  return os;
}

}  // namespace std

namespace boost {
namespace container {

/// Vector output stream operator
template <typename T, std::size_t N>
inline std::ostream& operator<<(
    std::ostream& os, const boost::container::small_vector<T, N>& vec) {
  TiledArray::detail::print_array(os, vec);
  return os;
}

}  // namespace container
}  // namespace boost

#endif  // TILEDARRAY_UTILITY_H__INCLUDED
