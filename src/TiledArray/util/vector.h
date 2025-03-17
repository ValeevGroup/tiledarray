/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2020  Virginia Tech
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
 *  Eduard Valeyev
 *  Department of Chemistry, Virginia Tech
 *
 *  util/vector.h
 *  March 24, 2020
 *
 */

#ifndef TILEDARRAY_UTIL_VECTOR_H
#define TILEDARRAY_UTIL_VECTOR_H

#include <boost/container/small_vector.hpp>
#include <boost/version.hpp>

// Boost.Container 1.75 and earlier uses standard exception classes, 1.76+ use
// Boost.Container exceptions, unless BOOST_CONTAINER_USE_STD_EXCEPTIONS is
// defined:
// https://www.boost.org/doc/libs/master/doc/html/container/release_notes.html#container.release_notes.release_notes_boost_1_76_00
// Define BOOST_CONTAINER_USE_STD_EXCEPTIONS for Boost <1.76 so that exception
// checking can use this macro with all versions of Boost
#if BOOST_VERSION < 107600 && !defined(BOOST_CONTAINER_USE_STD_EXCEPTIONS)
#define BOOST_CONTAINER_USE_STD_EXCEPTIONS 1
#endif

#include <vector>
#include "TiledArray/config.h"

#include <TiledArray/utility.h>
#include <madness/world/archive.h>
#include "TiledArray/error.h"

namespace TiledArray {

namespace container {

template <typename T>
using vector = std::vector<T>;
template <typename T, std::size_t N = TA_MAX_SOO_RANK_METADATA>
using svector = boost::container::small_vector<T, N>;

template <typename Range>
std::enable_if_t<detail::is_integral_range_v<Range> &&
                     detail::is_sized_range_v<Range>,
                 svector<detail::value_t<Range>>>
iv(Range&& rng) {
  svector<detail::value_t<Range>> result(std::size(rng));
  long count = 0;
  for (auto&& v : rng) {
    result[count] = v;
    ++count;
  }
  return result;
}

template <typename Int, typename = std::enable_if_t<std::is_integral_v<Int>>>
svector<Int> iv(std::initializer_list<Int> list) {
  return svector<Int>(data(list), data(list) + size(list));
}

namespace detail {
template <typename Vec, typename T, typename... Ts>
void iv_assign(Vec& d, size_t i, T v, Ts... vrest) {
  d[i] = v;
  if constexpr (sizeof...(Ts) > 0) {
    iv_assign(d, i + 1, vrest...);
  }
}
}  // namespace detail

template <typename Int, typename... Ints,
          typename = std::enable_if_t<std::is_integral_v<Int> &&
                                      (std::is_integral_v<Ints> && ...)>>
constexpr auto iv(Int i0, Ints... rest) {
  constexpr const auto sz = sizeof...(Ints) + 1;
  svector<std::common_type_t<Int, Ints...>, sz> result(sz);
  detail::iv_assign(result, 0, i0, rest...);
  return result;
}

namespace operators {

template <typename T1, std::size_t N1, typename T2, std::size_t N2>
decltype(auto) operator+(const boost::container::small_vector<T1, N1>& v1,
                         const boost::container::small_vector<T2, N2>& v2) {
  TA_ASSERT(v1.size() == v2.size());
  boost::container::small_vector<std::common_type_t<T1, T2>, std::max(N1, N2)>
      result(v1.size());
  std::transform(v1.begin(), v1.end(), v2.begin(), result.begin(),
                 [](auto&& a, auto&& b) { return a + b; });
  return result;
}

template <typename T1, std::size_t N1, typename T2, std::size_t N2>
decltype(auto) operator-(const boost::container::small_vector<T1, N1>& v1,
                         const boost::container::small_vector<T2, N2>& v2) {
  TA_ASSERT(v1.size() == v2.size());
  boost::container::small_vector<std::common_type_t<T1, T2>, std::max(N1, N2)>
      result(v1.size());
  std::transform(v1.begin(), v1.end(), v2.begin(), result.begin(),
                 [](auto&& a, auto&& b) { return a - b; });
  return result;
}

}  // namespace operators

}  // namespace container
}  // namespace TiledArray

namespace TiledArray {

/// Vector output stream operator
template <typename Char, typename CharTraits, typename T, typename A>
inline std::basic_ostream<Char, CharTraits>& operator<<(
    std::basic_ostream<Char, CharTraits>& os, const std::vector<T, A>& vec) {
  TiledArray::detail::print_array(os, vec);
  return os;
}

}  // namespace TiledArray

namespace boost {
namespace container {

/// Vector output stream operator
template <typename Char, typename CharTraits, typename T, std::size_t N>
inline std::basic_ostream<Char, CharTraits>& operator<<(
    std::basic_ostream<Char, CharTraits>& os,
    const boost::container::small_vector<T, N>& vec) {
  TiledArray::detail::print_array(os, vec);
  return os;
}

}  // namespace container
}  // namespace boost

#endif  // TILEDARRAY_UTIL_VECTOR_H
