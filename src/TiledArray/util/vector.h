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
#include <vector>

#include <TiledArray/utility.h>
#include <madness/world/archive.h>

namespace TiledArray {

namespace container {

template <typename T>
using vector = std::vector<T>;
template <typename T, std::size_t N = 8>
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

}  // namespace container
}  // namespace TiledArray

namespace madness {
namespace archive {

template <class Archive, typename T, std::size_t N, typename A>
struct ArchiveLoadImpl<Archive, boost::container::small_vector<T, N, A>> {
  static inline void load(const Archive& ar,
                          boost::container::small_vector<T, N, A>& x) {
    std::size_t n{};
    ar& n;
    x.resize(n);
    for (auto& xi : x) ar& xi;
  }
};

template <class Archive, typename T, std::size_t N, typename A>
struct ArchiveStoreImpl<Archive, boost::container::small_vector<T, N, A>> {
  static inline void store(const Archive& ar,
                           const boost::container::small_vector<T, N, A>& x) {
    ar& x.size();
    for (const auto& xi : x) ar& xi;
  }
};

}  // namespace archive
}  // namespace madness

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

#endif  // TILEDARRAY_UTIL_VECTOR_H
