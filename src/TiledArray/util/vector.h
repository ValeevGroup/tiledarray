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

namespace TiledArray {

/// Vector output stream operator
template <typename T, typename A>
inline std::ostream& operator<<(std::ostream& os,
                                const std::vector<T, A>& vec) {
  TiledArray::detail::print_array(os, vec);
  return os;
}

}  // namespace TiledArray

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
