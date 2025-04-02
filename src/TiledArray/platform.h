/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2018  Virginia Tech
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
 *  Mar 18, 2018
 *
 */

#ifndef TILEDARRAY_PLATFORM_H__INCLUDED
#define TILEDARRAY_PLATFORM_H__INCLUDED

#include <TiledArray/fwd.h>

#include <TiledArray/type_traits.h>

namespace TiledArray {

/// enumerates the memory spaces
enum class MemorySpace {
  // MemorySpace is represented as a bitfield to compute unions and
  // intersections easier
  Null = 0b00,
  Host = 0b01,
  Device = 0b10,
  Device_UM = Host | Device  // union of host and device spaces
};

// customization point: in_memory_space<S>(O) -> bool
// it can be used to query if object O is in space S

/// @return intersection of @c space1 and @c space2
constexpr MemorySpace operator&(MemorySpace space1, MemorySpace space2) {
  return static_cast<MemorySpace>(static_cast<int>(space1) &
                                  static_cast<int>(space2));
}
/// @return union of @c space1 and @c space2
constexpr MemorySpace operator|(MemorySpace space1, MemorySpace space2) {
  return static_cast<MemorySpace>(static_cast<int>(space1) |
                                  static_cast<int>(space2));
}
/// @return true if intersection of @c space1 and @c space2 is nonnull
constexpr bool overlap(MemorySpace space1, MemorySpace space2) {
  return (space1 & space2) != MemorySpace::Null;
}

// customization point: is_constexpr_size_of_v<S,T> reports whether
// size_of<S>(T) is the same for all T
template <MemorySpace S, typename T>
inline constexpr bool is_constexpr_size_of_v = detail::is_numeric_v<T>;

// customization point: size_of<S>(O) -> std::size_t reports the number of
// bytes occupied by O in S
template <MemorySpace S, typename T,
          typename = std::enable_if_t<is_constexpr_size_of_v<S, T>>>
constexpr std::size_t size_of(const T& t) {
  return sizeof(T);
}

// customization point: allocates_memory_space<S>(A) -> bool reports whether
// allocator A allocates memory in space S
template <MemorySpace S, typename T>
constexpr bool allocates_memory_space(const std::allocator<T>& a) {
  return S == MemorySpace::Host;
}
template <MemorySpace S, typename T>
constexpr bool allocates_memory_space(const Eigen::aligned_allocator<T>& a) {
  return S == MemorySpace::Host;
}
template <MemorySpace S, typename T>
constexpr bool allocates_memory_space(const host_allocator<T>& a) {
  return S == MemorySpace::Host;
}
#ifdef TILEDARRAY_HAS_DEVICE
template <MemorySpace S, typename T>
constexpr bool allocates_memory_space(const device_um_allocator<T>& a) {
  return S == MemorySpace::Device_UM;
}
#endif

/// enumerates the execution spaces
enum class ExecutionSpace { Host, Device };

// customization point: to_execution_space<S>(O) -> void
// "moves" O to execution space S

}  // namespace TiledArray

#endif  // TILEDARRAY_PLATFORM_H__INCLUDED
