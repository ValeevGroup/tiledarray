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

#ifndef TILEDARRAY_CUDA_PLATFORM_H__INCLUDED
#define TILEDARRAY_CUDA_PLATFORM_H__INCLUDED

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

/// enumerates the execution spaces
enum class ExecutionSpace { Host, Device };

// customization point: to_execution_space<S>(O) -> void
// "moves" O to execution space S

}  // namespace TiledArray

#endif  // TILEDARRAY_CUDA_PLATFORM_H__INCLUDED
