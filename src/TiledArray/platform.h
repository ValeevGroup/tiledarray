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

#include <TiledArray/error.h>
#include <TiledArray/fwd.h>

#include <TiledArray/type_traits.h>

#include <madness/world/thread_info.h>

#include <atomic>
#include <cstddef>

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

/// the memory space allocator type \c A allocates in, known at compile time
/// (unlike allocates_memory_space, which takes an allocator object);
/// specialize for new allocators
template <typename A>
inline constexpr MemorySpace allocator_memory_space_v = MemorySpace::Host;
#ifdef TILEDARRAY_HAS_DEVICE
template <typename T>
inline constexpr MemorySpace allocator_memory_space_v<device_um_allocator<T>> =
    MemorySpace::Device_UM;
#endif

/// enumerates the execution spaces
enum class ExecutionSpace { Host, Device };

namespace detail {
inline std::atomic<ExecutionSpace>& execution_space_accessor() {
  static std::atomic<ExecutionSpace> val{ExecutionSpace::Device};
  return val;
}
}  // namespace detail

/// \return where ops on tiles in device-accessible (unified) memory run:
/// Device unless an ExecutionSpaceScope says otherwise. Tiles in host memory
/// always run on the host.
inline ExecutionSpace execution_space() {
  return detail::execution_space_accessor().load();
}

/// Sets execution_space() for its lifetime and restores the previous value on
/// destruction; scopes nest.
/// \code
/// {
///   ExecutionSpaceScope host(ExecutionSpace::Host);
///   c("i,j") = a("i,k") * b("k,j");  // UMTensor tiles computed on the host
/// }
/// \endcode
/// \warning the setting is process-wide and is read when each task runs.
///          Work still running when the scope ends uses the restored value,
///          so call `.get()` on returned Futures (e.g. from `dot()`) inside
///          the scope. Open scopes from the main thread only, after
///          TA::initialize; the constructor throws otherwise.
///          Being process-wide, it also switches tasks of other arrays that
///          are in flight when the scope opens or closes.
/// \note the scope decides which code the tile ops run, not which tasks
///       run them: only the elementwise evaluators (UnaryEvalImpl,
///       BinaryEvalImpl) skip the device task under a Host scope. Contraction
///       argument conversion and reduction (contraction_eval.h,
///       reduce_task.h), assignment-time conversions (expr.h) and clone still
///       submit device tasks, whose tile ops then run the host code after
///       waiting for the task's device work.
class ExecutionSpaceScope {
 public:
  explicit ExecutionSpaceScope(ExecutionSpace space) {
    if (!madness::is_main_thread())
      TA_EXCEPTION(
          "ExecutionSpaceScope: open scopes from the main thread only");
    previous_ = detail::execution_space_accessor().exchange(space);
  }
  ~ExecutionSpaceScope() { detail::execution_space_accessor() = previous_; }

  ExecutionSpaceScope(const ExecutionSpaceScope&) = delete;
  ExecutionSpaceScope& operator=(const ExecutionSpaceScope&) = delete;

  // stack only, so that scopes end in the reverse order they began
  static void* operator new(std::size_t) = delete;
  static void* operator new[](std::size_t) = delete;

 private:
  ExecutionSpace previous_;
};

// customization point: to_execution_space<S>(O) -> void
// "moves" O to execution space S

}  // namespace TiledArray

#endif  // TILEDARRAY_PLATFORM_H__INCLUDED
