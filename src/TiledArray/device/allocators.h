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
 *  Jan 31, 2018
 *
 */

#ifndef TILEDARRAY_DEVICE_ALLOCATORS_H___INCLUDED
#define TILEDARRAY_DEVICE_ALLOCATORS_H___INCLUDED

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_DEVICE

#include <TiledArray/external/device.h>
#include <TiledArray/external/umpire.h>

#include <madness/world/archive.h>

#include <memory>
#include <stdexcept>

namespace TiledArray {

template <class T, class StaticLock, typename UmpireAllocatorAccessor>
class umpire_based_allocator
    : public umpire_based_allocator_impl<T, StaticLock> {
 public:
  using base_type = umpire_based_allocator_impl<T, StaticLock>;
  using typename base_type::const_pointer;
  using typename base_type::const_reference;
  using typename base_type::pointer;
  using typename base_type::reference;
  using typename base_type::value_type;

  umpire_based_allocator() noexcept : base_type(&UmpireAllocatorAccessor{}()) {}

  template <class U>
  umpire_based_allocator(
      const umpire_based_allocator<U, StaticLock, UmpireAllocatorAccessor>&
          rhs) noexcept
      : base_type(
            static_cast<const umpire_based_allocator_impl<U, StaticLock>&>(
                rhs)) {}

  template <typename T1, typename T2, class StaticLock_,
            typename UmpireAllocatorAccessor_>
  friend bool operator==(
      const umpire_based_allocator<T1, StaticLock_, UmpireAllocatorAccessor_>&
          lhs,
      const umpire_based_allocator<T2, StaticLock_, UmpireAllocatorAccessor_>&
          rhs) noexcept;
};  // class umpire_based_allocator

template <class T1, class T2, class StaticLock,
          typename UmpireAllocatorAccessor>
bool operator==(
    const umpire_based_allocator<T1, StaticLock, UmpireAllocatorAccessor>& lhs,
    const umpire_based_allocator<T2, StaticLock, UmpireAllocatorAccessor>&
        rhs) noexcept {
  return lhs.umpire_allocator() == rhs.umpire_allocator();
}

template <class T1, class T2, class StaticLock,
          typename UmpireAllocatorAccessor>
bool operator!=(
    const umpire_based_allocator<T1, StaticLock, UmpireAllocatorAccessor>& lhs,
    const umpire_based_allocator<T2, StaticLock, UmpireAllocatorAccessor>&
        rhs) noexcept {
  return !(lhs == rhs);
}

namespace detail {

struct get_um_allocator {
  umpire::Allocator& operator()() {
    return deviceEnv::instance()->um_allocator();
  }
};

struct get_pinned_allocator {
  umpire::Allocator& operator()() {
    return deviceEnv::instance()->pinned_allocator();
  }
};

}  // namespace detail

}  // namespace TiledArray

namespace madness {
namespace archive {

template <class Archive, class T, class StaticLock,
          typename UmpireAllocatorAccessor>
struct ArchiveLoadImpl<Archive, TiledArray::umpire_based_allocator<
                                    T, StaticLock, UmpireAllocatorAccessor>> {
  static inline void load(
      const Archive& ar,
      TiledArray::umpire_based_allocator<T, StaticLock,
                                         UmpireAllocatorAccessor>& allocator) {
    allocator = TiledArray::umpire_based_allocator<T, StaticLock,
                                                   UmpireAllocatorAccessor>{};
  }
};

template <class Archive, class T, class StaticLock,
          typename UmpireAllocatorAccessor>
struct ArchiveStoreImpl<Archive, TiledArray::umpire_based_allocator<
                                     T, StaticLock, UmpireAllocatorAccessor>> {
  static inline void store(
      const Archive& ar,
      const TiledArray::umpire_based_allocator<
          T, StaticLock, UmpireAllocatorAccessor>& allocator) {}
};

}  // namespace archive
}  // namespace madness

#endif  // TILEDARRAY_HAS_DEVICE

#endif  // TILEDARRAY_DEVICE_ALLOCATORS_H___INCLUDED
