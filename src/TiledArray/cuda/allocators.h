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

#ifndef TILEDARRAY_CUDA_ALLOCATORS_H___INCLUDED
#define TILEDARRAY_CUDA_ALLOCATORS_H___INCLUDED

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_CUDA

#include <TiledArray/external/cuda.h>
#include <TiledArray/external/umpire.h>

#include <madness/world/archive.h>

#include <memory>
#include <stdexcept>

namespace TiledArray {

template <class T, class StaticLock, typename UmpireAllocatorAccessor>
class cuda_allocator_impl : public umpire_allocator_impl<T, StaticLock> {
 public:
  using base_type = umpire_allocator_impl<T, StaticLock>;
  using typename base_type::const_pointer;
  using typename base_type::const_reference;
  using typename base_type::pointer;
  using typename base_type::reference;
  using typename base_type::value_type;

  cuda_allocator_impl() noexcept : base_type(&UmpireAllocatorAccessor{}()) {}

  template <class U>
  cuda_allocator_impl(
      const cuda_allocator_impl<U, StaticLock, UmpireAllocatorAccessor>&
          rhs) noexcept
      : base_type(
            static_cast<const umpire_allocator_impl<U, StaticLock>&>(rhs)) {}

  template <typename T1, typename T2, class StaticLock_,
            typename UmpireAllocatorAccessor_>
  friend bool operator==(
      const cuda_allocator_impl<T1, StaticLock_, UmpireAllocatorAccessor_>& lhs,
      const cuda_allocator_impl<T2, StaticLock_, UmpireAllocatorAccessor_>&
          rhs) noexcept;
};  // class cuda_allocator_impl

template <class T1, class T2, class StaticLock,
          typename UmpireAllocatorAccessor>
bool operator==(
    const cuda_allocator_impl<T1, StaticLock, UmpireAllocatorAccessor>& lhs,
    const cuda_allocator_impl<T2, StaticLock, UmpireAllocatorAccessor>&
        rhs) noexcept {
  return lhs.umpire_allocator() == rhs.umpire_allocator();
}

template <class T1, class T2, class StaticLock,
          typename UmpireAllocatorAccessor>
bool operator!=(
    const cuda_allocator_impl<T1, StaticLock, UmpireAllocatorAccessor>& lhs,
    const cuda_allocator_impl<T2, StaticLock, UmpireAllocatorAccessor>&
        rhs) noexcept {
  return !(lhs == rhs);
}

namespace detail {

struct get_um_allocator {
  umpire::Allocator& operator()() {
    return cudaEnv::instance()->um_allocator();
  }
};

struct get_pinned_allocator {
  umpire::Allocator& operator()() {
    return cudaEnv::instance()->pinned_allocator();
  }
};

}  // namespace detail

}  // namespace TiledArray

namespace madness {
namespace archive {

template <class Archive, class T, class StaticLock,
          typename UmpireAllocatorAccessor>
struct ArchiveLoadImpl<Archive, TiledArray::cuda_allocator_impl<
                                    T, StaticLock, UmpireAllocatorAccessor>> {
  static inline void load(
      const Archive& ar,
      TiledArray::cuda_allocator_impl<T, StaticLock, UmpireAllocatorAccessor>&
          allocator) {
    allocator = TiledArray::cuda_allocator_impl<T, StaticLock,
                                                UmpireAllocatorAccessor>{};
  }
};

template <class Archive, class T, class StaticLock,
          typename UmpireAllocatorAccessor>
struct ArchiveStoreImpl<Archive, TiledArray::cuda_allocator_impl<
                                     T, StaticLock, UmpireAllocatorAccessor>> {
  static inline void store(
      const Archive& ar,
      const TiledArray::cuda_allocator_impl<
          T, StaticLock, UmpireAllocatorAccessor>& allocator) {}
};

}  // namespace archive
}  // namespace madness

#endif  // TILEDARRAY_HAS_CUDA

#endif  // TILEDARRAY_CUDA_ALLOCATORS_H___INCLUDED
