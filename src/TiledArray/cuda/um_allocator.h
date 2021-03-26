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

#ifndef TILEDARRAY_CUDA_UM_ALLOCATOR_H___INCLUDED
#define TILEDARRAY_CUDA_UM_ALLOCATOR_H___INCLUDED

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_CUDA

#include <TiledArray/util/allocator.h>

#include <TiledArray/external/cuda.h>

#include <memory>
#include <stdexcept>

namespace TiledArray {

namespace detail {

/// CUDA UM allocator, based on boilerplate by Howard Hinnant
/// (https://howardhinnant.github.io/allocator_boilerplate.html)
template <class T>
class cuda_um_allocator_impl {
 public:
  using value_type = T;
  using pointer = T*;
  using reference = T&;
  using const_reference = const T&;

  cuda_um_allocator_impl() noexcept
      : um_dynamic_pool_(&cudaEnv::instance()->um_dynamic_pool()) {}

  template <class U>
  cuda_um_allocator_impl(const cuda_um_allocator_impl<U>& rhs) noexcept
      : um_dynamic_pool_(rhs.um_dynamic_pool_) {}

  /// allocates um memory using umpire dynamic pool
  pointer allocate(size_t n) {
    pointer result = nullptr;

    TA_ASSERT(um_dynamic_pool_);

    result = static_cast<pointer>(um_dynamic_pool_->allocate(n * sizeof(T)));

    return result;
  }

  /// deallocate um memory using umpire dynamic pool
  void deallocate(value_type* ptr, size_t) {
    TA_ASSERT(um_dynamic_pool_);
    um_dynamic_pool_->deallocate(ptr);
  }

  template <typename T1, typename T2>
  friend bool operator==(const cuda_um_allocator_impl<T1>& lhs,
                         const cuda_um_allocator_impl<T2>& rhs) noexcept;

 private:
  umpire::Allocator* um_dynamic_pool_;
};  // class cuda_um_allocator

template <class T1, class T2>
bool operator==(const cuda_um_allocator_impl<T1>& lhs,
                const cuda_um_allocator_impl<T2>& rhs) noexcept {
  return lhs.um_dynamic_pool_ == rhs.um_dynamic_pool_;
}

template <class T1, class T2>
bool operator!=(const cuda_um_allocator_impl<T1>& lhs,
                const cuda_um_allocator_impl<T2>& rhs) noexcept {
  return !(lhs == rhs);
}

}  // namespace detail

template <typename T>
using cuda_um_allocator =
    default_init_allocator<T, detail::cuda_um_allocator_impl<T>>;

}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_CUDA

#endif  // TILEDARRAY_CUDA_UM_ALLOCATOR_H___INCLUDED
