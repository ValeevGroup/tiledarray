/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2021  Virginia Tech
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

#ifndef TILEDARRAY_EXTERNAL_UMPIRE_H___INCLUDED
#define TILEDARRAY_EXTERNAL_UMPIRE_H___INCLUDED

#include <TiledArray/fwd.h>

#include <TiledArray/error.h>

// for memory management
#include <umpire/Umpire.hpp>
#include <umpire/strategy/QuickPool.hpp>
#include <umpire/strategy/SizeLimiter.hpp>
#include <umpire/strategy/ThreadSafeAllocator.hpp>

#include <memory>
#include <stdexcept>

namespace TiledArray {

/// wraps Umpire allocator into a standard-compliant allocator,
/// based on the boilerplate by Howard Hinnant
/// (https://howardhinnant.github.io/allocator_boilerplate.html)
template <class T>
class umpire_allocator_impl {
 public:
  using value_type = T;
  using pointer = value_type*;
  using const_pointer =
      typename std::pointer_traits<pointer>::template rebind<value_type const>;
  using void_pointer =
      typename std::pointer_traits<pointer>::template rebind<void>;
  using const_void_pointer =
      typename std::pointer_traits<pointer>::template rebind<const void>;

  using reference = T&;
  using const_reference = const T&;

  using difference_type =
      typename std::pointer_traits<pointer>::difference_type;
  using size_type = std::make_unsigned_t<difference_type>;

  umpire_allocator_impl(umpire::Allocator* umpalloc) noexcept
      : umpalloc_(umpalloc) {}

  template <class U>
  umpire_allocator_impl(const umpire_allocator_impl<U>& rhs) noexcept
      : umpalloc_(rhs.umpalloc_) {}

  /// allocates um memory using umpire dynamic pool
  pointer allocate(size_t n) {
    pointer result = nullptr;

    TA_ASSERT(umpalloc_);

    result = static_cast<pointer>(umpalloc_->allocate(n * sizeof(T)));

    return result;
  }

  /// deallocate um memory using umpire dynamic pool
  void deallocate(pointer ptr, size_t) {
    TA_ASSERT(umpalloc_);
    umpalloc_->deallocate(ptr);
  }

  const umpire::Allocator* umpire_allocator() const { return umpalloc_; }

 private:
  umpire::Allocator* umpalloc_;
};  // class umpire_allocator

template <class T1, class T2>
bool operator==(const umpire_allocator_impl<T1>& lhs,
                const umpire_allocator_impl<T2>& rhs) noexcept {
  return lhs.um_dynamic_pool() == rhs.um_dynamic_pool();
}

template <class T1, class T2>
bool operator!=(const umpire_allocator_impl<T1>& lhs,
                const umpire_allocator_impl<T2>& rhs) noexcept {
  return !(lhs == rhs);
}

/// see
/// https://stackoverflow.com/questions/21028299/is-this-behavior-of-vectorresizesize-type-n-under-c11-and-boost-container/21028912#21028912
template <typename T, typename A>
class default_init_allocator : public A {
  using a_t = std::allocator_traits<A>;

 public:
  using reference = typename A::reference;  // std::allocator<T>::reference
                                            // deprecated in C++17, but thrust
                                            // still relying on this
  using const_reference = typename A::const_reference;  // ditto

  template <typename U>
  struct rebind {
    using other =
        default_init_allocator<U, typename a_t::template rebind_alloc<U>>;
  };

  using A::A;

  template <typename U>
  void construct(U* ptr) noexcept(
      std::is_nothrow_default_constructible<U>::value) {
    ::new (static_cast<void*>(ptr)) U;
  }
  template <typename U, typename... Args>
  void construct(U* ptr, Args&&... args) {
    a_t::construct(static_cast<A&>(*this), ptr, std::forward<Args>(args)...);
  }
};

}  // namespace TiledArray

#endif  // TILEDARRAY_CUDA_UM_ALLOCATOR_H___INCLUDED
