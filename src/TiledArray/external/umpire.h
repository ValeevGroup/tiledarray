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

#include <madness/world/archive.h>

#include <memory>
#include <stdexcept>

namespace TiledArray {

namespace detail {

struct NullLock {
  static void lock() {}
  static void unlock() {}
};

template <typename Tag>
class MutexLock {
  static std::mutex mtx_;

 public:
  static void lock() { mtx_.lock(); }
  static void unlock() { mtx_.unlock(); }
};

template <typename Tag>
std::mutex MutexLock<Tag>::mtx_;

}  // namespace detail

/// wraps a Umpire allocator into a
/// *standard-compliant* C++ allocator

/// Optionally can be made thread safe by providing an appropriate \p StaticLock
/// \details based on the boilerplate by Howard Hinnant
/// (https://howardhinnant.github.io/allocator_boilerplate.html)
/// \tparam T type of allocated objects
/// \tparam StaticLock a type providing static `lock()` and `unlock()` methods ;
///         defaults to NullLock which does not lock
template <class T, class StaticLock = detail::NullLock>
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
  umpire_allocator_impl(
      const umpire_allocator_impl<U, StaticLock>& rhs) noexcept
      : umpalloc_(rhs.umpalloc_) {}

  /// allocates memory using umpire dynamic pool
  pointer allocate(size_t n) {
    TA_ASSERT(umpalloc_);

    // QuickPool::allocate_internal does not handle zero-size allocations
    size_t nbytes = n == 0 ? 1 : n * sizeof(T);
    pointer result = nullptr;
    auto* allocation_strategy = umpalloc_->getAllocationStrategy();

    // critical section
    StaticLock::lock();
    // this, instead of umpalloc_->allocate(n*sizeof(T)), profiles memory use
    // even if introspection is off
    result =
        static_cast<pointer>(allocation_strategy->allocate_internal(nbytes));
    StaticLock::unlock();

    return result;
  }

  /// deallocate memory using umpire dynamic pool
  void deallocate(pointer ptr, size_t n) {
    TA_ASSERT(umpalloc_);

    // QuickPool::allocate_internal does not handle zero-size allocations
    const auto nbytes = n == 0 ? 1 : n * sizeof(T);
    auto* allocation_strategy = umpalloc_->getAllocationStrategy();

    // N.B. with multiple threads would have to do this test in
    // the critical section of Umpire's ThreadSafeAllocator::deallocate
    StaticLock::lock();
    TA_ASSERT(nbytes <= allocation_strategy->getCurrentSize());
    // this, instead of umpalloc_->deallocate(ptr, nbytes), profiles memory use
    // even if introspection is off
    allocation_strategy->deallocate_internal(ptr, nbytes);
    StaticLock::unlock();
  }

  /// @return the underlying Umpire allocator
  const umpire::Allocator* umpire_allocator() const { return umpalloc_; }

 private:
  umpire::Allocator* umpalloc_;
};  // class umpire_allocator_impl

template <class T1, class T2, class StaticLock>
bool operator==(const umpire_allocator_impl<T1, StaticLock>& lhs,
                const umpire_allocator_impl<T2, StaticLock>& rhs) noexcept {
  return lhs.umpire_allocator() == rhs.umpire_allocator();
}

template <class T1, class T2, class StaticLock>
bool operator!=(const umpire_allocator_impl<T1, StaticLock>& lhs,
                const umpire_allocator_impl<T2, StaticLock>& rhs) noexcept {
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

  default_init_allocator(A const& a) noexcept : A(a) {}
  default_init_allocator(A&& a) noexcept : A(std::move(a)) {}

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

namespace madness {
namespace archive {

template <class Archive, class T, class StaticLock>
struct ArchiveLoadImpl<Archive,
                       TiledArray::umpire_allocator_impl<T, StaticLock>> {
  static inline void load(
      const Archive& ar,
      TiledArray::umpire_allocator_impl<T, StaticLock>& allocator) {
    std::string allocator_name;
    ar& allocator_name;
    allocator = TiledArray::umpire_allocator_impl<T, StaticLock>(
        umpire::ResourceManager::getInstance().getAllocator(allocator_name));
  }
};

template <class Archive, class T, class StaticLock>
struct ArchiveStoreImpl<Archive,
                        TiledArray::umpire_allocator_impl<T, StaticLock>> {
  static inline void store(
      const Archive& ar,
      const TiledArray::umpire_allocator_impl<T, StaticLock>& allocator) {
    ar& allocator.umpire_allocator()->getName();
  }
};

template <class Archive, typename T, typename A>
struct ArchiveLoadImpl<Archive, TiledArray::default_init_allocator<T, A>> {
  static inline void load(const Archive& ar,
                          TiledArray::default_init_allocator<T, A>& allocator) {
    if constexpr (!std::allocator_traits<A>::is_always_equal::value) {
      A base_allocator;
      ar& base_allocator;
      allocator = TiledArray::default_init_allocator<T, A>(base_allocator);
    }
  }
};

template <class Archive, typename T, typename A>
struct ArchiveStoreImpl<Archive, TiledArray::default_init_allocator<T, A>> {
  static inline void store(
      const Archive& ar,
      const TiledArray::default_init_allocator<T, A>& allocator) {
    if constexpr (!std::allocator_traits<A>::is_always_equal::value) {
      ar& static_cast<const A&>(allocator);
    }
  }
};

}  // namespace archive
}  // namespace madness

#endif  // TILEDARRAY_EXTERNAL_UMPIRE_H___INCLUDED
