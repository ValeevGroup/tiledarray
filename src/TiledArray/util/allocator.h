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
 *  Edward Valeev
 *  Department of Chemistry, Virginia Tech
 *
 *  allocator.h
 *  Mar 26, 2021
 *
 */

#ifndef TILEDARRAY_SRC_TILEDARRAY_UTIL_ALLOCATOR_H
#define TILEDARRAY_SRC_TILEDARRAY_UTIL_ALLOCATOR_H

#include <TiledArray/error.h>

#include <memory>

namespace TiledArray {

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

namespace detail {

/// controls batch size at compile-time
template <std::size_t Size>
struct ctime_batch_size_base {
  static_assert(Size > 0,
                "ctime_batch_size_base<Size>: zero Size not supported");
  constexpr std::size_t batch_size() const noexcept { return Size; }
};

/// controls batch size at runtime
struct rtime_batch_size_base {
  rtime_batch_size_base(std::size_t batch_size) : batch_size_(batch_size) {
    TA_ASSERT(batch_size != 0);
  }

  rtime_batch_size_base() noexcept = default;
  rtime_batch_size_base(const rtime_batch_size_base&) noexcept = default;
  rtime_batch_size_base(rtime_batch_size_base&&) noexcept = default;
  ~rtime_batch_size_base() noexcept = default;

  rtime_batch_size_base& operator=(const rtime_batch_size_base&) noexcept =
      default;
  rtime_batch_size_base& operator=(rtime_batch_size_base&&) noexcept = default;

  /// @return the batch size
  std::size_t batch_size() const noexcept { return batch_size_; }

 private:
  std::size_t batch_size_ = 1;
};

/// see TiledArray::detail::batch_allocator
template <typename T, typename BaseAllocator = std::allocator<T>,
          std::size_t BatchSize = 0>
class batch_allocator_impl
    : public BaseAllocator,
      public std::conditional_t<(BatchSize != 0),
                                ctime_batch_size_base<BatchSize>,
                                rtime_batch_size_base> {
 public:
  using value_type = typename BaseAllocator::value_type;
  using pointer = value_type*;
  using base1_type = BaseAllocator;
  using base2_type =
      std::conditional_t<(BatchSize != 0), ctime_batch_size_base<BatchSize>,
                         rtime_batch_size_base>;

  batch_allocator_impl() noexcept = default;

  explicit batch_allocator_impl(std::size_t batch_size)
      : base2_type(batch_size) {}

  template <class U>
  batch_allocator_impl(const batch_allocator_impl<U>& other) noexcept {}

  /// allocates enough memory for `batch_size` instances of @p n objects of
  /// `value_type`
  /// @param[in] n the number of objects of type `value_type` to allocate
  /// @result pointer to the allocated memory
  pointer allocate(size_t n) {
    return static_cast<pointer>(
        base1_type::allocate(this->batch_size() * n * sizeof(value_type)));
  }

  /// deallocate the memory allocated with batch_allocator_impl::allocate()
  void deallocate(value_type* ptr, size_t) { base1_type::deallocate(ptr); }

  template <typename T1, typename BaseAllocator1, std::size_t BatchSize1,
            typename T2, typename BaseAllocator2, std::size_t BatchSize2>
  friend bool operator==(
      const batch_allocator_impl<T1, BaseAllocator1, BatchSize1>& lhs,
      const batch_allocator_impl<T2, BaseAllocator2, BatchSize2>& rhs) noexcept;
};  // class batch_allocator_impl

template <typename T1, typename BaseAllocator1, std::size_t BatchSize1,
          typename T2, typename BaseAllocator2, std::size_t BatchSize2>
bool operator==(
    const batch_allocator_impl<T1, BaseAllocator1, BatchSize1>& lhs,
    const batch_allocator_impl<T2, BaseAllocator2, BatchSize2>& rhs) noexcept {
  return BatchSize1 == BatchSize2 && lhs.batch_size() == rhs.batch_size() &&
         static_cast<const BaseAllocator1&>(lhs) ==
             static_cast<const BaseAllocator2&>(rhs);
}

template <typename BaseAllocator1, typename BaseAllocator2>
bool operator!=(const batch_allocator_impl<BaseAllocator1>& lhs,
                const batch_allocator_impl<BaseAllocator2>& rhs) noexcept {
  return !(lhs == rhs);
}

}  // namespace detail

/// @brief batch allocator

/// batch allocator allocates memory for `n * this->batch_size()`
/// @tparam T the type of values allocated by this allocator
/// @tparam A the base allocator type which is used to actually allocate the
/// memory
/// @tparam BatchSize if nonzero, the batch size is controlled at runtime (this
/// is the default), otherwise batch size is fixed
template <typename T, typename A = std::allocator<T>, std::size_t BatchSize = 0>
using batch_allocator =
    default_init_allocator<T, detail::batch_allocator_impl<T, A, BatchSize>>;

namespace detail {

template <typename A>
struct is_batch_allocator : std::false_type {};

template <typename T, typename A, std::size_t B>
struct is_batch_allocator<batch_allocator<T, A, B>> : std::true_type {};

template <typename A>
constexpr inline bool is_batch_allocator_v = is_batch_allocator<A>::value;

template <typename T, typename Enabler = void>
struct has_batch_allocator : std::false_type {};

template <typename T>
struct has_batch_allocator<
    T, std::void_t<
           typename T::allocator_type,
           std::enable_if_t<is_batch_allocator_v<typename T::allocator_type>>>>
    : std::true_type {};

template <typename T>
constexpr inline bool has_batch_allocator_v = has_batch_allocator<T>::value;

}  // namespace detail

}  // namespace TiledArray

#endif  // TILEDARRAY_SRC_TILEDARRAY_UTIL_ALLOCATOR_H
