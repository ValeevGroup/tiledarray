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

#include <TiledArray/external/cuda.h>
#include <TiledArray/external/umpire.h>

#include <memory>
#include <stdexcept>

namespace TiledArray {

/// CUDA UM allocator, based on boilerplate by Howard Hinnant
/// (https://howardhinnant.github.io/allocator_boilerplate.html)
template <class T>
class cuda_um_allocator_impl : public umpire_allocator_impl<T> {
 public:
  using base_type = umpire_allocator_impl<T>;
  using typename base_type::const_pointer;
  using typename base_type::const_reference;
  using typename base_type::pointer;
  using typename base_type::reference;
  using typename base_type::value_type;

  cuda_um_allocator_impl() noexcept
      : base_type(&cudaEnv::instance()->um_dynamic_pool()) {}

  template <class U>
  cuda_um_allocator_impl(const cuda_um_allocator_impl<U>& rhs) noexcept
      : base_type(static_cast<const umpire_allocator_impl<U>&>(rhs)) {}

  template <typename T1, typename T2>
  friend bool operator==(const cuda_um_allocator_impl<T1>& lhs,
                         const cuda_um_allocator_impl<T2>& rhs) noexcept;
};  // class cuda_um_allocator

template <class T1, class T2>
bool operator==(const cuda_um_allocator_impl<T1>& lhs,
                const cuda_um_allocator_impl<T2>& rhs) noexcept {
  return lhs.umpire_allocator() == rhs.umpire_allocator();
}

template <class T1, class T2>
bool operator!=(const cuda_um_allocator_impl<T1>& lhs,
                const cuda_um_allocator_impl<T2>& rhs) noexcept {
  return !(lhs == rhs);
}

template <typename T>
using cuda_um_allocator = default_init_allocator<T, cuda_um_allocator_impl<T>>;

}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_CUDA

#endif  // TILEDARRAY_CUDA_UM_ALLOCATOR_H___INCLUDED
