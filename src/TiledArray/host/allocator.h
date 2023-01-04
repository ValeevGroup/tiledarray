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

#ifndef TILEDARRAY_HOST_ALLOCATOR_H___INCLUDED
#define TILEDARRAY_HOST_ALLOCATOR_H___INCLUDED

#include <TiledArray/config.h>

#include <TiledArray/external/umpire.h>
#include <TiledArray/host/env.h>

#include <TiledArray/fwd.h>

#include <memory>
#include <stdexcept>

namespace TiledArray {

/// pooled, thread-safe allocator for host memory
template <class T>
class host_allocator_impl
    : public umpire_allocator_impl<T, detail::MutexLock<hostEnv>> {
 public:
  using base_type = umpire_allocator_impl<T, detail::MutexLock<hostEnv>>;
  using typename base_type::const_pointer;
  using typename base_type::const_reference;
  using typename base_type::pointer;
  using typename base_type::reference;
  using typename base_type::value_type;

  host_allocator_impl() noexcept
      : base_type(&hostEnv::instance()->host_allocator()) {}

  template <class U>
  host_allocator_impl(const host_allocator_impl<U>& rhs) noexcept
      : base_type(static_cast<const umpire_allocator_impl<U>&>(rhs)) {}

  template <typename T1, typename T2>
  friend bool operator==(const host_allocator_impl<T1>& lhs,
                         const host_allocator_impl<T2>& rhs) noexcept;
};  // class host_allocator_impl

template <class T1, class T2>
bool operator==(const host_allocator_impl<T1>& lhs,
                const host_allocator_impl<T2>& rhs) noexcept {
  return lhs.umpire_allocator() == rhs.umpire_allocator();
}

template <class T1, class T2>
bool operator!=(const host_allocator_impl<T1>& lhs,
                const host_allocator_impl<T2>& rhs) noexcept {
  return !(lhs == rhs);
}

}  // namespace TiledArray

#endif  // TILEDARRAY_HOST_ALLOCATOR_H___INCLUDED
