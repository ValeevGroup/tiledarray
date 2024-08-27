/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2017  Virginia Tech
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
 *
 *  meta.h
 *  April 11, 2017
 *
 */

#ifndef TILEDARRAY_UTIL_INVOKE_H
#define TILEDARRAY_UTIL_INVOKE_H

#include <TiledArray/external/madness.h>
#include <madness/world/future.h>
#include <madness/world/world.h>
#include <madness/world/world_task_queue.h>

namespace TiledArray {
namespace detail {

/// ||'s bools
template <bool head, bool... tail>
struct or_reduce {
  static constexpr bool value = head || or_reduce<tail...>::value;
};

template <bool b>
struct or_reduce<b> {
  static constexpr bool value = b;
};

// is any argument a Future?
// - yes: async launch
// -  no: direct launch
template <typename Function, typename... Args>
auto invoke(Function&& fn, Args&&... args) -> typename std::enable_if<
    !or_reduce<false, madness::is_future<std::decay_t<Args>>::value...>::value,
    decltype(fn(args...))>::type {
  return fn(std::forward<Args>(args)...);
}

template <
    typename Function, typename... Args,
    typename = typename std::enable_if<or_reduce<
        false, madness::is_future<std::decay_t<Args>>::value...>::value>::type>
auto invoke(Function&& fn, Args&&... args) {
  return TiledArray::get_default_world().taskq.add(std::forward<Function>(fn),
                                                   std::forward<Args>(args)...);
}

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_UTIL_INVOKE_H
