/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2019  Virginia Tech
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
 *  util/time.h
 *  Oct 24, 2019
 *
 */

#ifndef TILEDARRAY_UTIL_TIME_H__INCLUDED
#define TILEDARRAY_UTIL_TIME_H__INCLUDED

#include <chrono>

namespace TiledArray {

using time_point = std::chrono::high_resolution_clock::time_point;

inline time_point now() { return std::chrono::high_resolution_clock::now(); }

inline std::chrono::system_clock::time_point system_now() {
  return std::chrono::system_clock::now();
}

inline double duration_in_s(time_point const &t0, time_point const &t1) {
  return std::chrono::duration<double>{t1 - t0}.count();
}

inline int64_t duration_in_ns(time_point const &t0, time_point const &t1) {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count();
}

}  // namespace TiledArray

#endif  // TILEDARRAY_UTIL_TIME_H__INCLUDED
