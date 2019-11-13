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
 *  util/logger.h
 *  Mar 11, 2019
 *
 */

#ifndef TILEDARRAY_UTIL_LOGGER_H__INCLUDED
#define TILEDARRAY_UTIL_LOGGER_H__INCLUDED

#include <functional>
#include <ostream>

#include <TiledArray/config.h>
#include <TiledArray/range.h>
#include <TiledArray/util/singleton.h>

namespace TiledArray {

struct TileOpsLogger : public Singleton<TileOpsLogger> {

  using range_transform_t = std::function<Range(const Range&)>;
  using range_filter_t = std::function<bool(const Range&)>;

  // GEMM task logging
  bool gemm = false;
  range_transform_t gemm_left_range_transform;
  range_transform_t gemm_right_range_transform;
  range_transform_t gemm_result_range_transform;
  range_filter_t gemm_result_range_filter;

  // logging
  std::ostream* log = &std::cout;

  template <typename T>
  TileOpsLogger& operator<<(T&& arg) {
    *log << std::forward<T>(arg);
    return *this;
  }

  TileOpsLogger& operator<<(std::ostream& (*func)(std::ostream&)) {
    *log << func;
    return *this;
  }

 private:
  friend class Singleton<TileOpsLogger>;
  TileOpsLogger(int log_level = TA_TILE_OPS_LOG_LEVEL) {
    if (log_level > 0) {
      gemm = true;
    }
  }
};

}

#endif // TILEDARRAY_UTIL_LOGGER_H__INCLUDED
