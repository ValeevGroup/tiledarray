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
 *  Chong Peng
 *  Department of Chemistry, Virginia Tech
 *  July 23, 2018
 *
 */

#ifndef TILEDARRAY_DEVICE_BLAS_H__INCLUDED
#define TILEDARRAY_DEVICE_BLAS_H__INCLUDED

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_DEVICE

#include <TiledArray/error.h>
#include <TiledArray/external/device.h>
#include <TiledArray/tensor/complex.h>
#include <blas/device.hh>

namespace TiledArray {

/**
 * BLASQueuePool is a singleton controlling a pool of blas::Queue objects:
 * - queues map to stream 1-to-1, so do not call Queue::set_stream to maintain
 * this invariant
 * - can access queues by the corresponding stream ordinal a la
 * deviceEnv::stream()
 */
struct BLASQueuePool {
  static bool initialized();
  static void initialize();
  static void finalize();

  static blas::Queue &queue(std::size_t ordinal = 0);
  static blas::Queue &queue(const device::Stream &s);

 private:
  static std::vector<std::unique_ptr<blas::Queue>> queues_;
};

/// maps a (tile) Range to blas::Queue; if had already pushed work into a
/// device::Stream (as indicated by madness_task_current_stream() )
/// will return that Stream instead
/// @param[in] range will determine the device::Stream to compute an object
/// associated with this Range object
/// @return the device::Stream to use for creating tasks generating work
/// associated with Range \p range
template <typename Range>
blas::Queue &blasqueue_for(const Range &range) {
  auto stream_opt = device::madness_task_current_stream();
  if (!stream_opt) {
    auto stream_ord =
        range.offset() % device::Env::instance()->num_streams_total();
    return BLASQueuePool::queue(stream_ord);
  } else
    return BLASQueuePool::queue(*stream_opt);
}

}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_DEVICE

#endif  // TILEDARRAY_DEVICE_BLAS_H__INCLUDED
