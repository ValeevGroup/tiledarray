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

#include <TiledArray/external/device.h>

#include <TiledArray/error.h>
#include <TiledArray/tensor/complex.h>

#include <TiledArray/math/blas.h>

namespace TiledArray {

/*
 * cuBLAS interface functions
 */

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
  static blas::Queue &queue(const device::stream_t &stream);

 private:
  static std::vector<std::unique_ptr<blas::Queue>> queues_;
};

namespace detail {
template <typename Range>
blas::Queue &get_blasqueue_based_on_range(const Range &range) {
  // TODO better way to get stream based on the id of tensor
  auto stream_ord = range.offset() % device::Env::instance()->num_streams();
  return BLASQueuePool::queue(stream_ord);
}
}  // namespace detail

}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_DEVICE

#endif  // TILEDARRAY_DEVICE_BLAS_H__INCLUDED
