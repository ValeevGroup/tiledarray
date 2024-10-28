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
 *  Sept 19, 2023
 *
 */

#include <TiledArray/device/blas.h>

#include <memory>
#include <vector>

namespace TiledArray {

std::vector<std::unique_ptr<blas::Queue>> BLASQueuePool::queues_;

bool BLASQueuePool::initialized() { return !queues_.empty(); }

void BLASQueuePool::initialize() {
  if (initialized()) return;
  queues_.reserve(deviceEnv::instance()->num_streams_total());
  for (std::size_t sidx = 0; sidx != deviceEnv::instance()->num_streams_total();
       ++sidx) {
    auto q = deviceEnv::instance()->stream(
        sidx);  // blaspp forsome reason wants non-const lvalue ref to stream
    queues_.emplace_back(std::make_unique<blas::Queue>(q.device, q.stream));
  }
}

void BLASQueuePool::finalize() { queues_.clear(); }

blas::Queue& BLASQueuePool::queue(std::size_t ordinal) {
  TA_ASSERT(initialized());
  TA_ASSERT(ordinal < deviceEnv::instance()->num_streams_total());
  return *(queues_[ordinal]);
}

blas::Queue& BLASQueuePool::queue(device::Stream const& stream) {
  TA_ASSERT(initialized());
  for (auto&& q : queues_) {
    if (q->device() == stream.device && q->stream() == stream.stream) return *q;
  }
  throw TiledArray::Exception(
      "no matching device stream found in the BLAS queue pool");
}

}  // namespace TiledArray
