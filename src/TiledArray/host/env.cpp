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
 *  Chong Peng
 *  Department of Chemistry, Virginia Tech
 *  July 23, 2018
 *
 */

#include <TiledArray/host/env.h>

#include <TiledArray/error.h>

#include <umpire/strategy/QuickPool.hpp>
#include <umpire/strategy/SizeLimiter.hpp>

namespace TiledArray {

namespace detail {

umpire::Allocator& get_host_allocator::operator()() {
  return TiledArray::host::Env::instance()->host_allocator();
}

}  // namespace detail

namespace host {

std::unique_ptr<Env>& Env::instance() {
  if (!instance_accessor()) {
    initialize();
  }
  return instance_accessor();
}

void Env::initialize(World& world, const std::uint64_t host_alloc_limit,
                     const std::uint64_t page_size) {
  static std::mutex mtx;  // to make initialize() reentrant
  std::scoped_lock lock{mtx};
  // only the winner of the lock race gets to initialize
  if (instance_accessor() == nullptr) {
    // uncomment to debug umpire ops
    //
    //      umpire::util::Logger::getActiveLogger()->setLoggingMsgLevel(
    //          umpire::util::message::Debug);

    //       make thread-safe size-limited pool of host memory

    auto& rm = umpire::ResourceManager::getInstance();

    // N.B. we don't rely on Umpire introspection (even for profiling)
    constexpr auto introspect = false;

    // use QuickPool for host memory allocation, with min grain of 1 page
    auto host_size_limited_alloc =
        rm.makeAllocator<umpire::strategy::SizeLimiter, introspect>(
            "SizeLimited_HOST", rm.getAllocator("HOST"), host_alloc_limit);
    auto host_dynamic_pool =
        rm.makeAllocator<umpire::strategy::QuickPool, introspect>(
            "QuickPool_SizeLimited_HOST", host_size_limited_alloc, page_size,
            page_size, /* alignment */ TILEDARRAY_ALIGN_SIZE);

    auto host_env = std::unique_ptr<Env>(new Env(world, host_dynamic_pool));
    instance_accessor() = std::move(host_env);
  }
}

World& Env::world() const { return *world_; }

umpire::Allocator& Env::host_allocator() { return host_allocator_; }

std::size_t Env::host_allocator_getActualHighWatermark() {
  TA_ASSERT(dynamic_cast<umpire::strategy::QuickPool*>(
                host_allocator_.getAllocationStrategy()) != nullptr);
  return dynamic_cast<umpire::strategy::QuickPool*>(
             host_allocator_.getAllocationStrategy())
      ->getActualHighwaterMark();
}

Env::Env(World& world, umpire::Allocator host_alloc)
    : world_(&world), host_allocator_(host_alloc) {}

std::unique_ptr<Env>& Env::instance_accessor() {
  static std::unique_ptr<Env> instance_{nullptr};
  return instance_;
}

}  // namespace host

}  // namespace TiledArray
