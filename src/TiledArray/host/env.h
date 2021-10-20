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

#ifndef TILEDARRAY_HOST_ENV_H__INCLUDED
#define TILEDARRAY_HOST_ENV_H__INCLUDED

#include <TiledArray/config.h>

// for memory management
#include <umpire/Umpire.hpp>
#include <umpire/strategy/QuickPool.hpp>
#include <umpire/strategy/SizeLimiter.hpp>
#include <umpire/strategy/ThreadSafeAllocator.hpp>

#include <TiledArray/external/madness.h>
#include <madness/world/print.h>
#include <madness/world/safempi.h>
#include <madness/world/thread.h>

#include <TiledArray/error.h>

namespace TiledArray {

/**
 * hostEnv set up global environment
 *
 * Singleton class
 */

class hostEnv {
 public:
  ~hostEnv() = default;

  hostEnv(const hostEnv&) = delete;
  hostEnv(hostEnv&&) = delete;
  hostEnv& operator=(const hostEnv&) = delete;
  hostEnv& operator=(hostEnv&&) = delete;

  /// access the instance, if not initialized will be initialized using default
  /// params
  static std::unique_ptr<hostEnv>& instance() {
    if (!instance_accessor()) {
      initialize(TiledArray::get_default_world());
    }
    return instance_accessor();
  }

  /// initialize the instance using explicit params
  static void initialize(World& world,
                         const std::uint64_t max_memory_size = (1ul << 40),
                         const std::uint64_t page_size = (1ul << 22)) {
    // initialize only when not initialized
    if (instance_accessor() == nullptr) {
      // uncomment to debug umpire ops
      //
      //      umpire::util::Logger::getActiveLogger()->setLoggingMsgLevel(
      //          umpire::util::message::Debug);

      //       make thread-safe size-limited pool of host memory

      auto& rm = umpire::ResourceManager::getInstance();

      // turn off Umpire introspection for non-Debug builds
#ifndef NDEBUG
      constexpr auto introspect = true;
#else
      constexpr auto introspect = false;
#endif

      // allocate zero memory for device pool, same grain for subsequent allocs
      auto host_size_limited_alloc =
          rm.makeAllocator<umpire::strategy::SizeLimiter, introspect>(
              "size_limited_alloc", rm.getAllocator("HOST"), max_memory_size);
      auto host_dynamic_pool =
          rm.makeAllocator<umpire::strategy::QuickPool, introspect>(
              "HostDynamicPool", host_size_limited_alloc, 0, page_size);
      auto thread_safe_host_dynamic_pool =
          rm.makeAllocator<umpire::strategy::ThreadSafeAllocator, introspect>(
              "ThreadSafeHostDynamicPool", host_dynamic_pool);

      auto host_env = std::unique_ptr<hostEnv>(
          new hostEnv(world, thread_safe_host_dynamic_pool));
      instance_accessor() = std::move(host_env);
    }
  }

  World& world() const { return *world_; }

  umpire::Allocator& host_allocator() { return host_allocator_; }

 protected:
  hostEnv(World& world, umpire::Allocator host_alloc)
      : world_(&world), host_allocator_(host_alloc) {}

 private:
  // the world used to initialize this
  World* world_;

  /// allocates from a thread-safe, dynamic, size-limited host memory pool
  umpire::Allocator host_allocator_;

  inline static std::unique_ptr<hostEnv>& instance_accessor() {
    static std::unique_ptr<hostEnv> instance_{nullptr};
    return instance_;
  }
};

}  // namespace TiledArray

#endif  // TILEDARRAY_HOST_ENV_H__INCLUDED
