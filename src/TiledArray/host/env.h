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
#include <umpire/strategy/AlignedAllocator.hpp>
#include <umpire/strategy/QuickPool.hpp>
#include <umpire/strategy/SizeLimiter.hpp>

#include <TiledArray/external/madness.h>
#include <madness/world/print.h>
#include <madness/world/safempi.h>
#include <madness/world/thread.h>

#include <TiledArray/error.h>

namespace TiledArray {

/**
 * hostEnv maintains the (host-side, as opposed to device-side) environment,
 * such as memory allocators
 *
 * \note this is a Singleton
 */
class hostEnv {
 public:
  ~hostEnv() = default;

  hostEnv(const hostEnv&) = delete;
  hostEnv(hostEnv&&) = delete;
  hostEnv& operator=(const hostEnv&) = delete;
  hostEnv& operator=(hostEnv&&) = delete;

  /// access the singleton instance; if not initialized will be
  /// initialized via hostEnv::initialize() with the default params
  static std::unique_ptr<hostEnv>& instance() {
    if (!instance_accessor()) {
      initialize();
    }
    return instance_accessor();
  }

  // clang-format off
  /// initialize the instance using explicit params
  /// \param world the world to use for initialization
  /// \param host_alloc_limit the maximum total amount of memory (in bytes) that
  ///        allocator returned by `this->host_allocator()` can allocate;
  ///        this allocator is used by TiledArray for storage of TA::Tensor objects (these by default
  ///        store DistArray tile data and (if sparse) shape [default=2^40]
  /// \param page_size memory added to the pool in chunks of at least
  ///                  this size (bytes) [default=2^25]
  // clang-format on
  static void initialize(World& world = TiledArray::get_default_world(),
                         const std::uint64_t host_alloc_limit = (1ul << 40),
                         const std::uint64_t page_size = (1ul << 25)) {
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

      auto host_env =
          std::unique_ptr<hostEnv>(new hostEnv(world, host_dynamic_pool));
      instance_accessor() = std::move(host_env);
    }
  }

  World& world() const { return *world_; }

  /// @return an Umpire allocator that allocates from a
  ///         host memory pool
  /// @warning this is not a thread-safe allocator, should be only used when
  ///          wrapped into umpire_based_allocator_impl
  umpire::Allocator& host_allocator() { return host_allocator_; }

  // clang-format off
  /// @return the max actual amount of memory held by host_allocator()
  /// @details returns the value provided by `umpire::strategy::QuickPool::getHighWatermark()`
  /// @note if there is only 1 Umpire allocator using HOST memory this should be identical to the value returned by `umpire::ResourceManager::getInstance().getAllocator("HOST").getHighWatermark()`
  // clang-format on
  std::size_t host_allocator_getActualHighWatermark() {
    TA_ASSERT(dynamic_cast<umpire::strategy::QuickPool*>(
                  host_allocator_.getAllocationStrategy()) != nullptr);
    return dynamic_cast<umpire::strategy::QuickPool*>(
               host_allocator_.getAllocationStrategy())
        ->getActualHighwaterMark();
  }

 protected:
  hostEnv(World& world, umpire::Allocator host_alloc)
      : world_(&world), host_allocator_(host_alloc) {}

 private:
  // the world used to initialize this
  World* world_;

  // allocates from a dynamic, size-limited host memory pool
  // N.B. not thread safe, so must be wrapped into umpire_based_allocator_impl
  umpire::Allocator host_allocator_;

  inline static std::unique_ptr<hostEnv>& instance_accessor() {
    static std::unique_ptr<hostEnv> instance_{nullptr};
    return instance_;
  }
};

namespace detail {

struct get_host_allocator {
  umpire::Allocator& operator()() {
    return hostEnv::instance()->host_allocator();
  }
};

}  // namespace detail

}  // namespace TiledArray

#endif  // TILEDARRAY_HOST_ENV_H__INCLUDED
