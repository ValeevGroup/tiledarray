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
#include <umpire_cxx_allocator.hpp>

#include <TiledArray/external/madness.h>

namespace TiledArray {

namespace detail {

struct get_host_allocator {
  umpire::Allocator& operator()();
};

}  // namespace detail

namespace host {

/**
 * Env maintains the (host-side, as opposed to device-side) environment,
 * such as memory allocators
 *
 * \note this is a Singleton
 */
class Env {
 public:
  ~Env() = default;

  Env(const Env&) = delete;
  Env(Env&&) = delete;
  Env& operator=(const Env&) = delete;
  Env& operator=(Env&&) = delete;

  /// access the singleton instance; if not initialized will be
  /// initialized via Env::initialize() with the default params
  static std::unique_ptr<Env>& instance();

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
                         const std::uint64_t page_size = (1ul << 25));

  World& world() const;

  /// @return an Umpire allocator that allocates from a
  ///         host memory pool
  /// @warning this is not a thread-safe allocator, should be only used when
  ///          wrapped into umpire_based_allocator_impl
  umpire::Allocator& host_allocator();

  // clang-format off
  /// @return the max actual amount of memory held by host_allocator()
  /// @details returns the value provided by `umpire::strategy::QuickPool::getHighWatermark()`
  /// @note if there is only 1 Umpire allocator using HOST memory this should be identical to the value returned by `umpire::ResourceManager::getInstance().getAllocator("HOST").getHighWatermark()`
  // clang-format on
  std::size_t host_allocator_getActualHighWatermark();

 protected:
  Env(World& world, umpire::Allocator host_alloc);

 private:
  // the world used to initialize this
  World* world_;

  // allocates from a dynamic, size-limited host memory pool
  // N.B. not thread safe, so must be wrapped into umpire_based_allocator_impl
  umpire::Allocator host_allocator_;

  inline static std::unique_ptr<Env>& instance_accessor();
};

}  // namespace host

}  // namespace TiledArray

#endif  // TILEDARRAY_HOST_ENV_H__INCLUDED
