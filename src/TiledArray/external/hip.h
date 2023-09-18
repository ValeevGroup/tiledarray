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

#ifndef TILEDARRAY_EXTERNAL_HIP_H__INCLUDED
#define TILEDARRAY_EXTERNAL_HIP_H__INCLUDED

#include <cassert>
#include <cstdlib>
#include <vector>

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_HIP

#include <hip/hip_runtime.h>

#include <TiledArray/external/umpire.h>

#include <TiledArray/external/madness.h>
#include <madness/world/print.h>
#include <madness/world/safempi.h>
#include <madness/world/thread.h>

#include <TiledArray/error.h>

#define HipSafeCall(err) __hipSafeCall(err, __FILE__, __LINE__)
#define HipSafeCallNoThrow(err) __hipSafeCallNoThrow(err, __FILE__, __LINE__)
#define HipCheckError() __hipCheckError(__FILE__, __LINE__)

inline void __hipSafeCall(hipError_t err, const char* file, const int line) {
  if (hipSuccess != err) {
    std::stringstream ss;
    ss << "hipSafeCall() failed at: " << file << ":" << line << ": ";
    ss << hipGetErrorString(err);
    throw std::runtime_error(ss.str());
  }
}

inline void __hipSafeCallNoThrow(hipError_t err, const char* file,
                                 const int line) {
  if (hipSuccess != err) {
    madness::print_error("hipSafeCallNoThrow() failed at: ", file, ":", line,
                         ": ", hipGetErrorString(err));
  }
}

inline void __hipCheckError(const char* file, const int line) {
  auto err = hipGetLastError();
  if (hipSuccess != err) {
    std::stringstream ss;
    ss << "hipCheckError() failed at: " << file << ":" << line << ": ";
    ss << hipGetErrorString(err);
    throw std::runtime_error(ss.str());
  }
}

namespace TiledArray {

namespace detail {

inline int num_streams() {
  int num_streams = -1;
  char* num_stream_char = std::getenv("TA_HIP_NUM_STREAMS");
  /// default num of streams is 3
  if (num_stream_char) {
    num_streams = std::atoi(num_stream_char);
  } else {
    num_streams = 3;
  }
  return num_streams;
}

inline int num_devices() {
  int num_devices = -1;
  HipSafeCall(hipGetDeviceCount(&num_devices));
  return num_devices;
}

inline int current_device_id(World& world) {
  int mpi_local_size = -1;
  int mpi_local_rank = -1;
  std::tie(mpi_local_rank, mpi_local_size) = mpi_local_rank_size(world);

  int num_devices = detail::num_devices();

  int device_id = -1;
  // devices may already be pre-mapped
  // if mpi_local_size <= num_devices : all ranks are in same resource set, map
  // round robin
  if (mpi_local_size <= num_devices) {
    device_id = mpi_local_rank % num_devices;
  } else {  // mpi_local_size > num_devices
    char* cvd_cstr = std::getenv("HIP_VISIBLE_DEVICES");
    if (cvd_cstr) {  // HIP_VISIBLE_DEVICES is set, assume that pre-mapped
      // make sure that there is only 1 device available here
      if (num_devices != 1) {
        throw std::runtime_error(
            std::string(
                "HIP_VISIBLE_DEVICES environment variable is set, hence using "
                "the provided device-to-rank mapping; BUT TiledArray found ") +
            std::to_string(num_devices) +
            " HIP devices; only 1 HIP device / MPI process is supported");
      }
      device_id = 0;
    } else {  // not enough devices + devices are not pre-mapped
      throw std::runtime_error(
          std::string("TiledArray found ") + std::to_string(mpi_local_size) +
          " MPI ranks on a node with " + std::to_string(num_devices) +
          " HIP devices; only 1 MPI process / HIP device model is currently "
          "supported");
    }
  }

  return device_id;
}

inline void HIPRT_CB hip_readyflag_callback(void* userData) {
  // convert void * to std::atomic<bool>
  std::atomic<bool>* flag = static_cast<std::atomic<bool>*>(userData);
  // set the flag to be true
  flag->store(true);
}

struct ProbeFlag {
  ProbeFlag(std::atomic<bool>* f) : flag(f) {}

  bool operator()() const { return flag->load(); }

  std::atomic<bool>* flag;
};

inline void thread_wait_stream(const hipStream_t& stream) {
  std::atomic<bool>* flag = new std::atomic<bool>(false);

  HipSafeCall(hipLaunchHostFunc(stream, detail::hip_readyflag_callback, flag));

  detail::ProbeFlag probe(flag);

  // wait with sleep and do not do work
  madness::ThreadPool::await(probe, false, true);
  //    madness::ThreadPool::await(probe, true, true);

  delete flag;
}

}  // namespace detail

inline const hipStream_t*& tls_stream_accessor() {
  static thread_local const hipStream_t* thread_local_stream_ptr{nullptr};
  return thread_local_stream_ptr;
}

inline void synchronize_stream(const hipStream_t* stream) {
  tls_stream_accessor() = stream;
}

/**
 * hipEnv maintains the HIP-related part of the runtime environment,
 * such as HIP-specific memory allocators
 *
 * \note this is a Singleton
 */
class hipEnv {
 public:
  ~hipEnv() {
    // destroy streams on current device
    for (auto& stream : streams_) {
      HipSafeCallNoThrow(hipStreamDestroy(stream));
    }
  }

  hipEnv(const hipEnv&) = delete;
  hipEnv(hipEnv&&) = delete;
  hipEnv& operator=(const hipEnv&) = delete;
  hipEnv& operator=(hipEnv&&) = delete;

  /// access the singleton instance; if not initialized will be
  /// initialized via hipEnv::initialize() with the default params
  static std::unique_ptr<hipEnv>& instance() {
    if (!instance_accessor()) {
      initialize();
    }
    return instance_accessor();
  }

  // clang-format off
  /// initialize the instance using explicit params
  /// \param world the world to use for initialization
  /// \param page_size memory added to the pools supporting `this->um_allocator()`, `this->device_allocator()`, and `this->pinned_allocator()` in chunks of at least
  ///                  this size (bytes) [default=2^25]
  /// \param pinned_alloc_limit the maximum total amount of memory (in bytes) that
  ///        allocator returned by `this->pinned_allocator()` can allocate;
  ///        this allocator is not used by default [default=0]
  // clang-format on
  static void initialize(World& world = TiledArray::get_default_world(),
                         const std::uint64_t page_size = (1ul << 25),
                         const std::uint64_t pinned_alloc_limit = (1ul << 40)) {
    static std::mutex mtx;  // to make initialize() reentrant
    std::scoped_lock lock{mtx};
    // only the winner of the lock race gets to initialize
    if (instance_accessor() == nullptr) {
      int num_streams = detail::num_streams();
      int num_devices = detail::num_devices();
      int device_id = detail::current_device_id(world);
      // set device for current MPI process .. will be set in the ctor as well
      HipSafeCall(hipSetDevice(device_id));
      HipSafeCall(hipDeviceSetCacheConfig(hipFuncCachePreferShared));

      // uncomment to debug umpire ops
      //
      //      umpire::util::Logger::getActiveLogger()->setLoggingMsgLevel(
      //          umpire::util::message::Debug);

      //       make Thread Safe UM Dynamic POOL

      auto& rm = umpire::ResourceManager::getInstance();

      auto mem_total_free = hipEnv::memory_total_and_free_device();

      // turn off Umpire introspection for non-Debug builds
#ifndef NDEBUG
      constexpr auto introspect = true;
#else
      constexpr auto introspect = false;
#endif

      // allocate all currently-free memory for UM pool
      auto um_dynamic_pool =
          rm.makeAllocator<umpire::strategy::QuickPool, introspect>(
              "UMDynamicPool", rm.getAllocator("UM"), mem_total_free.second,
              pinned_alloc_limit);

      // allocate zero memory for device pool
      auto dev_size_limited_alloc =
          rm.makeAllocator<umpire::strategy::SizeLimiter, introspect>(
              "size_limited_alloc", rm.getAllocator("DEVICE"),
              mem_total_free.first);
      auto dev_dynamic_pool =
          rm.makeAllocator<umpire::strategy::QuickPool, introspect>(
              "HIPDynamicPool", dev_size_limited_alloc, 0, pinned_alloc_limit);

      // allocate pinned_alloc_limit in pinned memory
      auto pinned_size_limited_alloc =
          rm.makeAllocator<umpire::strategy::SizeLimiter, introspect>(
              "SizeLimited_PINNED", rm.getAllocator("PINNED"),
              pinned_alloc_limit);
      auto pinned_dynamic_pool =
          rm.makeAllocator<umpire::strategy::QuickPool, introspect>(
              "QuickPool_SizeLimited_PINNED", pinned_size_limited_alloc,
              page_size, page_size, /* alignment */ TILEDARRAY_ALIGN_SIZE);

      auto hip_env = std::unique_ptr<hipEnv>(
          new hipEnv(world, num_devices, device_id, num_streams,
                     um_dynamic_pool, dev_dynamic_pool, pinned_dynamic_pool));
      instance_accessor() = std::move(hip_env);
    }
  }

  World& world() const { return *world_; }

  int num_devices() const { return num_devices_; }

  int current_device_id() const { return current_device_id_; }

  int num_streams() const { return num_streams_; }

  bool concurrent_managed_access() const {
    return device_concurrent_managed_access_;
  }

  size_t stream_id(const hipStream_t& stream) const {
    auto it = std::find(streams_.begin(), streams_.end(), stream);
    if (it == streams_.end()) abort();
    return it - streams_.begin();
  }

  /// @return the total size of all and free device memory on the current device
  static std::pair<size_t, size_t> memory_total_and_free_device() {
    std::pair<size_t, size_t> result;
    // N.B. hipMemGetInfo returns {free,total}
    HipSafeCall(hipMemGetInfo(&result.second, &result.first));
    return result;
  }

  /// Collective call to probe HIP {total,free} memory

  /// @return the total size of all and free device memory on every rank's
  /// device
  std::vector<std::pair<size_t, size_t>> memory_total_and_free() const {
    auto world_size = world_->size();
    std::vector<size_t> total_memory(world_size, 0), free_memory(world_size, 0);
    auto rank = world_->rank();
    std::tie(total_memory.at(rank), free_memory.at(rank)) =
        hipEnv::memory_total_and_free_device();
    world_->gop.sum(total_memory.data(), total_memory.size());
    world_->gop.sum(free_memory.data(), free_memory.size());
    std::vector<std::pair<size_t, size_t>> result(world_size);
    for (int r = 0; r != world_size; ++r) {
      result.at(r) = {total_memory.at(r), free_memory.at(r)};
    }
    return result;
  }

  const hipStream_t& stream(std::size_t i) const { return streams_.at(i); }

  const hipStream_t& stream_h2d() const { return streams_[num_streams_]; }

  const hipStream_t& stream_d2h() const { return streams_[num_streams_ + 1]; }

  /// @return a (non-thread-safe) Umpire allocator for device UM
  umpire::Allocator& um_allocator() { return um_allocator_; }

  // clang-format off
  /// @return the max actual amount of memory held by um_allocator()
  /// @details returns the value provided by `umpire::strategy::QuickPool::getHighWatermark()`
  /// @note if there is only 1 Umpire allocator using UM memory should be identical to the value returned by `umpire::ResourceManager::getInstance().getAllocator("UM").getHighWatermark()`
  // clang-format on
  std::size_t um_allocator_getActualHighWatermark() {
    TA_ASSERT(dynamic_cast<umpire::strategy::QuickPool*>(
                  um_allocator_.getAllocationStrategy()) != nullptr);
    return dynamic_cast<umpire::strategy::QuickPool*>(
               um_allocator_.getAllocationStrategy())
        ->getActualHighwaterMark();
  }

  /// @return a (non-thread-safe) Umpire allocator for device memory
  umpire::Allocator& device_allocator() { return device_allocator_; }

  // clang-format off
  /// @return the max actual amount of memory held by um_allocator()
  /// @details returns the value provided by `umpire::strategy::QuickPool::getHighWatermark()`
  /// @note if there is only 1 Umpire allocator using DEVICE memory should be identical to the value returned by `umpire::ResourceManager::getInstance().getAllocator("DEVICE").getHighWatermark()`
  // clang-format on
  std::size_t device_allocator_getActualHighWatermark() {
    TA_ASSERT(dynamic_cast<umpire::strategy::QuickPool*>(
                  device_allocator_.getAllocationStrategy()) != nullptr);
    return dynamic_cast<umpire::strategy::QuickPool*>(
               device_allocator_.getAllocationStrategy())
        ->getActualHighwaterMark();
  }

  /// @return an Umpire allocator that allocates from a
  ///         pinned memory pool
  /// @warning this is not a thread-safe allocator, should be only used when
  ///          wrapped into umpire_allocator_impl
  umpire::Allocator& pinned_allocator() { return pinned_allocator_; }

  // clang-format off
  /// @return the max actual amount of memory held by pinned_allocator()
  /// @details returns the value provided by `umpire::strategy::QuickPool::getHighWatermark()`
  /// @note if there is only 1 Umpire allocator using PINNED memory this should be identical to the value returned by `umpire::ResourceManager::getInstance().getAllocator("PINNED").getHighWatermark()`
  // clang-format on
  std::size_t pinned_allocator_getActualHighWatermark() {
    TA_ASSERT(dynamic_cast<umpire::strategy::QuickPool*>(
                  pinned_allocator_.getAllocationStrategy()) != nullptr);
    return dynamic_cast<umpire::strategy::QuickPool*>(
               pinned_allocator_.getAllocationStrategy())
        ->getActualHighwaterMark();
  }

 protected:
  hipEnv(World& world, int num_devices, int device_id, int num_streams,
         umpire::Allocator um_alloc, umpire::Allocator device_alloc,
         umpire::Allocator pinned_alloc)
      : world_(&world),
        um_allocator_(um_alloc),
        device_allocator_(device_alloc),
        pinned_allocator_(pinned_alloc),
        num_devices_(num_devices),
        current_device_id_(device_id),
        num_streams_(num_streams) {
    if (num_devices <= 0) {
      throw std::runtime_error("No HIP-Enabled GPUs Found!\n");
    }

    // set device for current MPI process
    HipSafeCall(hipSetDevice(current_device_id_));

    /// check the capability of HIP device
    hipDeviceProp prop;
    HipSafeCall(hipGetDeviceProperties(&prop, device_id));
    if (!prop.managedMemory) {
      throw std::runtime_error("HIP Device doesn't support managedMemory\n");
    }
    int concurrent_managed_access;
    HipSafeCall(hipDeviceGetAttribute(&concurrent_managed_access,
                                      hipDeviceAttributeConcurrentManagedAccess,
                                      device_id));
    device_concurrent_managed_access_ = concurrent_managed_access;
    if (!device_concurrent_managed_access_) {
      std::cout << "\nWarning: HIP Device doesn't support "
                   "ConcurrentManagedAccess!\n\n";
    }

    // creates streams on current device
    streams_.resize(num_streams_ + 2);
    for (auto& stream : streams_) {
      HipSafeCall(hipStreamCreateWithFlags(&stream, hipStreamNonBlocking));
    }
    std::cout << "created " << num_streams_ << " HIP streams + 2 I/O streams"
              << std::endl;
  }

 private:
  // the world used to initialize this
  World* world_;

  /// allocator backed by a (non-thread-safe) dynamically-sized pool for UM
  umpire::Allocator um_allocator_;
  /// allocator backed by a (non-thread-safe) dynamically-sized pool for device
  /// memory
  umpire::Allocator device_allocator_;
  // allocates from a dynamic, size-limited pinned memory pool
  // N.B. not thread safe, so must be wrapped into umpire_allocator_impl
  umpire::Allocator pinned_allocator_;

  int num_devices_;
  int current_device_id_;
  bool device_concurrent_managed_access_;

  int num_streams_;
  std::vector<hipStream_t> streams_;

  inline static std::unique_ptr<hipEnv>& instance_accessor() {
    static std::unique_ptr<hipEnv> instance_{nullptr};
    return instance_;
  }
};

namespace detail {

template <typename Range>
const hipStream_t& get_stream_based_on_range(const Range& range) {
  // TODO better way to get stream based on the id of tensor
  auto stream_id = range.offset() % hipEnv::instance()->num_streams();
  auto& stream = hipEnv::instance()->stream(stream_id);
  return stream;
}

}  // namespace detail

}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_HIP

#endif  // TILEDARRAY_EXTERNAL_HIP_H__INCLUDED
