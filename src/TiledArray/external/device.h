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

#ifndef TILEDARRAY_EXTERNAL_DEVICE_H__INCLUDED
#define TILEDARRAY_EXTERNAL_DEVICE_H__INCLUDED

#include <cassert>
#include <cstdlib>
#include <optional>
#include <vector>

#include <TiledArray/config.h>

#if defined(TILEDARRAY_HAS_HIP)
#include <hip/hip_runtime.h>
#elif defined(TILEDARRAY_HAS_CUDA)
#include <cuda.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>
#endif

#include <TiledArray/external/umpire.h>

#include <TiledArray/external/madness.h>
#include <madness/world/print.h>
#include <madness/world/safempi.h>
#include <madness/world/thread.h>

#include <TiledArray/error.h>
#include <TiledArray/initialize.h>

#if defined(TILEDARRAY_HAS_CUDA)

inline void __DeviceSafeCall(cudaError err, const char* file, const int line) {
  if (cudaSuccess != err) {
    std::stringstream ss;
    ss << "DeviceSafeCall() failed at: " << file << ":" << line;
    std::string what = ss.str();
    throw thrust::system_error(err, thrust::cuda_category(), what);
  }
}

inline void __cudaSafeCallNoThrow(cudaError err, const char* file,
                                  const int line) {
  if (cudaSuccess != err) {
    madness::print_error("cudaSafeCallNoThrow() failed at: ", file, ":", line);
  }
}

inline void __cudaCheckError(const char* file, const int line) {
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    std::stringstream ss;
    ss << "cudaCheckError() failed at: " << file << ":" << line;
    std::string what = ss.str();
    throw thrust::system_error(err, thrust::cuda_category(), what);
  }
}

#define DeviceSafeCall(err) __DeviceSafeCall(err, __FILE__, __LINE__)
#define DeviceSafeCallNoThrow(err) \
  __cudaSafeCallNoThrow(err, __FILE__, __LINE__)
#define DeviceCheckError() __cudaCheckError(__FILE__, __LINE__)

#elif defined(TILEDARRAY_HAS_HIP)

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

#define DeviceSafeCall(err) __hipSafeCall(err, __FILE__, __LINE__)
#define DeviceSafeCallNoThrow(err) __hipSafeCallNoThrow(err, __FILE__, __LINE__)
#define DeviceCheckError() __hipCheckError(__FILE__, __LINE__)

#endif

namespace TiledArray {
namespace device {

#if defined(TILEDARRAY_HAS_CUDA)
inline namespace cuda {
using stream_t = cudaStream_t;
using error_t = cudaError_t;
using hostFn_t = cudaHostFn_t;
using deviceProp_t = cudaDeviceProp;
using deviceAttr_t = cudaDeviceAttr;
#define DeviceAttributeConcurrentManagedAccess \
  cudaDevAttrConcurrentManagedAccess
#define DEVICERT_CB CUDART_CB

const inline auto Success = cudaSuccess;

enum DeviceId {
  CpuDeviceId = cudaCpuDeviceId,
  InvalidDeviceId = cudaInvalidDeviceId
};

enum MemAttach {
  MemAttachGlobal = cudaMemAttachGlobal,
  MemAttachHost = cudaMemAttachHost,
  MemAttachSingle = cudaMemAttachSingle
};

enum MemcpyKind {
  MemcpyHostToHost = cudaMemcpyHostToHost,
  MemcpyHostToDevice = cudaMemcpyHostToDevice,
  MemcpyDeviceToHost = cudaMemcpyDeviceToHost,
  MemcpyDeviceToDevice = cudaMemcpyDeviceToDevice,
  MemcpyDefault = cudaMemcpyDefault
};

enum FuncCache {
  FuncCachePreferNone = cudaFuncCachePreferNone,
  FuncCachePreferShared = cudaFuncCachePreferShared,
  FuncCachePreferL1 = cudaFuncCachePreferL1,
  FuncCachePreferEqual = cudaFuncCachePreferEqual
};

enum StreamCreateFlags {
  StreamDefault = cudaStreamDefault,
  StreamNonBlocking = cudaStreamNonBlocking
};

constexpr inline auto DevAttrUnifiedAddressing = cudaDevAttrUnifiedAddressing;
constexpr inline auto DevAttrConcurrentManagedAccess =
    cudaDevAttrConcurrentManagedAccess;

inline error_t driverVersion(int* v) { return cudaDriverGetVersion(v); }

inline error_t runtimeVersion(int* v) { return cudaRuntimeGetVersion(v); }

inline error_t setDevice(int device) { return cudaSetDevice(device); }

inline error_t getDevice(int* device) { return cudaGetDevice(device); }

inline error_t getDeviceCount(int* num_devices) {
  return cudaGetDeviceCount(num_devices);
}

inline error_t deviceSetCacheConfig(FuncCache cache_config) {
  return cudaDeviceSetCacheConfig(static_cast<cudaFuncCache>(cache_config));
}

inline error_t memGetInfo(size_t* free, size_t* total) {
  return cudaMemGetInfo(free, total);
}

inline error_t getDeviceProperties(deviceProp_t* prop, int device) {
  return cudaGetDeviceProperties(prop, device);
}

inline error_t deviceGetAttribute(int* value, deviceAttr_t attr, int device) {
  return cudaDeviceGetAttribute(value, attr, device);
}

inline error_t streamCreate(stream_t* pStream) {
  return cudaStreamCreate(pStream);
}

inline error_t streamCreateWithFlags(stream_t* pStream,
                                     StreamCreateFlags flags) {
  return cudaStreamCreateWithFlags(pStream, flags);
}

inline error_t deviceSynchronize() { return cudaDeviceSynchronize(); }
inline error_t streamSynchronize(stream_t stream) {
  return cudaStreamSynchronize(stream);
}

template <typename T>
inline error_t malloc(T** devPtr, size_t size) {
  return cudaMalloc(devPtr, size);
}

template <typename T>
inline error_t mallocHost(T** devPtr, size_t size) {
  return cudaMallocHost(devPtr, size);
}

template <typename T>
inline error_t mallocManaged(T** devPtr, size_t size,
                             unsigned int flag = MemAttachGlobal) {
  return cudaMallocManaged(devPtr, size, flag);
}

template <typename T>
error_t free(T* devPtr) {
  return cudaFree(devPtr);
}

template <typename T>
error_t memcpy(T* dst, const T* src, size_t count, MemcpyKind kind) {
  return cudaMemcpy(dst, src, count, static_cast<cudaMemcpyKind>(kind));
}

template <typename T>
error_t memcpyAsync(T* dst, const T* src, size_t count, MemcpyKind kind,
                    stream_t stream = 0) {
  return cudaMemcpyAsync(dst, src, count, static_cast<cudaMemcpyKind>(kind),
                         stream);
}

template <typename T>
error_t memPrefetchAsync(const T* devPtr, size_t count, int dstDevice,
                         stream_t stream = 0) {
  return cudaMemPrefetchAsync(devPtr, count, dstDevice, stream);
}

inline error_t launchHostFunc(stream_t stream, hostFn_t fn, void* userData) {
  return cudaLaunchHostFunc(stream, fn, userData);
}

inline error_t streamDestroy(stream_t stream) {
  return cudaStreamDestroy(stream);
}

}  // namespace cuda
#elif defined(TILEDARRAY_HAS_HIP)
inline namespace hip {
using stream_t = hipStream_t;
using error_t = hipError_t;
using hostFn_t = hipHostFn_t;
using deviceProp_t = hipDeviceProp_t;
using deviceAttr_t = hipDeviceAttribute_t;
#define DeviceAttributeConcurrentManagedAccess \
  hipDeviceAttributeConcurrentManagedAccess
#define DEVICERT_CB

const inline auto Success = hipSuccess;

enum DeviceId {
  CpuDeviceId = hipCpuDeviceId,
  InvalidDeviceId = hipInvalidDeviceId
};

enum MemcpyKind {
  MemcpyHostToHost = hipMemcpyHostToHost,
  MemcpyHostToDevice = hipMemcpyHostToDevice,
  MemcpyDeviceToHost = hipMemcpyDeviceToHost,
  MemcpyDeviceToDevice = hipMemcpyDeviceToDevice,
  MemcpyDefault = hipMemcpyDefault
};

enum MemAttach {
  MemAttachGlobal = hipMemAttachGlobal,
  MemAttachHost = hipMemAttachHost,
  MemAttachSingle = hipMemAttachSingle
};

enum FuncCache {
  FuncCachePreferNone = hipFuncCachePreferNone,
  FuncCachePreferShared = hipFuncCachePreferShared,
  FuncCachePreferL1 = hipFuncCachePreferL1,
  FuncCachePreferEqual = hipFuncCachePreferEqual
};

enum StreamCreateFlags {
  StreamDefault = hipStreamDefault,
  StreamNonBlocking = hipStreamNonBlocking
};

constexpr inline auto DevAttrUnifiedAddressing =
    hipDeviceAttributeUnifiedAddressing;
constexpr inline auto DevAttrConcurrentManagedAccess =
    hipDeviceAttributeConcurrentManagedAccess;

inline error_t driverVersion(int* v) { return hipDriverGetVersion(v); }

inline error_t runtimeVersion(int* v) { return hipRuntimeGetVersion(v); }

inline error_t setDevice(int device) { return hipSetDevice(device); }

inline error_t getDevice(int* device) { return hipGetDevice(device); }

inline error_t getDeviceCount(int* num_devices) {
  return hipGetDeviceCount(num_devices);
}

inline error_t deviceSetCacheConfig(FuncCache cache_config) {
  return hipDeviceSetCacheConfig(static_cast<hipFuncCache_t>(cache_config));
}

inline error_t memGetInfo(size_t* free, size_t* total) {
  return hipMemGetInfo(free, total);
}

inline error_t getDeviceProperties(deviceProp_t* prop, int device) {
  return hipGetDeviceProperties(prop, device);
}

inline error_t deviceGetAttribute(int* value, deviceAttr_t attr, int device) {
  return hipDeviceGetAttribute(value, attr, device);
}

inline error_t streamCreate(stream_t* pStream) {
  return hipStreamCreate(pStream);
}

inline error_t streamCreateWithFlags(stream_t* pStream,
                                     StreamCreateFlags flags) {
  return hipStreamCreateWithFlags(pStream, flags);
}

inline error_t deviceSynchronize() { return hipDeviceSynchronize(); }

inline error_t streamSynchronize(stream_t stream) {
  return hipStreamSynchronize(stream);
}

template <typename T>
inline error_t malloc(T** devPtr, size_t size) {
  return hipMalloc(devPtr, size);
}

template <typename T>
inline error_t mallocHost(T** devPtr, size_t size) {
  return hipMallocHost(devPtr, size);
}

template <typename T>
inline error_t mallocManaged(T** devPtr, size_t size,
                             unsigned int flag = MemAttachGlobal) {
  return hipMallocManaged(devPtr, size, flag);
}

template <typename T>
error_t free(T* devPtr) {
  return hipFree(devPtr);
}

template <typename T>
error_t memcpy(T* dst, const T* src, size_t count, MemcpyKind kind) {
  return hipMemcpy(dst, src, count, static_cast<hipMemcpyKind>(kind));
}

template <typename T>
error_t memcpyAsync(T* dst, const T* src, size_t count, MemcpyKind kind,
                    stream_t stream = 0) {
  return hipMemcpyAsync(dst, src, count, static_cast<hipMemcpyKind>(kind),
                        stream);
}

template <typename T>
error_t memPrefetchAsync(const T* devPtr, size_t count, int dstDevice,
                         stream_t stream = 0) {
  return hipMemPrefetchAsync(devPtr, count, dstDevice, stream);
}

inline error_t launchHostFunc(stream_t stream, hostFn_t fn, void* userData) {
  return hipLaunchHostFunc(stream, fn, userData);
}

inline error_t streamDestroy(stream_t stream) {
  return hipStreamDestroy(stream);
}

}  // namespace hip
#endif

#ifdef TILEDARRAY_HAS_DEVICE

inline int num_streams_per_device() {
  int num_streams = -1;
  char* num_stream_char = std::getenv("TA_DEVICE_NUM_STREAMS");
  if (num_stream_char) {
    num_streams = std::atoi(num_stream_char);
  } else {
#if defined(TILEDARRAY_HAS_CUDA)
    char* num_stream_char = std::getenv("TA_CUDA_NUM_STREAMS");
#elif defined(TILEDARRAY_HAS_HIP)
    char* num_stream_char = std::getenv("TA_HIP_NUM_STREAMS");
#endif
    if (num_stream_char) {
      num_streams = std::atoi(num_stream_char);
    } else {
      /// default num of streams is 3
      num_streams = 3;
    }
  }
  return num_streams;
}

inline void DEVICERT_CB readyflag_callback(void* userData) {
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

inline void thread_wait_stream(const stream_t& stream) {
  std::atomic<bool>* flag = new std::atomic<bool>(false);

  DeviceSafeCall(launchHostFunc(stream, readyflag_callback, flag));

  ProbeFlag probe(flag);

  // wait with sleep and do not do work
  madness::ThreadPool::await(probe, false, true);
  //    madness::ThreadPool::await(probe, true, true);

  delete flag;
}

/// Stream is a `{device, stream_t}` pair, i.e. the analog of blas::Queue.
/// It exists as a syntactic sugar around stream_t, and to avoid the need
/// to deduce the device from stream
/// \internal did not name it queue to avoid naming dichotomies
/// all over the place
struct Stream {
  int device;
  stream_t stream;
  Stream(int device, stream_t stream) : device(device), stream(stream) {}

  /// Stream is implicitly convertible to stream
  operator stream_t() const { return stream; }
};

/**
 * Env maintains the device-related part of the runtime environment,
 * such as specialized data structures like device streams and memory allocators
 *
 * \note this is a Singleton
 */
class Env {
 public:
  ~Env() {
    // destroy streams on current device
    for (auto& [device, stream] : streams_) {
      DeviceSafeCallNoThrow(streamDestroy(stream));
    }
  }

  Env(const Env&) = delete;
  Env(Env&&) = delete;
  Env& operator=(const Env&) = delete;
  Env& operator=(Env&&) = delete;

  /// access the singleton instance; if not initialized will be
  /// initialized via Env::initialize() with the default params
  static std::unique_ptr<Env>& instance() {
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
      int num_streams_per_device = device::num_streams_per_device();
      const int num_visible_devices = []() {
        int num_visible_devices = -1;
        DeviceSafeCall(getDeviceCount(&num_visible_devices));
        return num_visible_devices;
      }();
      const auto compute_devices = [num_visible_devices](World& world) {
        std::vector<int> compute_devices;
        static const std::tuple<int, int> local_rank_size =
            TiledArray::detail::mpi_local_rank_size(world);
        const auto& [mpi_local_rank, mpi_local_size] = local_rank_size;
        // map ranks to default device round robin
        int device_id = mpi_local_rank % num_visible_devices;
        while (device_id < num_visible_devices) {
          compute_devices.push_back(device_id);
          device_id += mpi_local_size;
        }

        return compute_devices;
      }(world);

      // configure devices for this rank
      for (auto device : compute_devices) {
        DeviceSafeCall(setDevice(device));
        DeviceSafeCall(deviceSetCacheConfig(FuncCachePreferShared));
      }
      // use the first device as default:
      DeviceSafeCall(setDevice(compute_devices[0]));

      // uncomment to debug umpire ops
      //
      //      umpire::util::Logger::getActiveLogger()->setLoggingMsgLevel(
      //          umpire::util::message::Debug);

      //       make Thread Safe UM Dynamic POOL

      auto& rm = umpire::ResourceManager::getInstance();

      auto mem_total_free = Env::memory_total_and_free_device();

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
              "DEVICEDynamicPool", dev_size_limited_alloc, 0,
              pinned_alloc_limit);

      // allocate pinned_alloc_limit in pinned memory
      auto pinned_size_limited_alloc =
          rm.makeAllocator<umpire::strategy::SizeLimiter, introspect>(
              "SizeLimited_PINNED", rm.getAllocator("PINNED"),
              pinned_alloc_limit);
      auto pinned_dynamic_pool =
          rm.makeAllocator<umpire::strategy::QuickPool, introspect>(
              "QuickPool_SizeLimited_PINNED", pinned_size_limited_alloc,
              page_size, page_size, /* alignment */ TILEDARRAY_ALIGN_SIZE);

      auto env = std::unique_ptr<Env>(new Env(
          world, num_visible_devices, compute_devices, num_streams_per_device,
          um_dynamic_pool, dev_dynamic_pool, pinned_dynamic_pool));
      instance_accessor() = std::move(env);
    }
  }

  World& world() const { return *world_; }

  /// @return the number of devices visible to this rank
  int num_visible_devices() const { return num_devices_visible_; }

  /// @return the number of compute devices assigned to this rank
  int num_compute_devices() const { return compute_devices_.size(); }

  /// @return the device pointed to by the currently-active device runtime
  /// context
  int current_device_id() const {
    TA_ASSERT(num_compute_devices() > 0);
    int current_device = -1;
    DeviceSafeCall(getDevice(&current_device));
    return current_device;
  }

  /// @return the total number of compute streams (for all devices)
  /// visible to this rank
  int num_streams_total() const { return streams_.size(); }

  bool concurrent_managed_access() const {
    return device_concurrent_managed_access_;
  }

  size_t stream_id(const stream_t& stream) const {
    auto it = std::find(streams_.begin(), streams_.end(), stream);
    if (it == streams_.end()) abort();
    return it - streams_.begin();
  }

  /// @return the total size of all and free device memory on the current device
  static std::pair<size_t, size_t> memory_total_and_free_device() {
    std::pair<size_t, size_t> result;
    // N.B. *MemGetInfo returns {free,total}
    DeviceSafeCall(memGetInfo(&result.second, &result.first));
    return result;
  }

  /// Collective call to probe device {total,free} memory

  /// @return the total size of all and free device memory on every rank's
  /// device
  std::vector<std::pair<size_t, size_t>> memory_total_and_free() const {
    auto world_size = world_->size();
    std::vector<size_t> total_memory(world_size, 0), free_memory(world_size, 0);
    auto rank = world_->rank();
    std::tie(total_memory.at(rank), free_memory.at(rank)) =
        Env::memory_total_and_free_device();
    world_->gop.sum(total_memory.data(), total_memory.size());
    world_->gop.sum(free_memory.data(), free_memory.size());
    std::vector<std::pair<size_t, size_t>> result(world_size);
    for (int r = 0; r != world_size; ++r) {
      result.at(r) = {total_memory.at(r), free_memory.at(r)};
    }
    return result;
  }

  /// @param[in] i compute stream ordinal
  /// @pre `i<num_stream_total()`
  /// @return `i`th compute stream
  const Stream& stream(std::size_t i) const {
    TA_ASSERT(i < this->num_streams_total());
    return streams_[i];
  }

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
  ///          wrapped into umpire_based_allocator_impl
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
  Env(World& world, int num_visible_devices, std::vector<int> compute_devices,
      int num_streams_per_device, umpire::Allocator um_alloc,
      umpire::Allocator device_alloc, umpire::Allocator pinned_alloc)
      : world_(&world),
        um_allocator_(um_alloc),
        device_allocator_(device_alloc),
        pinned_allocator_(pinned_alloc),
        num_devices_visible_(num_visible_devices),
        compute_devices_(std::move(compute_devices)),
        num_streams_per_device_(num_streams_per_device) {
    if (compute_devices_.size() <= 0) {
      throw std::runtime_error("No " TILEDARRAY_DEVICE_RUNTIME_STR
                               " compute devices found!\n");
    }

    streams_.reserve(num_streams_per_device_ * compute_devices_.size());

    /// ensure the desired capabilities of each device
    for (auto device : compute_devices_) {
      deviceProp_t prop;
      DeviceSafeCall(getDeviceProperties(&prop, device));
      if (!prop.managedMemory) {
        throw std::runtime_error(TILEDARRAY_DEVICE_RUNTIME_STR
                                 "device doesn't support managedMemory\n");
      }
      int concurrent_managed_access;
      DeviceSafeCall(deviceGetAttribute(&concurrent_managed_access,
                                        DeviceAttributeConcurrentManagedAccess,
                                        device));
      device_concurrent_managed_access_ =
          device_concurrent_managed_access_ && concurrent_managed_access;
      if (!initialized_to_be_quiet() && !device_concurrent_managed_access_) {
        std::cout << "\nWarning: " TILEDARRAY_DEVICE_RUNTIME_STR
                     " device doesn't support "
                     "ConcurrentManagedAccess!\n\n";
      }

      // creates streams on current device
      DeviceSafeCall(setDevice(device));
      for (int s = 0; s != num_streams_per_device_; ++s) {
        stream_t stream;
        DeviceSafeCall(streamCreateWithFlags(&stream, StreamNonBlocking));
        streams_.emplace_back(device, stream);
      }
    }

    if (!initialized_to_be_quiet() && world.rank() == 0) {
      auto nstreams = streams_.size();
      std::cout << "created " << nstreams
                << " " TILEDARRAY_DEVICE_RUNTIME_STR " stream"
                << (nstreams == 1 ? "" : "s") << std::endl;
    }

    // lastly, set default device for current MPI process's (main) thread
    DeviceSafeCall(setDevice(compute_devices_.front()));
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
  // N.B. not thread safe, so must be wrapped into umpire_based_allocator_impl
  umpire::Allocator pinned_allocator_;

  int num_devices_visible_;  // total number of devices visible to this rank
  std::vector<int>
      compute_devices_;  // list of devices assigned to this rank,
                         // compute_devices_.size()<=num_devices_visible_
  bool device_concurrent_managed_access_ = true;

  int num_streams_per_device_;
  std::vector<Stream> streams_;  // streams_.size() == (num_streams_per_device_)
                                 // * compute_devices_.size()

  inline static std::unique_ptr<Env>& instance_accessor() {
    static std::unique_ptr<Env> instance_{nullptr};
    return instance_;
  }
};

namespace detail {
// in a madness device task point to its local optional stream to use by
// madness_task_stream_opt; set to nullptr after task callable finished
inline std::optional<Stream>*& madness_task_stream_opt_ptr_accessor() {
  static thread_local std::optional<Stream>* stream_opt_ptr = nullptr;
  return stream_opt_ptr;
}

inline std::optional<Stream>& tls_stream_opt_accessor() {
  static thread_local std::optional<Stream> stream_opt =
      device::Env::instance()->stream(0);
  return stream_opt;
}

inline std::optional<Stream>& madness_task_stream_opt_accessor() {
  if (madness_task_stream_opt_ptr_accessor() != nullptr)  // in a device task?
    return *madness_task_stream_opt_ptr_accessor();
  else
    return tls_stream_opt_accessor();
}
}  // namespace detail

/// must call this before exiting the device task submitted to
/// the MADNESS runtime via madness::add_device_task
/// to synchronize with \p s
/// before task completion
/// \param s the stream to synchronize this task with
inline void sync_madness_task_with(const Stream& s) {
  if (!detail::madness_task_stream_opt_accessor())
    detail::madness_task_stream_opt_accessor() = s;
  else {
    TA_ASSERT(*detail::madness_task_stream_opt_accessor() == s);
  }
}

/// must call this before exiting the device task submitted to
/// the MADNESS runtime via madness::add_device_task
/// to synchronize with \p stream associated with device \p device
/// on the *current* device before task completion
/// \param device the device associated with \p stream
/// \param stream the stream to synchronize this task with
inline void sync_madness_task_with(int device, stream_t stream) {
  sync_madness_task_with(Stream{device, stream});
}

/// must call this before exiting the device task submitted to
/// the MADNESS runtime via madness::add_device_task
/// to synchronize with \p stream on the *current* device
/// before task completion
/// \param stream the stream to synchronize this task with
inline void sync_madness_task_with(stream_t stream) {
  TA_ASSERT(stream != nullptr);
  int current_device = -1;
  DeviceSafeCall(getDevice(&current_device));
  sync_madness_task_with(current_device, stream);
}

/// @return the optional Stream with which this task will be synced
inline std::optional<Stream> madness_task_current_stream() {
  return detail::madness_task_stream_opt_accessor();
}

/// should call this within a task submitted to
/// the MADNESS runtime via madness::add_device_task
/// to cancel the previous calls to sync_madness_task_with()
/// if, e.g., it synchronized with any work performed
/// before exiting
inline void cancel_madness_task_sync() {
  detail::madness_task_stream_opt_accessor() = {};
}

/// maps a (tile) Range to device::Stream; if had already pushed work into a
/// device::Stream (as indicated by madness_task_current_stream() )
/// will return that Stream instead
/// @param[in] range will determine the device::Stream to compute an object
/// associated with this Range object
/// @return the device::Stream to use for creating tasks generating work
/// associated with Range \p range
template <typename Range>
device::Stream stream_for(const Range& range) {
  const auto stream_opt = madness_task_current_stream();
  if (!stream_opt) {
    auto stream_ord =
        range.offset() % device::Env::instance()->num_streams_total();
    return device::Env::instance()->stream(stream_ord);
  } else
    return *stream_opt;
}

}  // namespace device

#endif  // TILEDARRAY_HAS_DEVICE

#ifdef TILEDARRAY_HAS_CUDA
namespace nvidia {

// Color definitions for nvtxcalls
enum class argbColor : uint32_t {
  red = 0xFFFF0000,
  blue = 0xFF0000FF,
  green = 0xFF008000,
  yellow = 0xFFFFFF00,
  cyan = 0xFF00FFFF,
  magenta = 0xFFFF00FF,
  gray = 0xFF808080,
  purple = 0xFF800080
};

/// enter a profiling range by calling nvtxRangePushEx
/// \param[in] range_title a char string containing the range title
/// \param[in] range_color the range color
inline void range_push(const char* range_title, argbColor range_color) {
  nvtxEventAttributes_t eventAttrib = {0};
  eventAttrib.version = NVTX_VERSION;
  eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
  eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
  eventAttrib.colorType = NVTX_COLOR_ARGB;
  eventAttrib.color = static_cast<uint32_t>(range_color);
  eventAttrib.message.ascii = range_title;
  nvtxRangePushEx(&eventAttrib);
}

/// exits the current profiling range by calling nvtxRangePopEx
inline void range_pop() { nvtxRangePop(); }

}  // namespace nvidia
#endif  // TILEDARRAY_HAS_CUDA

}  // namespace TiledArray

#endif  // TILEDARRAY_EXTERNAL_DEVICE_H__INCLUDED
