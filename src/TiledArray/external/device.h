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

inline int num_streams() {
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

inline int num_devices() {
  int num_devices = -1;
#if defined(TILEDARRAY_HAS_CUDA)
  DeviceSafeCall(cudaGetDeviceCount(&num_devices));
#elif defined(TILEDARRAY_HAS_HIP)
  DeviceSafeCall(hipGetDeviceCount(&num_devices));
#endif
  return num_devices;
}

inline int current_device_id(World& world) {
  int mpi_local_size = -1;
  int mpi_local_rank = -1;
  std::tie(mpi_local_rank, mpi_local_size) = detail::mpi_local_rank_size(world);

  int num_devices = device::num_devices();

  int device_id = -1;
  // devices may already be pre-mapped
  // if mpi_local_size <= num_devices : all ranks are in same resource set, map
  // round robin
  if (mpi_local_size <= num_devices) {
    device_id = mpi_local_rank % num_devices;
  } else {  // mpi_local_size > num_devices
    const char* vd_cstr =
        std::getenv(TILEDARRAY_DEVICE_RUNTIME_STR "_VISIBLE_DEVICES");
    if (vd_cstr) {  // *_VISIBLE_DEVICES is set, assume that pre-mapped
      // make sure that there is only 1 device available here
      if (num_devices != 1) {
        throw std::runtime_error(
            std::string(
                TILEDARRAY_DEVICE_RUNTIME_STR
                "_VISIBLE_DEVICES environment variable is set, hence using "
                "the provided device-to-rank mapping; BUT TiledArray found ") +
            std::to_string(num_devices) +
            " devices; only 1 device / MPI process is supported");
      }
      device_id = 0;
    } else {  // not enough devices + devices are not pre-mapped
      throw std::runtime_error(
          std::string("TiledArray found ") + std::to_string(mpi_local_size) +
          " MPI ranks on a node with " + std::to_string(num_devices) +
          " devices; only 1 MPI process / device model is currently "
          "supported");
    }
  }

  return device_id;
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

inline const stream_t*& tls_stream_accessor() {
  static thread_local const stream_t* thread_local_stream_ptr{nullptr};
  return thread_local_stream_ptr;
}

/// must call this before exiting the device task executed via
/// the MADNESS runtime (namely, via madness::add_device_task )
/// to inform the runtime which stream the task
/// launched its kernels into
inline void synchronize_stream(const stream_t* stream) {
  tls_stream_accessor() = stream;
}

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
    for (auto& stream : streams_) {
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
      int num_streams = device::num_streams();
      int num_devices = device::num_devices();
      int device_id = device::current_device_id(world);
      // set device for current MPI process .. will be set in the ctor as well
      DeviceSafeCall(setDevice(device_id));
      DeviceSafeCall(deviceSetCacheConfig(FuncCachePreferShared));

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

      auto env = std::unique_ptr<Env>(
          new Env(world, num_devices, device_id, num_streams, um_dynamic_pool,
                  dev_dynamic_pool, pinned_dynamic_pool));
      instance_accessor() = std::move(env);
    }
  }

  World& world() const { return *world_; }

  int num_devices() const { return num_devices_; }

  int current_device_id() const { return current_device_id_; }

  int num_streams() const { return num_streams_; }

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

  const stream_t& stream(std::size_t i) const { return streams_.at(i); }

  const stream_t& stream_h2d() const { return streams_[num_streams_]; }

  const stream_t& stream_d2h() const { return streams_[num_streams_ + 1]; }

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
  Env(World& world, int num_devices, int device_id, int num_streams,
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
      throw std::runtime_error("No " TILEDARRAY_DEVICE_RUNTIME_STR
                               " compute devices found!\n");
    }

    // set device for current MPI process
    DeviceSafeCall(setDevice(current_device_id_));

    /// check the capability of device
    deviceProp_t prop;
    DeviceSafeCall(getDeviceProperties(&prop, device_id));
    if (!prop.managedMemory) {
      throw std::runtime_error(TILEDARRAY_DEVICE_RUNTIME_STR
                               "device doesn't support managedMemory\n");
    }
    int concurrent_managed_access;
    DeviceSafeCall(deviceGetAttribute(&concurrent_managed_access,
                                      DeviceAttributeConcurrentManagedAccess,
                                      device_id));
    device_concurrent_managed_access_ = concurrent_managed_access;
    if (!device_concurrent_managed_access_) {
      std::cout << "\nWarning: " TILEDARRAY_DEVICE_RUNTIME_STR
                   " device doesn't support "
                   "ConcurrentManagedAccess!\n\n";
    }

    // creates streams on current device
    streams_.resize(num_streams_ + 2);
    for (auto& stream : streams_) {
      DeviceSafeCall(streamCreateWithFlags(&stream, StreamNonBlocking));
    }
    std::cout << "created " << num_streams_
              << " " TILEDARRAY_DEVICE_RUNTIME_STR " streams + 2 I/O streams"
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
  // N.B. not thread safe, so must be wrapped into umpire_based_allocator_impl
  umpire::Allocator pinned_allocator_;

  int num_devices_;
  int current_device_id_;
  bool device_concurrent_managed_access_;

  int num_streams_;
  std::vector<stream_t> streams_;

  inline static std::unique_ptr<Env>& instance_accessor() {
    static std::unique_ptr<Env> instance_{nullptr};
    return instance_;
  }
};

}  // namespace device

namespace detail {
template <typename Range>
const device::stream_t& get_stream_based_on_range(const Range& range) {
  // TODO better way to get stream based on the id of tensor
  auto stream_id = range.offset() % device::Env::instance()->num_streams();
  auto& stream = device::Env::instance()->stream(stream_id);
  return stream;
}
}  // namespace detail

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
