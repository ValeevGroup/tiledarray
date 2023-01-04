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

#ifndef TILEDARRAY_EXTERNAL_CUDA_H__INCLUDED
#define TILEDARRAY_EXTERNAL_CUDA_H__INCLUDED

#include <cassert>
#include <cstdlib>
#include <vector>

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvToolsExt.h>
#include <thrust/system/cuda/error.h>
#include <thrust/system_error.h>

#include <TiledArray/external/umpire.h>

#include <TiledArray/external/madness.h>
#include <madness/world/print.h>
#include <madness/world/safempi.h>
#include <madness/world/thread.h>

#include <TiledArray/error.h>

#define CudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define CudaSafeCallNoThrow(err) __cudaSafeCallNoThrow(err, __FILE__, __LINE__)
#define CudaCheckError() __cudaCheckError(__FILE__, __LINE__)

inline void __cudaSafeCall(cudaError err, const char* file, const int line) {
#ifdef TILEDARRAY_CHECK_CUDA_ERROR
  if (cudaSuccess != err) {
    std::stringstream ss;
    ss << "cudaSafeCall() failed at: " << file << ":" << line;
    std::string what = ss.str();
    throw thrust::system_error(err, thrust::cuda_category(), what);
  }
#endif
}

inline void __cudaSafeCallNoThrow(cudaError err, const char* file,
                                  const int line) {
#ifdef TILEDARRAY_CHECK_CUDA_ERROR
  if (cudaSuccess != err) {
    madness::print_error("cudaSafeCallNoThrow() failed at: ", file, ":", line);
  }
#endif
}

inline void __cudaCheckError(const char* file, const int line) {
#ifdef TILEDARRAY_CHECK_CUDA_ERROR
  cudaError err = cudaGetLastError();
  if (cudaSuccess != err) {
    std::stringstream ss;
    ss << "cudaCheckError() failed at: " << file << ":" << line;
    std::string what = ss.str();
    throw thrust::system_error(err, thrust::cuda_category(), what);
  }
#endif
}

namespace TiledArray {

namespace detail {

inline std::pair<int, int> mpi_local_rank_size(World& world) {
  auto host_comm =
      world.mpi.comm().Split_type(SafeMPI::Intracomm::SHARED_SPLIT_TYPE, 0);
  return std::make_pair(host_comm.Get_rank(), host_comm.Get_size());
}

inline int num_cuda_streams() {
  int num_streams = -1;
  char* num_stream_char = std::getenv("TA_CUDA_NUM_STREAMS");
  /// default num of streams is 3
  if (num_stream_char) {
    num_streams = std::atoi(num_stream_char);
  } else {
    num_streams = 3;
  }
  return num_streams;
}

inline int num_cuda_devices() {
  int num_devices = -1;
  CudaSafeCall(cudaGetDeviceCount(&num_devices));
  return num_devices;
}

inline int current_cuda_device_id(World& world) {
  int mpi_local_size = -1;
  int mpi_local_rank = -1;
  std::tie(mpi_local_rank, mpi_local_size) = mpi_local_rank_size(world);

  int num_devices = detail::num_cuda_devices();

  int cuda_device_id = -1;
  // devices may already be pre-mapped
  // if mpi_local_size <= num_devices : all ranks are in same resource set, map
  // round robin
  if (mpi_local_size <= num_devices) {
    cuda_device_id = mpi_local_rank % num_devices;
  } else {  // mpi_local_size > num_devices
    char* cvd_cstr = std::getenv("CUDA_VISIBLE_DEVICES");
    if (cvd_cstr) {  // CUDA_VISIBLE_DEVICES is set, assume that pre-mapped
      // make sure that there is only 1 device available here
      if (num_devices != 1) {
        throw std::runtime_error(
            std::string(
                "CUDA_VISIBLE_DEVICES environment variable is set, hence using "
                "the provided device-to-rank mapping; BUT TiledArray found ") +
            std::to_string(num_devices) +
            " CUDA devices; only 1 CUDA device / MPI process is supported");
      }
      cuda_device_id = 0;
    } else {  // not enough devices + devices are not pre-mapped
      throw std::runtime_error(
          std::string("TiledArray found ") + std::to_string(mpi_local_size) +
          " MPI ranks on a node with " + std::to_string(num_devices) +
          " CUDA devices; only 1 MPI process / CUDA device model is currently "
          "supported");
    }
  }

  return cuda_device_id;
}

inline void CUDART_CB cuda_readyflag_callback(void* userData) {
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

inline void thread_wait_cuda_stream(const cudaStream_t& stream) {
  std::atomic<bool>* flag = new std::atomic<bool>(false);

  CudaSafeCall(
      cudaLaunchHostFunc(stream, detail::cuda_readyflag_callback, flag));

  detail::ProbeFlag probe(flag);

  // wait with sleep and do not do work
  madness::ThreadPool::await(probe, false, true);
  //    madness::ThreadPool::await(probe, true, true);

  delete flag;
}

}  // namespace detail

inline const cudaStream_t*& tls_cudastream_accessor() {
  static thread_local const cudaStream_t* thread_local_stream_ptr{nullptr};
  return thread_local_stream_ptr;
}

inline void synchronize_stream(const cudaStream_t* stream) {
  tls_cudastream_accessor() = stream;
}

/**
 * cudaEnv set up global environment
 *
 * Singleton class
 */

class cudaEnv {
 public:
  ~cudaEnv() {
    // destroy cuda streams on current device
    for (auto& stream : cuda_streams_) {
      CudaSafeCallNoThrow(cudaStreamDestroy(stream));
    }
  }

  cudaEnv(const cudaEnv&) = delete;
  cudaEnv(cudaEnv&&) = delete;
  cudaEnv& operator=(const cudaEnv&) = delete;
  cudaEnv& operator=(cudaEnv&&) = delete;

  /// access to static member
  static std::unique_ptr<cudaEnv>& instance() {
    static std::unique_ptr<cudaEnv> instance_{nullptr};
    if (!instance_) {
      initialize(instance_, TiledArray::get_default_world());
    }
    return instance_;
  }

  /// initialize static member
  static void initialize(std::unique_ptr<cudaEnv>& instance, World& world) {
    // initialize only when not initialized
    if (instance == nullptr) {
      int num_streams = detail::num_cuda_streams();
      int num_devices = detail::num_cuda_devices();
      int device_id = detail::current_cuda_device_id(world);
      // set device for current MPI process .. will be set in the ctor as well
      CudaSafeCall(cudaSetDevice(device_id));
      CudaSafeCall(cudaDeviceSetCacheConfig(cudaFuncCachePreferShared));

      // uncomment to debug umpire ops
      //
      //      umpire::util::Logger::getActiveLogger()->setLoggingMsgLevel(
      //          umpire::util::message::Debug);

      //       make Thread Safe UM Dynamic POOL

      auto& rm = umpire::ResourceManager::getInstance();

      auto mem_total_free = cudaEnv::memory_total_and_free_device();

      // turn off Umpire introspection for non-Debug builds
#ifndef NDEBUG
      constexpr auto introspect = true;
#else
      constexpr auto introspect = false;
#endif

      // allocate all free memory for UM pool
      // subsequent allocs will use 1/10 of the total device memory
      auto alloc_grain = mem_total_free.second / 10;
      auto um_dynamic_pool =
          rm.makeAllocator<umpire::strategy::QuickPool, introspect>(
              "UMDynamicPool", rm.getAllocator("UM"), mem_total_free.second,
              alloc_grain);

      // allocate zero memory for device pool, same grain for subsequent allocs
      auto dev_size_limited_alloc =
          rm.makeAllocator<umpire::strategy::SizeLimiter, introspect>(
              "size_limited_alloc", rm.getAllocator("DEVICE"),
              mem_total_free.first);
      auto dev_dynamic_pool =
          rm.makeAllocator<umpire::strategy::QuickPool, introspect>(
              "CUDADynamicPool", dev_size_limited_alloc, 0, alloc_grain);

      auto cuda_env = std::unique_ptr<cudaEnv>(
          new cudaEnv(world, num_devices, device_id, num_streams,
                      um_dynamic_pool, dev_dynamic_pool));
      instance = std::move(cuda_env);
    }
  }

  World& world() const { return *world_; }

  int num_cuda_devices() const { return num_cuda_devices_; }

  int current_cuda_device_id() const { return current_cuda_device_id_; }

  int num_cuda_streams() const { return num_cuda_streams_; }

  bool concurrent_managed_access() const {
    return cuda_device_concurrent_managed_access_;
  }

  size_t stream_id(const cudaStream_t& stream) const {
    auto it = std::find(cuda_streams_.begin(), cuda_streams_.end(), stream);
    if (it == cuda_streams_.end()) abort();
    return it - cuda_streams_.begin();
  }

  /// @return the total size of all and free device memory on the current device
  static std::pair<size_t, size_t> memory_total_and_free_device() {
    std::pair<size_t, size_t> result;
    // N.B. cudaMemGetInfo returns {free,total}
    CudaSafeCall(cudaMemGetInfo(&result.second, &result.first));
    return result;
  }

  /// Collective call to probe CUDA {total,free} memory

  /// @return the total size of all and free device memory on every rank's
  /// device
  std::vector<std::pair<size_t, size_t>> memory_total_and_free() const {
    auto world_size = world_->size();
    std::vector<size_t> total_memory(world_size, 0), free_memory(world_size, 0);
    auto rank = world_->rank();
    std::tie(total_memory.at(rank), free_memory.at(rank)) =
        cudaEnv::memory_total_and_free_device();
    world_->gop.sum(total_memory.data(), total_memory.size());
    world_->gop.sum(free_memory.data(), free_memory.size());
    std::vector<std::pair<size_t, size_t>> result(world_size);
    for (int r = 0; r != world_size; ++r) {
      result.at(r) = {total_memory.at(r), free_memory.at(r)};
    }
    return result;
  }

  const cudaStream_t& cuda_stream(std::size_t i) const {
    return cuda_streams_.at(i);
  }

  const cudaStream_t& cuda_stream_h2d() const {
    return cuda_streams_[num_cuda_streams_];
  }

  const cudaStream_t& cuda_stream_d2h() const {
    return cuda_streams_[num_cuda_streams_ + 1];
  }

  /// @return a (non-thread-safe) Umpire allocator for CUDA UM
  umpire::Allocator& um_dynamic_pool() { return um_dynamic_pool_; }

  /// @return a (non-thread-safe) Umpire allocator for CUDA device memory
  umpire::Allocator& device_dynamic_pool() { return device_dynamic_pool_; }

 protected:
  cudaEnv(World& world, int num_devices, int device_id, int num_streams,
          umpire::Allocator um_alloc, umpire::Allocator device_alloc)
      : world_(&world),
        um_dynamic_pool_(um_alloc),
        device_dynamic_pool_(device_alloc),
        num_cuda_devices_(num_devices),
        current_cuda_device_id_(device_id),
        num_cuda_streams_(num_streams) {
    if (num_devices <= 0) {
      throw std::runtime_error("No CUDA-Enabled GPUs Found!\n");
    }

    // set device for current MPI process
    CudaSafeCall(cudaSetDevice(current_cuda_device_id_));

    /// check the capability of CUDA device
    cudaDeviceProp prop;
    CudaSafeCall(cudaGetDeviceProperties(&prop, device_id));
    if (!prop.managedMemory) {
      throw std::runtime_error("CUDA Device doesn't support managedMemory\n");
    }
    int concurrent_managed_access;
    CudaSafeCall(cudaDeviceGetAttribute(&concurrent_managed_access,
                                        cudaDevAttrConcurrentManagedAccess,
                                        device_id));
    cuda_device_concurrent_managed_access_ = concurrent_managed_access;
    if (!cuda_device_concurrent_managed_access_) {
      std::cout << "\nWarning: CUDA Device doesn't support "
                   "ConcurrentManagedAccess!\n\n";
    }

    // creates cuda streams on current device
    cuda_streams_.resize(num_cuda_streams_ + 2);
    for (auto& stream : cuda_streams_) {
      CudaSafeCall(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
    std::cout << "created " << num_cuda_streams_
              << " CUDA streams + 2 I/O streams" << std::endl;
  }

 private:
  // the world used to initialize this
  World* world_;

  /// a thread-safe dynamically-sized memory pool for CUDA Unified Memory
  umpire::Allocator um_dynamic_pool_;
  /// a thread-safe size-limited dynamically-sized memory pool for CUDA Memory
  umpire::Allocator device_dynamic_pool_;

  int num_cuda_devices_;
  int current_cuda_device_id_;
  bool cuda_device_concurrent_managed_access_;

  int num_cuda_streams_;
  std::vector<cudaStream_t> cuda_streams_;
};

namespace detail {

template <typename Range>
const cudaStream_t& get_stream_based_on_range(const Range& range) {
  // TODO better way to get stream based on the id of tensor
  auto stream_id = range.offset() % cudaEnv::instance()->num_cuda_streams();
  auto& stream = cudaEnv::instance()->cuda_stream(stream_id);
  return stream;
}

}  // namespace detail

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

}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_CUDA

#endif  // TILEDARRAY_EXTERNAL_CUDA_H__INCLUDED
