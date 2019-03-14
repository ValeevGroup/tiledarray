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

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_CUDA

#include <cuda.h>
#include <cuda_runtime.h>
#include <thrust/system_error.h>
#include <thrust/system/cuda/error.h>

// for memory management
#include <umpire/Umpire.hpp>
#include <umpire/strategy/DynamicPool.hpp>
#include <umpire/strategy/ThreadSafeAllocator.hpp>

#include <madness/tensor/cblas.h>
#include <madness/world/thread.h>
#include <mpi.h>

#include <TiledArray/error.h>
#include <TiledArray/math/cublas.h>

#include <cassert>
#include <vector>


#define CudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define CudaCheckError()    __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall( cudaError err, const char *file, const int line )
{
#ifdef TILEDARRAY_CHECK_CUDA_ERROR
  if ( cudaSuccess != err )
  {
    std::stringstream ss;
    ss << "cudaSafeCall() failed at: " << file << "(" << line << ")";
    std::string what = ss.str();
    throw thrust::system_error(err, thrust::cuda_category(), what);
  }
#endif

  return;
}

inline void __cudaCheckError( const char *file, const int line )
{
#ifdef TILEDARRAY_CHECK_CUDA_ERROR
  cudaError err = cudaGetLastError();
  if ( cudaSuccess != err )
  {
    std::stringstream ss;
    ss << "cudaCheckError() failed at: " << file << "(" << line << ")";
    std::string what = ss.str();
    throw thrust::system_error(err, thrust::cuda_category(), what);
  }
#endif

  return;
}

namespace TiledArray {

namespace detail {

inline std::pair<int, int> mpi_local_rank_size(int mpi_rank) {
  MPI_Comm local_comm;
  MPI_Info info;
  MPI_Info_create(&info);
  MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, mpi_rank, info,
                      &local_comm);

  int local_rank = -1;
  int local_size = -1;
  MPI_Comm_rank(local_comm, &local_rank);
  MPI_Comm_size(local_comm, &local_size);

  MPI_Comm_free(&local_comm);
  MPI_Info_free(&info);

  return std::make_pair(local_rank, local_size);
}

inline int num_cuda_streams() {
  int num_streams = -1;
  char* num_stream_char = std::getenv("TA_NUM_STREAMS");
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

inline int current_cuda_device_id() {
  int mpi_rank = -1;
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);

  int mpi_local_size = -1;
  int mpi_local_rank = -1;

  std::tie(mpi_local_rank, mpi_local_size) = mpi_local_rank_size(mpi_rank);

  int num_devices = detail::num_cuda_devices();

  if (mpi_local_size > num_devices) {
    throw std::runtime_error(
        "TiledArray only support 1 MPI Process per CUDA Device Model "
        "Currently!\n");
  }

  int cuda_device_id = mpi_local_rank % num_devices;

  return cuda_device_id;
}



inline void CUDART_CB cuda_readyflag_callback(cudaStream_t stream,
                                              cudaError_t status,
                                              void *userData) {
  // convert void * to std::atomic<bool>
  std::atomic<bool> *flag = static_cast<std::atomic<bool> *>(userData);
  // set the flag to be true
  flag->store(true);
}

struct ProbeFlag {
  ProbeFlag(std::atomic<bool> *f) : flag(f) {}

  bool operator()() const { return flag->load(); }

  std::atomic<bool> *flag;
};

inline void thread_wait_cuda_stream(const cudaStream_t &stream) {
  std::atomic<bool> *flag = new std::atomic<bool>(false);

  CudaSafeCall(cudaStreamAddCallback(stream, detail::cuda_readyflag_callback, flag, 0));

  detail::ProbeFlag probe(flag);

  // wait with sleep and do not do work
  madness::ThreadPool::await(probe, false, true);
  //    madness::ThreadPool::await(probe, true, true);

  delete flag;
}

}  // namespace detail



inline const cudaStream_t* & tls_cudastream_accessor() {
  static thread_local const cudaStream_t* thread_local_stream_ptr {nullptr};
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
      CudaSafeCall(cudaStreamDestroy(stream));
    }
  }

  /// no copy constructor
  cudaEnv(cudaEnv& cuda_global) = delete;

  /// no assignment constructor
  cudaEnv operator=(cudaEnv& cuda_global) = delete;

  /// access to static member
  static std::unique_ptr<cudaEnv>& instance() {
    static std::unique_ptr<cudaEnv> instance_{nullptr};
    if (!instance_) {
      initialize(instance_);
    }
    return instance_;
  }

  /// initialize static member
  static void initialize(std::unique_ptr<cudaEnv>& instance) {
    // initialize only when not initialized
    if (instance == nullptr) {

      int num_streams = detail::num_cuda_streams();
      int num_devices = detail::num_cuda_devices();
      int device_id = detail::current_cuda_device_id();

//      umpire::util::Logger::getActiveLogger()->setLoggingMsgLevel(
//          umpire::util::message::Debug);
      //       make Thread Safe UM Dynamic POOL

      auto& rm = umpire::ResourceManager::getInstance();

      auto um_dynamic_pool = rm.makeAllocator<umpire::strategy::DynamicPool>(
          "UMDynamicPool", rm.getAllocator("UM"), 2048*1024*1024ul, 10*1024*1024ul);
      auto thread_safe_um_dynamic_pool =
          rm.makeAllocator<umpire::strategy::ThreadSafeAllocator>(
              "ThreadSafeUMDynamicPool", rm.getAllocator("UMDynamicPool"));

      auto cuda_env =
          std::unique_ptr<cudaEnv> (new cudaEnv(num_devices, device_id, num_streams,
                                    rm.getAllocator("ThreadSafeUMDynamicPool")));
      instance = std::move(cuda_env);
    }
  }

  int num_cuda_devices() const { return num_cuda_devices_; }

  int current_cuda_device_id() const { return current_cuda_device_id_; }

  int num_cuda_streams() const { return num_cuda_streams_; }

  bool concurrent_managed_access() const {return cuda_device_concurrent_manged_access_; }

  const cudaStream_t& cuda_stream(std::size_t i) const {
    TA_ASSERT(i < cuda_streams_.size());
    return cuda_streams_[i];
  }

  umpire::Allocator& um_dynamic_pool() { return um_dynamic_pool_; }

 protected:

  cudaEnv(int num_devices, int device_id, int num_streams,
          umpire::Allocator alloc)
          : um_dynamic_pool_(alloc),
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
    if(!prop.managedMemory){
      throw std::runtime_error("CUDA Device doesn't support managedMemory\n");
    }
    int concurrent_managed_access;
    CudaSafeCall(cudaDeviceGetAttribute(&concurrent_managed_access, cudaDevAttrConcurrentManagedAccess, device_id));
    cuda_device_concurrent_manged_access_ = concurrent_managed_access;
    if(!cuda_device_concurrent_manged_access_){
      std::cout << "\nWarning: CUDA Device doesn't support ConcurrentManagedAccess!\n\n";
    }

    // creates cuda streams on current device
    cuda_streams_.resize(num_cuda_streams_);
    for (auto& stream : cuda_streams_) {
      CudaSafeCall(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    }
  }

 private:
  /// a Thread Safe, Dynamic memory pool for Unified Memory
  umpire::Allocator um_dynamic_pool_;

  int num_cuda_devices_;
  int current_cuda_device_id_;
  bool cuda_device_concurrent_manged_access_;

  int num_cuda_streams_;
  std::vector<cudaStream_t> cuda_streams_;
};

/// initialize cuda environment
inline void cuda_initialize() {
  /// initialize cudaGlobal
  cudaEnv::instance();
  //
  cuBLASHandlePool::handle();
}

/// finalize cuda environment
inline void cuda_finalize() {
  CudaSafeCall(cudaDeviceSynchronize());
  cublasDestroy(cuBLASHandlePool::handle());
  delete &cuBLASHandlePool::handle();
  cudaEnv::instance().reset(nullptr);
}

namespace detail {

template <typename Range>
const cudaStream_t &get_stream_based_on_range(const Range &range) {
  // TODO better way to get stream based on the id of tensor
  auto stream_id = range.offset() % cudaEnv::instance()->num_cuda_streams();
  auto &stream = cudaEnv::instance()->cuda_stream(stream_id);
  return stream;
}

} //namespace detail

}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_CUDA

#endif  // TILEDARRAY_EXTERNAL_CUDA_H__INCLUDED
