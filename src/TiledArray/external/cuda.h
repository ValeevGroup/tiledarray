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

// for memory management
#include <umpire/Umpire.hpp>
#include <umpire/strategy/DynamicPool.hpp>
#include <umpire/strategy/ThreadSafeAllocator.hpp>

#include <madness/tensor/cblas.h>
#include <mpi.h>

#include <TiledArray/error.h>
#include <TiledArray/math/cublas.h>

#include <cassert>
#include <vector>

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
  /// default num of streams is 1
  if (num_stream_char) {
    num_streams = std::atoi(num_stream_char);
  } else {
    num_streams = 1;
  }
  return num_streams;
}

inline int num_cuda_devices() {
  int num_devices = -1;
  auto error = cudaGetDeviceCount(&num_devices);
  TA_ASSERT(error == cudaSuccess);
  //  if (error != cudaSuccess) {
  //    std::cout << "error(cudaGetDeviceCount) = " << error << std::endl;
  //  }
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

}  // namespace detail



inline const cudaStream_t* & tls_cudastreamptr_accessor() {
  static thread_local const cudaStream_t* thread_local_stream_ptr {nullptr};
  return thread_local_stream_ptr;
}

inline void synchronize_stream(const cudaStream_t* stream) {
  tls_cudastreamptr_accessor() = stream;
}

/**
 * cudaEnv set up global environment
 *
 * Singleton class
 */

class cudaEnv {
 public:
  cudaEnv(int num_devices, int device_id, int num_streams,
          umpire::Allocator alloc)
      : um_dynamic_pool_(alloc),
        num_cuda_devices_(num_devices),
        current_cuda_device_id_(device_id),
        num_cuda_streams_(num_streams) {
    // set device for current MPI process
    auto error = cudaSetDevice(current_cuda_device_id_);
    TA_ASSERT(error == cudaSuccess);
    //    if (error != cudaSuccess) {
    //      std::cout << "error(cudaSetDevice) = " << error << std::endl;
    //    }

    /// TODO set device for all MAD_THREADS

    // creates cuda streams on current device
    cuda_streams_.resize(num_cuda_streams_);
    for (auto& stream : cuda_streams_) {
      auto error = cudaStreamCreate(&stream);
      TA_ASSERT(error == cudaSuccess);
    }
  }

  ~cudaEnv() {
    // destroy cuda streams on current device
    for (auto& stream : cuda_streams_) {
      cudaStreamDestroy(stream);
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
          std::make_unique<cudaEnv>(num_devices, device_id, num_streams,
                                    rm.getAllocator("ThreadSafeUMDynamicPool"));
      instance = std::move(cuda_env);
    }
  }

  int num_cuda_devices() const { return num_cuda_devices_; }

  int current_cuda_device_id() const { return current_cuda_device_id_; }

  int num_cuda_streams() const { return num_cuda_streams_; }

  const cudaStream_t& cuda_stream(std::size_t i) const {
    TA_ASSERT(i < cuda_streams_.size());
    return cuda_streams_[i];
  }

  umpire::Allocator& um_dynamic_pool() { return um_dynamic_pool_; }

 private:
  /// a Thread Safe, Dynamic memory pool for Unified Memory
  umpire::Allocator um_dynamic_pool_;

  int num_cuda_devices_;
  int current_cuda_device_id_;

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
  cudaDeviceSynchronize();
  cublasDestroy(cuBLASHandlePool::handle());
}

}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_CUDA

#endif  // TILEDARRAY_EXTERNAL_CUDA_H__INCLUDED
