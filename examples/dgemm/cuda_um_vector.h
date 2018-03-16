//
// Created by Eduard Valeyev on 2/6/18.
//

#ifndef TILEDARRAY_CUDA_UM_VECTOR_H
#define TILEDARRAY_CUDA_UM_VECTOR_H

#include "cuda_um_allocator.h"
#include "platform.h"
#include "thrust.h"
#include <TiledArray/utility.h>

namespace TiledArray {

/// \brief a vector that lives in CUDA Unified Memory, with most operations implemented on the CPU
template<typename T> using cuda_um_vector = std::vector<T, TiledArray::cuda_um_allocator<T>>;
//template<typename T> using cuda_um_vector = thrust::device_vector<T, TiledArray::cuda_um_allocator<T>>;

/// @return true if @c dev_vec is present in space @space
template <MemorySpace Space, typename T>
bool in_memory_space(const cuda_um_vector<T> &vec) noexcept {
  return overlap(MemorySpace::CUDA_UM, Space);
}

template <ExecutionSpace Space, typename T>
void to_execution_space(cuda_um_vector<T>& vec) {
  switch(Space) {
    case ExecutionSpace::CPU: {
      using detail::size;
      cudaMemPrefetchAsync(data(vec), size(vec) * sizeof(T), cudaCpuDeviceId);
      break;
    }
    case ExecutionSpace::CUDA: {
      using detail::size;
      int device = -1;
      cudaGetDevice(&device);
      cudaMemPrefetchAsync(data(vec), size(vec) * sizeof(T), device);
      break;
    }
    default:
      throw std::runtime_error("invalid execution space");
  }
}

}  // namespace TiledArray

namespace madness {
namespace archive {

// forward decls
template<class Archive, typename T> struct ArchiveLoadImpl;
template<class Archive, typename T> struct ArchiveStoreImpl;

template<class Archive, typename T>
struct ArchiveLoadImpl<Archive, thrust::device_vector<T, TiledArray::cuda_um_allocator<T>> > {
  static inline void load(const Archive& ar, thrust::device_vector<T, TiledArray::cuda_um_allocator<T>>& x) {
    typename thrust::device_vector<T, TiledArray::cuda_um_allocator<T>>::size_type n;
    ar & n;
    x.resize(n);
    for (auto& xi : x)
      ar & xi;
  }
};

template<class Archive, typename T>
struct ArchiveStoreImpl<Archive, thrust::device_vector<T, TiledArray::cuda_um_allocator<T>> > {
  static inline void store(const Archive& ar, const thrust::device_vector<T, TiledArray::cuda_um_allocator<T>>& x) {
    ar & x.size();
    for (const auto& xi : x)
      ar & xi;
  }
};

}
}

#endif //TILEDARRAY_CUDA_UM_VECTOR_H
