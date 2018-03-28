//
// Created by Eduard Valeyev on 2/6/18.
//

#ifndef TILEDARRAY_CUDA_UM_VECTOR_H
#define TILEDARRAY_CUDA_UM_VECTOR_H

#include <TiledArray/utility.h>
#include "btas/array_adaptor.h"
#include "btas/varray/varray.h"
#include "cuda_um_allocator.h"
#include "platform.h"
#include "thrust.h"

namespace TiledArray {

/// \brief a vector that lives in CUDA Unified Memory, with most operations
/// implemented on the CPU
template <typename T>
using cuda_um_vector = std::vector<T, TiledArray::cuda_um_allocator<T>>;
template <typename T>
using cuda_um_btas_varray = btas::varray<T, TiledArray::cuda_um_allocator<T>>;
template <typename T>
using cuda_um_thrust_vector =
    thrust::device_vector<T, TiledArray::cuda_um_allocator<T>>;

/// @return true if @c dev_vec is present in space @space
template <MemorySpace Space, typename Storage>
bool in_memory_space(const Storage& vec) noexcept {
  return overlap(MemorySpace::CUDA_UM, Space);
}
/**
 * @tparam Space
 * @tparam Storage  the Storage type of the vector, such as cuda_um_vector,
 * cuda_um_btas_varray
 */
template <ExecutionSpace Space, typename Storage>
void to_execution_space(Storage& vec, cudaStream_t stream=0 ) {
  switch (Space) {
    case ExecutionSpace::CPU: {
      using detail::size;
      using std::data;
      using value_type = typename Storage::value_type;
      cudaMemPrefetchAsync(data(vec), size(vec) * sizeof(value_type),
                           cudaCpuDeviceId, stream);
      break;
    }
    case ExecutionSpace::CUDA: {
      using detail::size;
      using std::data;
      using value_type = typename Storage::value_type;
      int device = -1;
      cudaGetDevice(&device);
      cudaMemPrefetchAsync(data(vec), size(vec) * sizeof(value_type), device, stream);
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
template <class Archive, typename T>
struct ArchiveLoadImpl;
template <class Archive, typename T>
struct ArchiveStoreImpl;

template <class Archive, typename T>
struct ArchiveLoadImpl<Archive, TiledArray::cuda_um_thrust_vector<T>> {
  static inline void load(const Archive& ar, TiledArray::cuda_um_thrust_vector<T>& x) {
    typename thrust::device_vector<
        T, TiledArray::cuda_um_allocator<T>>::size_type n(0);
    ar& n;
    x.resize(n);
    for (auto& xi : x) ar& xi;
  }
};

template <class Archive, typename T>
struct ArchiveStoreImpl<Archive, TiledArray::cuda_um_thrust_vector<T>> {
  static inline void store(const Archive& ar,
                           const TiledArray::cuda_um_thrust_vector<T>& x) {
    ar& x.size();
    for (const auto& xi : x) ar& xi;
  }
};

template <class Archive, typename T>
struct ArchiveLoadImpl<Archive, TiledArray::cuda_um_btas_varray<T>> {
  static inline void load(const Archive& ar, TiledArray::cuda_um_btas_varray<T> &x) {
    typename TiledArray::cuda_um_btas_varray<T>::size_type n(0);
    ar& n;
    x.resize(n);
    for (auto& xi : x) ar& xi;
  }
};

template <class Archive, typename T>
struct ArchiveStoreImpl<Archive, TiledArray::cuda_um_btas_varray<T>> {
  static inline void store(const Archive& ar,
                           const TiledArray::cuda_um_btas_varray<T>& x) {
    ar& x.size();
    for (const auto& xi : x) ar& xi;
  }
};

}  // namespace archive
}  // namespace madness

#endif  // TILEDARRAY_CUDA_UM_VECTOR_H
