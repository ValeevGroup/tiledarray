//
// Created by Eduard Valeyev on 2/6/18.
//

#ifndef TILEDARRAY_CUDA_UM_VECTOR_H
#define TILEDARRAY_CUDA_UM_VECTOR_H

#include "cuda_um_allocator.h"

namespace TiledArray {

/// \brief a vector that lives in CUDA Unified Memory, with most operations implemented on the CPU
template<typename T> using cuda_um_vector = std::vector<T, TiledArray::cuda_um_allocator<T>>;

/// @return true if @c dev_vec is present in space @space
template <MemorySpace Space, typename T>
bool in_memory_space(const cuda_um_vector<T> &vec) noexcept {
  return overlap(MemorySpace::CUDA_UM, Space);
}

}  // namespace TiledArray

#endif //TILEDARRAY_CUDA_UM_VECTOR_H
