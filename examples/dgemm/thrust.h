//
// Created by Eduard Valeyev on 3/16/18.
//

#ifndef TILEDARRAY_THRUST_H
#define TILEDARRAY_THRUST_H

#include <cublas_v2.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

// thrust::device_vector::data() returns a proxy, provide an overload for std::data() to provide raw ptr
namespace thrust {
template<typename T, typename Alloc>
const T* data (const thrust::device_vector<T, Alloc>& dev_vec) {
  return thrust::raw_pointer_cast(dev_vec.data());
}
template<typename T, typename Alloc>
T* data (thrust::device_vector<T, Alloc>& dev_vec) {
  return thrust::raw_pointer_cast(dev_vec.data());
}

// this must be instantiated in a .cu file
template <typename T, typename Alloc>
void resize(thrust::device_vector<T, Alloc>& dev_vec, size_t size);
}  // namespace thrust

#endif //TILEDARRAY_THRUST_H
