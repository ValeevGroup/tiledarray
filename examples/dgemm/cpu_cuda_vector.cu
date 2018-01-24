
#include "cpu_cuda_vector.h"

template<>
void resize<double,thrust::device_malloc_allocator<double>>(
    thrust::device_vector<double, thrust::device_malloc_allocator<double>>& dev_vec,
    size_t size) {
    dev_vec.resize(size);
}
template<>
void resize<float,thrust::device_malloc_allocator<float>>(
    thrust::device_vector<float, thrust::device_malloc_allocator<float>>& dev_vec,
    size_t size) {
    dev_vec.resize(size);
}

template class cpu_cuda_vector<double>;
template class cpu_cuda_vector<float>;

