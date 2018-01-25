
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


void force_missing_instantiations_double() {

using Real = double;
thrust::cuda_cub::execution_policy<thrust::cuda_cub::tag> policy;
thrust::detail::normal_iterator<thrust::device_ptr<Real const> > it;
thrust::device_ptr<Real> ptr;

auto x = thrust::cuda_cub::copy<thrust::cuda_cub::tag, thrust::detail::normal_iterator<thrust::device_ptr<Real const> >, thrust::device_ptr<Real> >(policy, it, it, ptr);

}

void force_missing_instantiations_float() {

using Real = float;
thrust::cuda_cub::execution_policy<thrust::cuda_cub::tag> policy;
thrust::detail::normal_iterator<thrust::device_ptr<Real const> > it;
thrust::device_ptr<Real> ptr;

auto x = thrust::cuda_cub::copy<thrust::cuda_cub::tag, thrust::detail::normal_iterator<thrust::device_ptr<Real const> >, thrust::device_ptr<Real> >(policy, it, it, ptr);

}

