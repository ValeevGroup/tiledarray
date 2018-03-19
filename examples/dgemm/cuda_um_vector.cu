
#include "cuda_um_allocator.h"
#include "thrust.h"

namespace thrust {
template<>
void resize<double,TiledArray::cuda_um_allocator<double>>(
    thrust::device_vector<double, TiledArray::cuda_um_allocator<double>>& dev_vec,
    size_t size) {
    dev_vec.resize(size);
}
template<>
void resize<float,TiledArray::cuda_um_allocator<float>>(
    thrust::device_vector<float, TiledArray::cuda_um_allocator<float>>& dev_vec,
    size_t size) {
    dev_vec.resize(size);
}
}

namespace thrust {
template class device_vector<double, TiledArray::cuda_um_allocator<double>>;
template class device_vector<float, TiledArray::cuda_um_allocator<float>>;
}
