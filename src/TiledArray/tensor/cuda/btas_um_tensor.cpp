//
// Created by Chong Peng on 7/24/18.
//

#include <TiledArray/tensor/cuda/btas_um_tensor.h>


#ifdef TILEDARRAY_HAS_CUDA

std::unique_ptr<TiledArray::cudaEnv> TiledArray::cudaEnv::instance_ = nullptr;
thread_local cublasHandle_t *TiledArray::cuBLASHandlePool::handle_ = nullptr;


//template class btas::Tensor<double,btas::RangeNd<CblasRowMajor, std::array<std::size_t, 2>>,TiledArray::cuda_um_btas_varray<double>>;
//template class btas::Tensor<float,btas::RangeNd<CblasRowMajor, std::array<std::size_t, 2>>,TiledArray::cuda_um_btas_varray<float>>;
//template class btas::Tensor<int,btas::RangeNd<CblasRowMajor, std::array<std::size_t, 2>>,TiledArray::cuda_um_btas_varray<int>>;
//template class btas::Tensor<long,btas::RangeNd<CblasRowMajor, std::array<std::size_t, 2>>,TiledArray::cuda_um_btas_varray<long>>;
//
//
//template class btas::Tensor<double,btas::RangeNd<CblasRowMajor, std::array<std::size_t, 2>>,TiledArray::cuda_um_thrust_vector<double>>;
//template class btas::Tensor<float,btas::RangeNd<CblasRowMajor, std::array<std::size_t, 2>>,TiledArray::cuda_um_thrust_vector<float>>;
//template class btas::Tensor<int,btas::RangeNd<CblasRowMajor, std::array<std::size_t, 2>>,TiledArray::cuda_um_thrust_vector<int>>;
//template class btas::Tensor<long,btas::RangeNd<CblasRowMajor, std::array<std::size_t, 2>>,TiledArray::cuda_um_thrust_vector<long>>;



#endif //TILEDARRAY_HAS_CUDA