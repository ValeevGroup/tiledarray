//
// Created by Chong Peng on 7/24/18.
//

// clang-format off
#include <btas/array_adaptor.h>  // provides c++17 features (stds::data, std::size) when compiling CUDA (i.e. c++14)
#include <TiledArray/gpu/btas_um_tensor.h>
// clang-format on

#ifdef TILEDARRAY_HAS_CUDA

template class btas::varray<double, TiledArray::cuda_um_allocator<double>>;
template class btas::varray<float, TiledArray::cuda_um_allocator<float>>;
template class btas::varray<int, TiledArray::cuda_um_allocator<int>>;
template class btas::varray<long, TiledArray::cuda_um_allocator<long>>;

template class btas::Tensor<double, TiledArray::Range,
                            TiledArray::cuda_um_btas_varray<double>>;
template class btas::Tensor<float, TiledArray::Range,
                            TiledArray::cuda_um_btas_varray<float>>;
template class btas::Tensor<int, TiledArray::Range,
                            TiledArray::cuda_um_btas_varray<int>>;
template class btas::Tensor<long, TiledArray::Range,
                            TiledArray::cuda_um_btas_varray<long>>;

template class TiledArray::Tile<btas::Tensor<
    double, TiledArray::Range, TiledArray::cuda_um_btas_varray<double>>>;
template class TiledArray::Tile<btas::Tensor<
    float, TiledArray::Range, TiledArray::cuda_um_btas_varray<float>>>;
template class TiledArray::Tile<
    btas::Tensor<int, TiledArray::Range, TiledArray::cuda_um_btas_varray<int>>>;
template class TiledArray::Tile<btas::Tensor<
    long, TiledArray::Range, TiledArray::cuda_um_btas_varray<long>>>;

#endif  // TILEDARRAY_HAS_CUDA
