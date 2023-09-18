//
// Created by Chong Peng on 7/24/18.
//

// clang-format off
#include <btas/array_adaptor.h>  // provides c++17 features (stds::data, std::size) when compiling CUDA (i.e. c++14)
#include <TiledArray/device/btas_um_tensor.h>
// clang-format on

#ifdef TILEDARRAY_HAS_CUDA

template class btas::varray<double, TiledArray::device_um_allocator<double>>;
template class btas::varray<float, TiledArray::device_um_allocator<float>>;
template class btas::varray<
    std::complex<double>,
    TiledArray::device_um_allocator<std::complex<double>>>;
template class btas::varray<
    std::complex<float>, TiledArray::device_um_allocator<std::complex<float>>>;
template class btas::varray<int, TiledArray::device_um_allocator<int>>;
template class btas::varray<long, TiledArray::device_um_allocator<long>>;

template class btas::Tensor<double, TiledArray::Range,
                            TiledArray::device_um_btas_varray<double>>;
template class btas::Tensor<float, TiledArray::Range,
                            TiledArray::device_um_btas_varray<float>>;
template class btas::Tensor<
    std::complex<double>, TiledArray::Range,
    TiledArray::device_um_btas_varray<std::complex<double>>>;
template class btas::Tensor<
    std::complex<float>, TiledArray::Range,
    TiledArray::device_um_btas_varray<std::complex<float>>>;
template class btas::Tensor<int, TiledArray::Range,
                            TiledArray::device_um_btas_varray<int>>;
template class btas::Tensor<long, TiledArray::Range,
                            TiledArray::device_um_btas_varray<long>>;

template class TiledArray::Tile<btas::Tensor<
    double, TiledArray::Range, TiledArray::device_um_btas_varray<double>>>;
template class TiledArray::Tile<btas::Tensor<
    float, TiledArray::Range, TiledArray::device_um_btas_varray<float>>>;
template class TiledArray::Tile<
    btas::Tensor<std::complex<double>, TiledArray::Range,
                 TiledArray::device_um_btas_varray<std::complex<double>>>>;
template class TiledArray::Tile<
    btas::Tensor<std::complex<float>, TiledArray::Range,
                 TiledArray::device_um_btas_varray<std::complex<float>>>>;
template class TiledArray::Tile<btas::Tensor<
    int, TiledArray::Range, TiledArray::device_um_btas_varray<int>>>;
template class TiledArray::Tile<btas::Tensor<
    long, TiledArray::Range, TiledArray::device_um_btas_varray<long>>>;

#endif  // TILEDARRAY_HAS_CUDA
