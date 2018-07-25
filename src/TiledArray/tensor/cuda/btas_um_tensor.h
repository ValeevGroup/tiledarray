/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2018  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  Chong Peng
 *  Department of Chemistry, Virginia Tech
 *  July 24, 2018
 *
 */

#ifndef TILEDARRAY_CUDA_TENSOR_CUDA_UM_TENSOR_H
#define TILEDARRAY_CUDA_TENSOR_CUDA_UM_TENSOR_H

#include <TiledArray/tensor/cuda/um_storage.h>

#ifdef TILEDARRAY_HAS_CUDA

#include <TiledArray/tensor/cuda/btas_cublas.h>
#include <TiledArray/range.h>

namespace TiledArray{

/*
 * btas::Tensor with UM storage cuda_um_btas_varray
 */

template <typename T, typename Range = btas::RangeNd<CblasRowMajor, std::array<std::size_t, 2>>>
using btasUMTensorVarray = btas::Tensor<T,Range,TiledArray::cuda_um_btas_varray<T>>;

template <typename T, typename Range>
btasUMTensorVarray<T, Range> gemm(
        const btasUMTensorVarray<T, Range> &left,
        const btasUMTensorVarray<T, Range> &right,
        T factor, const TiledArray::math::GemmHelper &gemm_helper) {
  return btas_tensor_gemm_cuda_impl(left, right, factor, gemm_helper);
}

template <typename T, typename Range>
void gemm(
        btasUMTensorVarray<T, Range>  &result,
        const btasUMTensorVarray<T, Range>  &left,
        const btasUMTensorVarray<T, Range>  &right,
        T factor, const TiledArray::math::GemmHelper &gemm_helper) {
  return btas_tensor_gemm_cuda_impl(result, left, right, factor, gemm_helper);
}

template <typename T, typename Range>
void add_to(
        btasUMTensorVarray<T, Range>  &result,
        const btasUMTensorVarray<T, Range> &arg) {
  btas_tensor_add_to_cuda_impl(result, arg);
}

template <typename T, typename Range>
typename btasUMTensorVarray<T, Range> ::value_type
squared_norm(
        const btasUMTensorVarray<T, Range> &arg) {
  return btas_tensor_squared_norm_cuda_impl(arg);
}

/*
 * btas::Tensor with UM storage cuda_um_thrust_vector
 */
template <typename T, typename Range = btas::RangeNd<CblasRowMajor, std::array<std::size_t, 2>>>
using btasUMTensorThrust = btas::Tensor<T,Range,TiledArray::cuda_um_thrust_vector<T>>;

template <typename T, typename Range>
btasUMTensorThrust<T,Range> gemm(
        const btasUMTensorThrust<T,Range>  &left,
        const btasUMTensorThrust<T,Range>  &right,
        T factor, const TiledArray::math::GemmHelper &gemm_helper) {
  return btas_tensor_gemm_cuda_impl(left, right, factor, gemm_helper);
}

template <typename T, typename Range>
void gemm(
        btasUMTensorThrust<T,Range>  &result,
        const btasUMTensorThrust<T,Range> &left,
        const btasUMTensorThrust<T,Range>  &right,
        T factor, const TiledArray::math::GemmHelper &gemm_helper) {
  return btas_tensor_gemm_cuda_impl(result, left, right, factor, gemm_helper);
}

template <typename T, typename Range>
void add_to(
        btasUMTensorThrust<T,Range>   &result,
        const btasUMTensorThrust<T,Range>  &arg) {
  btas_tensor_add_to_cuda_impl(result, arg);
}

template <typename T, typename Range>
typename btasUMTensorThrust<T, Range> ::value_type
squared_norm(
        const btasUMTensorThrust<T, Range> &arg) {
  return btas_tensor_squared_norm_cuda_impl(arg);
}

} // namespace TiledArray

#ifndef TILEDARRAY_HEADER_ONLY

//  extern template class btas::Tensor<double,btas::RangeNd<CblasRowMajor, std::array<std::size_t, 2>>,TiledArray::cuda_um_btas_varray<double>>;
//  extern template class btas::Tensor<float,btas::RangeNd<CblasRowMajor, std::array<std::size_t, 2>>,TiledArray::cuda_um_btas_varray<float>>;
//  extern template class btas::Tensor<int,btas::RangeNd<CblasRowMajor, std::array<std::size_t, 2>>,TiledArray::cuda_um_btas_varray<int>>;
//  extern template class btas::Tensor<long,btas::RangeNd<CblasRowMajor, std::array<std::size_t, 2>>,TiledArray::cuda_um_btas_varray<long>>;
//
//
//  extern template class btas::Tensor<double,btas::RangeNd<CblasRowMajor, std::array<std::size_t, 2>>,TiledArray::cuda_um_thrust_vector<double>>;
//  extern template class btas::Tensor<float,btas::RangeNd<CblasRowMajor, std::array<std::size_t, 2>>,TiledArray::cuda_um_thrust_vector<float>>;
//  extern template class btas::Tensor<int,btas::RangeNd<CblasRowMajor, std::array<std::size_t, 2>>,TiledArray::cuda_um_thrust_vector<int>>;
//  extern template class btas::Tensor<long,btas::RangeNd<CblasRowMajor, std::array<std::size_t, 2>>,TiledArray::cuda_um_thrust_vector<long>>;

#endif


#endif // TILEDARRAY_HAS_CUDA

#endif //TILEDARRAY_CUDA_TENSOR_CUDA_UM_TENSOR_H
