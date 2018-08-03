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
 *  MERCHANTiledArrayBILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
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

#include <cutt.h>
#include <TiledArray/range.h>
#include <TiledArray/tensor/cuda/btas_cublas.h>
#include <TiledArray/tensor/tensor.h>

/*
 * btas::Tensor with UM storage cuda_um_btas_varray
 */

template <typename T, typename Range = TiledArray::Range>
using btasUMTensorVarray =
    btas::Tensor<T, Range, TiledArray::cuda_um_btas_varray<T>>;

namespace TiledArray {

template <typename T, typename Range>
struct eval_trait<btasUMTensorVarray<T, Range>> {
  typedef btasUMTensorVarray<T, Range> type;
};

template <typename T, typename Range>
btasUMTensorVarray<T, Range> gemm(
    const btasUMTensorVarray<T, Range> &left,
    const btasUMTensorVarray<T, Range> &right, T factor,
    const TiledArray::math::GemmHelper &gemm_helper) {
  return btas_tensor_gemm_cuda_impl(left, right, factor, gemm_helper);
}

template <typename T, typename Range>
void gemm(btasUMTensorVarray<T, Range> &result,
          const btasUMTensorVarray<T, Range> &left,
          const btasUMTensorVarray<T, Range> &right, T factor,
          const TiledArray::math::GemmHelper &gemm_helper) {
  return btas_tensor_gemm_cuda_impl(result, left, right, factor, gemm_helper);
}

template <typename T, typename Range>
btasUMTensorVarray<T, Range> clone(const btasUMTensorVarray<T, Range> &arg) {
  return btas_tensor_clone_cuda_impl(arg);
}

template <typename T, typename Range>
btasUMTensorVarray<T, Range> permute(const btasUMTensorVarray<T, Range> &arg,
        const TiledArray::Permutation& perm) {

  auto& stream = detail::get_stream_based_on_range(arg.range());
  cudaSetDevice(cudaEnv::instance()->current_cuda_device_id());

  // compute result range
  auto result_range = perm*arg.range();

  auto extent = result_range.extent();
  std::vector<int> extent_int (extent.begin(), extent.end());

  std::vector<int> perm_int (perm.begin(), perm.end());


  // allocate result memory
  typename btasUMTensorVarray<T, Range>::storage_type storage;
  make_device_storage(storage, result_range.area(), stream);

  btasUMTensorVarray<T, Range> result(std::move(result_range),std::move(storage));

  cuttResult_t status;

  cuttHandle plan;
  status = cuttPlan(&plan, arg.rank(), extent_int.data(), perm_int.data(), sizeof(T), stream);

  TA_ASSERT(status == CUTT_SUCCESS);

  status = cuttExecute(plan, const_cast<T*>(device_data(arg.storage())), device_data(result.storage()));

  TA_ASSERT(status == CUTT_SUCCESS);

  cuttDestroy(plan);
  return result;
}

template <typename T, typename Range, typename Scalar>
btasUMTensorVarray<T, Range> scale(const btasUMTensorVarray<T, Range> &arg,
                                   const Scalar factor) {
  return btas_tensor_scale_cuda_impl(arg, factor);
}

template <typename T, typename Range, typename Scalar>
btasUMTensorVarray<T, Range> scale(const btasUMTensorVarray<T, Range> &arg,
                                   const Scalar factor,
                                   const TiledArray::Permutation &perm) {
  /// only support identity permation for now
  TA_ASSERT(perm == perm.identity());
  return btas_tensor_scale_cuda_impl(arg, factor);
}

template <typename T, typename Range, typename Scalar>
void scale_to(btasUMTensorVarray<T, Range> &arg,
                                   const Scalar factor) {
  btas_tensor_scale_to_cuda_impl(arg, factor);
}

template <typename T, typename Range>
btasUMTensorVarray<T, Range> neg(const btasUMTensorVarray<T, Range> &arg) {
  return btas_tensor_scale_cuda_impl(arg, T(-1.0));
}

template <typename T, typename Range>
btasUMTensorVarray<T, Range> neg(const btasUMTensorVarray<T, Range> &arg,
                                 const TiledArray::Permutation &perm) {
  /// only support identity permation for now
  TA_ASSERT(perm == perm.identity());
  return btas_tensor_scale_cuda_impl(arg, T(-1.0));
}

template <typename T, typename Range>
void neg_to(btasUMTensorVarray<T, Range> &arg) {
  btas_tensor_scale_to_cuda_impl(arg, T(-1.0));
}

template <typename T, typename Range>
btasUMTensorVarray<T, Range> subt(const btasUMTensorVarray<T, Range> &arg1,
                                  const btasUMTensorVarray<T, Range> &arg2) {
  return btas_tensor_subt_cuda_impl(arg1, arg2, T(1.0));
}

template <typename T, typename Range>
void subt_to(btasUMTensorVarray<T, Range> &result,
             const btasUMTensorVarray<T, Range> &arg1) {
  btas_tensor_subt_to_cuda_impl(result, arg1, T(1.0));
}

template <typename T, typename Scalar, typename Range>
void subt_to(btasUMTensorVarray<T, Range> &result,
             const btasUMTensorVarray<T, Range> &arg1, const Scalar factor) {
  btas_tensor_subt_to_cuda_impl(result, arg1, T(1.0));
  btas_tensor_scale_to_cuda_impl(result, factor);
}

template <typename T, typename Range>
btasUMTensorVarray<T, Range> subt(const btasUMTensorVarray<T, Range> &arg1,
                                  const btasUMTensorVarray<T, Range> &arg2,
                                  const TiledArray::Permutation &perm) {
  /// only support identity permation for now
  TA_ASSERT(perm == perm.identity());
  return subt(arg1, arg2);
}

template <typename T, typename Range>
void add_to(btasUMTensorVarray<T, Range> &result,
            const btasUMTensorVarray<T, Range> &arg) {
  btas_tensor_add_to_cuda_impl(result, arg, T(1.0));
}

template <typename T, typename Range>
btasUMTensorVarray<T, Range> add(const btasUMTensorVarray<T, Range> &arg1,
                                 const btasUMTensorVarray<T, Range> &arg2) {
  return btas_tensor_add_cuda_impl(arg1, arg2, T(1.0));
}

template <typename T, typename Range>
btasUMTensorVarray<T, Range> add(const btasUMTensorVarray<T, Range> &arg1,
                                 const btasUMTensorVarray<T, Range> &arg2,
                                 const TiledArray::Permutation &perm) {
  /// only support identity permation for now
  TA_ASSERT(perm == perm.identity());
  return btas_tensor_add_cuda_impl(arg1, arg2, T(1.0));
}

template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type squared_norm(
    const btasUMTensorVarray<T, Range> &arg) {
  return btas_tensor_squared_norm_cuda_impl(arg);
}

template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type norm(
    const btasUMTensorVarray<T, Range> &arg) {
  return std::sqrt(btas_tensor_squared_norm_cuda_impl(arg));
}

/*
 * btas::Tensor with UM storage cuda_um_thrust_vector
 */
template <typename T, typename Range = TiledArray::Range>
using btasUMTensorThrust =
    btas::Tensor<T, Range, TiledArray::cuda_um_thrust_vector<T>>;

template <typename T, typename Range>
struct eval_trait<btasUMTensorThrust<T, Range>> {
  typedef btasUMTensorThrust<T, Range> type;
};

template <typename T, typename Range>
btasUMTensorThrust<T, Range> gemm(
        const btasUMTensorThrust<T, Range> &left,
        const btasUMTensorThrust<T, Range> &right, T factor,
        const TiledArray::math::GemmHelper &gemm_helper) {
  return btas_tensor_gemm_cuda_impl(left, right, factor, gemm_helper);
}

template <typename T, typename Range>
void gemm(btasUMTensorThrust<T, Range> &result,
          const btasUMTensorThrust<T, Range> &left,
          const btasUMTensorThrust<T, Range> &right, T factor,
          const TiledArray::math::GemmHelper &gemm_helper) {
  return btas_tensor_gemm_cuda_impl(result, left, right, factor, gemm_helper);
}

template <typename T, typename Range>
btasUMTensorThrust<T, Range> clone(const btasUMTensorThrust<T, Range> &arg) {
  return btas_tensor_clone_cuda_impl(arg);
}

template <typename T, typename Range, typename Scalar>
btasUMTensorThrust<T, Range> scale(const btasUMTensorThrust<T, Range> &arg,
                                   const Scalar factor) {
  return btas_tensor_scale_cuda_impl(arg, factor);
}

template <typename T, typename Range, typename Scalar>
btasUMTensorThrust<T, Range> scale(const btasUMTensorThrust<T, Range> &arg,
                                   const Scalar factor,
                                   const TiledArray::Permutation &perm) {
  /// only support identity permation for now
  TA_ASSERT(perm == perm.identity());
  return btas_tensor_scale_cuda_impl(arg, factor);
}

template <typename T, typename Range, typename Scalar>
void scale_to(btasUMTensorThrust<T, Range> &arg,
              const Scalar factor) {
  btas_tensor_scale_to_cuda_impl(arg, factor);
}

template <typename T, typename Range>
btasUMTensorThrust<T, Range> neg(const btasUMTensorThrust<T, Range> &arg) {
  return btas_tensor_scale_cuda_impl(arg, T(-1.0));
}

template <typename T, typename Range>
btasUMTensorThrust<T, Range> neg(const btasUMTensorThrust<T, Range> &arg,
                                 const TiledArray::Permutation &perm) {
  /// only support identity permation for now
  TA_ASSERT(perm == perm.identity());
  return btas_tensor_scale_cuda_impl(arg, T(-1.0));
}

template <typename T, typename Range>
void neg_to(btasUMTensorThrust<T, Range> &arg) {
  btas_tensor_scale_to_cuda_impl(arg, T(-1.0));
}

template <typename T, typename Range>
btasUMTensorThrust<T, Range> subt(const btasUMTensorThrust<T, Range> &arg1,
                                  const btasUMTensorThrust<T, Range> &arg2) {
  return btas_tensor_subt_cuda_impl(arg1, arg2, T(1.0));
}

template <typename T, typename Range>
void subt_to(btasUMTensorThrust<T, Range> &result,
             const btasUMTensorThrust<T, Range> &arg1) {
  btas_tensor_subt_to_cuda_impl(result, arg1, T(1.0));
}

template <typename T, typename Scalar, typename Range>
void subt_to(btasUMTensorThrust<T, Range> &result,
             const btasUMTensorThrust<T, Range> &arg1, const Scalar factor) {
  btas_tensor_subt_to_cuda_impl(result, arg1, T(1.0));
  btas_tensor_scale_to_cuda_impl(result, factor);
}

template <typename T, typename Range>
btasUMTensorThrust<T, Range> subt(const btasUMTensorThrust<T, Range> &arg1,
                                  const btasUMTensorThrust<T, Range> &arg2,
                                  const TiledArray::Permutation &perm) {
  /// only support identity permation for now
  TA_ASSERT(perm == perm.identity());
  return subt(arg1, arg2);
}

template <typename T, typename Range>
void add_to(btasUMTensorThrust<T, Range> &result,
            const btasUMTensorThrust<T, Range> &arg) {
  btas_tensor_add_to_cuda_impl(result, arg, T(1.0));
}

template <typename T, typename Range>
btasUMTensorThrust<T, Range> add(const btasUMTensorThrust<T, Range> &arg1,
                                 const btasUMTensorThrust<T, Range> &arg2) {
  return btas_tensor_add_cuda_impl(arg1, arg2, T(1.0));
}

template <typename T, typename Range>
btasUMTensorThrust<T, Range> add(const btasUMTensorThrust<T, Range> &arg1,
                                 const btasUMTensorThrust<T, Range> &arg2,
                                 const TiledArray::Permutation &perm) {
  /// only support identity permation for now
  TA_ASSERT(perm == perm.identity());
  return btas_tensor_add_cuda_impl(arg1, arg2, T(1.0));
}

template <typename T, typename Range>
typename btasUMTensorThrust<T, Range>::value_type squared_norm(
        const btasUMTensorThrust<T, Range> &arg) {
  return btas_tensor_squared_norm_cuda_impl(arg);
}

template <typename T, typename Range>
typename btasUMTensorThrust<T, Range>::value_type norm(
        const btasUMTensorThrust<T, Range> &arg) {
  return std::sqrt(btas_tensor_squared_norm_cuda_impl(arg));
}

/// to host for UM Array
template <typename UMTensor, typename Policy>
void to_host(
    TiledArray::DistArray<TiledArray::Tile<UMTensor>, Policy> &um_array) {
  auto to_host = [](TiledArray::Tile<UMTensor> &tile) {

    auto &stream = detail::get_stream_based_on_range(tile.range());

    // do norm on GPU
    auto tile_norm = norm(tile.tensor());

    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CPU>(
        tile.tensor().storage(), stream);

    return tile_norm;
  };

  foreach_inplace(um_array, to_host);
  um_array.world().gop.fence();
  cudaDeviceSynchronize();
};

/// to device for UM Array
template <typename UMTensor, typename Policy>
void to_device(
    TiledArray::DistArray<TiledArray::Tile<UMTensor>, Policy> &um_array) {
  auto to_device = [](TiledArray::Tile<UMTensor> &tile) {

    auto &stream = detail::get_stream_based_on_range(tile.range());

    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CUDA>(
        tile.tensor().storage(), stream);

    return norm(tile.tensor());
  };

  foreach_inplace(um_array, to_device);
  um_array.world().gop.fence();
  cudaDeviceSynchronize();
};

/// convert array from UMTensor to TiledArray::Tensor
template <typename T, typename UMTensor, typename Policy>
TiledArray::DistArray<TiledArray::Tensor<T>, Policy> um_tensor_to_ta_tensor(
    TiledArray::DistArray<TiledArray::Tile<UMTensor>, Policy> &um_array) {
  const auto convert_tile = [](const TiledArray::Tile<UMTensor> &tile) {
    TiledArray::Tensor<T> result(tile.tensor().range());
    using std::begin;
    const auto n = tile.tensor().size();
    for (std::size_t i = 0; i < n; i++) {
      result[i] = tile[i];
    }
    return result;
  };

  to_host(um_array);

  auto ta_array = to_new_tile_type(um_array, convert_tile);

  um_array.world().gop.fence();
  return ta_array;
};

/// convert array from TiledArray::Tensor to UMTensor
template <typename T, typename UMTensor, typename Policy>
TiledArray::DistArray<TiledArray::Tile<UMTensor>, Policy>
ta_tensor_to_um_tensor(
    TiledArray::DistArray<TiledArray::Tensor<T>, Policy> &array) {

  auto convert_tile = [](const TiledArray::Tensor<T> &tile) {
    typename UMTensor::storage_type storage(tile.range().area());

    UMTensor result(tile.range(), std::move(storage));

    const auto n = tile.size();
    for (std::size_t i = 0; i < n; i++) {
      result[i] = tile[i];
    }

    return TiledArray::Tile<UMTensor>(result);
  };

  auto um_array = to_new_tile_type(array, convert_tile);

  array.world().gop.fence();
  return um_array;
};

}  // namespace TiledArray

#ifndef TILEDARRAY_HEADER_ONLY

//  extern template class
//  btas::Tensor<double,TiledArray::Range,TiledArray::cuda_um_btas_varray<double>>;
//  extern template class
//  btas::Tensor<float,TiledArray::Range,TiledArray::cuda_um_btas_varray<float>>;
//  extern template class
//  btas::Tensor<int,TiledArray::Range,TiledArray::cuda_um_btas_varray<int>>;
//  extern template class
//  btas::Tensor<long,TiledArray::Range,TiledArray::cuda_um_btas_varray<long>>;
//
//
//  extern template class
//  btas::Tensor<double,TiledArray::Range,TiledArray::cuda_um_thrust_vector<double>>;
//  extern template class
//  btas::Tensor<float,TiledArray::Range,TiledArray::cuda_um_thrust_vector<float>>;
//  extern template class
//  btas::Tensor<int,TiledArray::Range,TiledArray::cuda_um_thrust_vector<int>>;
//  extern template class
//  btas::Tensor<long,TiledArray::Range,TiledArray::cuda_um_thrust_vector<long>>;

#endif

#endif  // TILEDARRAY_HAS_CUDA

#endif  // TILEDARRAY_CUDA_TENSOR_CUDA_UM_TENSOR_H
