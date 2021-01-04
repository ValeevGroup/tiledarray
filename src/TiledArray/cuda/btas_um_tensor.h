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

#ifndef TILEDARRAY_CUDA_CUDA_UM_TENSOR_H
#define TILEDARRAY_CUDA_CUDA_UM_TENSOR_H

#include <tiledarray_fwd.h>

#include <TiledArray/external/btas.h>

#ifdef TILEDARRAY_HAS_CUDA

#include <TiledArray/cuda/btas_cublas.h>
#include <TiledArray/cuda/um_storage.h>
#include <TiledArray/external/cutt.h>
#include <TiledArray/tile.h>

namespace TiledArray {

namespace detail {
template <typename T, typename Range>
struct is_cuda_tile<
    ::btas::Tensor<T, Range, TiledArray::cuda_um_btas_varray<T>>>
    : public std::true_type {};

template <typename T>
void to_cuda(const TiledArray::btasUMTensorVarray<T> &tile) {
  cudaSetDevice(TiledArray::cudaEnv::instance()->current_cuda_device_id());
  auto &stream = TiledArray::detail::get_stream_based_on_range(tile.range());
  TiledArray::to_execution_space<TiledArray::ExecutionSpace::CUDA>(
      tile.storage(), stream);
}

}  // end of namespace detail

}  // end of namespace TiledArray

/// serialize functions
namespace madness {
namespace archive {

template <class Archive, typename T>
struct ArchiveLoadImpl<Archive, TiledArray::btasUMTensorVarray<T>> {
  static inline void load(const Archive &ar,
                          TiledArray::btasUMTensorVarray<T> &t) {
    TiledArray::Range range{};
    TiledArray::cuda_um_btas_varray<T> store{};
    ar &range &store;
    t = TiledArray::btasUMTensorVarray<T>(std::move(range), std::move(store));
    // cudaSetDevice(TiledArray::cudaEnv::instance()->current_cuda_device_id());
    // auto &stream = TiledArray::detail::get_stream_based_on_range(range);
    // TiledArray::to_execution_space<TiledArray::ExecutionSpace::CUDA>(t.storage(),
    // stream);
  }
};

template <class Archive, typename T>
struct ArchiveStoreImpl<Archive, TiledArray::btasUMTensorVarray<T>> {
  static inline void store(const Archive &ar,
                           const TiledArray::btasUMTensorVarray<T> &t) {
    CudaSafeCall(cudaSetDevice(
        TiledArray::cudaEnv::instance()->current_cuda_device_id()));
    auto &stream = TiledArray::detail::get_stream_based_on_range(t.range());
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CPU>(t.storage(),
                                                                    stream);
    ar &t.range() & t.storage();
  }
};

}  // namespace archive
}  // namespace madness

namespace TiledArray {
///
/// gemm
///

template <typename T, typename Scalar, typename Range, typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
btasUMTensorVarray<T, Range> gemm(
    const btasUMTensorVarray<T, Range> &left,
    const btasUMTensorVarray<T, Range> &right, Scalar factor,
    const TiledArray::math::GemmHelper &gemm_helper) {
  return btas_tensor_gemm_cuda_impl(left, right, factor, gemm_helper);
}

template <typename T, typename Scalar, typename Range, typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
void gemm(btasUMTensorVarray<T, Range> &result,
          const btasUMTensorVarray<T, Range> &left,
          const btasUMTensorVarray<T, Range> &right, Scalar factor,
          const TiledArray::math::GemmHelper &gemm_helper) {
  return btas_tensor_gemm_cuda_impl(result, left, right, factor, gemm_helper);
}

///
/// clone
///

template <typename T, typename Range>
btasUMTensorVarray<T, Range> clone(const btasUMTensorVarray<T, Range> &arg) {
  // TODO how to copy Unified Memory? from CPU or GPU? currently
  //  always copy on GPU, but need to investigate
  return btas_tensor_clone_cuda_impl(arg);
}

///
/// shift
///
template <typename T, typename Range, typename Index>
btasUMTensorVarray<T, Range> shift(const btasUMTensorVarray<T, Range> &arg,
                                   const Index &range_shift) {
  // make a copy of the old range
  Range result_range(arg.range());
  // shift the range
  result_range.inplace_shift(range_shift);

  CudaSafeCall(cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));

  // @important select the stream using the shifted range
  auto &cuda_stream = detail::get_stream_based_on_range(result_range);

  typename btasUMTensorVarray<T, Range>::storage_type result_storage;

  make_device_storage(result_storage, result_range.volume(), cuda_stream);
  btasUMTensorVarray<T, Range> result(std::move(result_range),
                                      std::move(result_storage));

  // call cublasCopy
  const auto &handle = cuBLASHandlePool::handle();
  CublasSafeCall(cublasSetStream(handle, cuda_stream));

  CublasSafeCall(cublasCopy(handle, result.size(), device_data(arg.storage()),
                            1, device_data(result.storage()), 1));

  synchronize_stream(&cuda_stream);
  return result;
}

///
/// shift to
///
template <typename T, typename Range, typename Index>
btasUMTensorVarray<T, Range>& shift_to(btasUMTensorVarray<T, Range> &arg,
                                      const Index &range_shift) {
  const_cast<Range &>(arg.range()).inplace_shift(range_shift);
  return arg;
}

///
/// permute
///

template <typename T, typename Range>
btasUMTensorVarray<T, Range> permute(const btasUMTensorVarray<T, Range> &arg,
                                     const TiledArray::Permutation &perm) {
  // compute result range
  auto result_range = perm * arg.range();
  CudaSafeCall(cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));

  // compute the stream to use
  auto &stream = detail::get_stream_based_on_range(result_range);

  // allocate result memory
  typename btasUMTensorVarray<T, Range>::storage_type storage;
  make_device_storage(storage, result_range.area(), stream);

  btasUMTensorVarray<T, Range> result(std::move(result_range),
                                      std::move(storage));

  // invoke the permute function
  cutt_permute(const_cast<T *>(device_data(arg.storage())),
               device_data(result.storage()), arg.range(), perm, stream);

  synchronize_stream(&stream);

  return result;
}

///
/// scale
///

template <typename T, typename Range, typename Scalar, typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
btasUMTensorVarray<T, Range> scale(const btasUMTensorVarray<T, Range> &arg,
                                   const Scalar factor) {
  detail::to_cuda(arg);
  return btas_tensor_scale_cuda_impl(arg, factor);
}

template <typename T, typename Range, typename Scalar, typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
btasUMTensorVarray<T, Range>& scale_to(btasUMTensorVarray<T, Range> &arg, const Scalar factor) {
  detail::to_cuda(arg);
  btas_tensor_scale_to_cuda_impl(arg, factor);
  return arg;
}

template <typename T, typename Range, typename Scalar, typename Perm, typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar> && TiledArray::detail::is_permutation_v<Perm>>>
btasUMTensorVarray<T, Range> scale(const btasUMTensorVarray<T, Range> &arg,
                                   const Scalar factor,
                                   const Perm &perm) {
  auto result = scale(arg, factor);

  // wait to finish before switch stream
  auto stream = tls_cudastream_accessor();
  cudaStreamSynchronize(*stream);

  return permute(result, perm);
}

///
/// neg
///

template <typename T, typename Range>
btasUMTensorVarray<T, Range> neg(const btasUMTensorVarray<T, Range> &arg) {
  detail::to_cuda(arg);
  return btas_tensor_scale_cuda_impl(arg, T(-1.0));
}

template <typename T, typename Range, typename Perm, typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
btasUMTensorVarray<T, Range> neg(const btasUMTensorVarray<T, Range> &arg,
                                 const Perm &perm) {
  auto result = neg(arg);

  // wait to finish before switch stream
  auto stream = tls_cudastream_accessor();
  cudaStreamSynchronize(*stream);

  return permute(result, perm);
}

template <typename T, typename Range>
btasUMTensorVarray<T, Range>& neg_to(btasUMTensorVarray<T, Range> &arg) {
  detail::to_cuda(arg);
  btas_tensor_scale_to_cuda_impl(arg, T(-1.0));
  return arg;
}

///
/// subt
///

template <typename T, typename Range>
btasUMTensorVarray<T, Range> subt(const btasUMTensorVarray<T, Range> &arg1,
                                  const btasUMTensorVarray<T, Range> &arg2) {
  detail::to_cuda(arg1);
  detail::to_cuda(arg2);
  return btas_tensor_subt_cuda_impl(arg1, arg2, T(1.0));
}

template <typename T, typename Scalar, typename Range, typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
btasUMTensorVarray<T, Range> subt(const btasUMTensorVarray<T, Range> &arg1,
                                  const btasUMTensorVarray<T, Range> &arg2,
                                  const Scalar factor) {
  auto result = subt(arg1, arg2);
  btas_tensor_scale_to_cuda_impl(result, factor);
  return result;
}

template <typename T, typename Range, typename Perm, typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
btasUMTensorVarray<T, Range> subt(const btasUMTensorVarray<T, Range> &arg1,
                                  const btasUMTensorVarray<T, Range> &arg2,
                                  const Perm &perm) {
  auto result = subt(arg1, arg2);

  // wait to finish before switch stream
  auto stream = tls_cudastream_accessor();
  cudaStreamSynchronize(*stream);

  return permute(result, perm);
}

template <typename T, typename Scalar, typename Range, typename Perm, typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar> && TiledArray::detail::is_permutation_v<Perm>>>
btasUMTensorVarray<T, Range> subt(const btasUMTensorVarray<T, Range> &arg1,
                                  const btasUMTensorVarray<T, Range> &arg2,
                                  const Scalar factor,
                                  const Perm &perm) {
  auto result = subt(arg1, arg2, factor);

  // wait to finish before switch stream
  auto stream = tls_cudastream_accessor();
  cudaStreamSynchronize(*stream);

  return permute(result, perm);
}

///
/// subt_to
///

template <typename T, typename Range>
btasUMTensorVarray<T, Range>& subt_to(btasUMTensorVarray<T, Range> &result,
             const btasUMTensorVarray<T, Range> &arg1) {
  detail::to_cuda(result);
  detail::to_cuda(arg1);
  btas_tensor_subt_to_cuda_impl(result, arg1, T(1.0));
  return result;
}

template <typename T, typename Scalar, typename Range, typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
btasUMTensorVarray<T, Range>& subt_to(btasUMTensorVarray<T, Range> &result,
             const btasUMTensorVarray<T, Range> &arg1, const Scalar factor) {
  subt_to(result, arg1);
  btas_tensor_scale_to_cuda_impl(result, factor);
  return result;
}

///
/// add
///

template <typename T, typename Range>
btasUMTensorVarray<T, Range> add(const btasUMTensorVarray<T, Range> &arg1,
                                 const btasUMTensorVarray<T, Range> &arg2) {
  detail::to_cuda(arg1);
  detail::to_cuda(arg2);
  return btas_tensor_add_cuda_impl(arg1, arg2, T(1.0));
}

template <typename T, typename Scalar, typename Range, typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
btasUMTensorVarray<T, Range> add(const btasUMTensorVarray<T, Range> &arg1,
                                 const btasUMTensorVarray<T, Range> &arg2,
                                 const Scalar factor) {
  auto result = add(arg1, arg2);
  btas_tensor_scale_to_cuda_impl(result, factor);
  return result;
}

template <typename T, typename Scalar, typename Range, typename Perm, typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar> && TiledArray::detail::is_permutation_v<Perm>>>
btasUMTensorVarray<T, Range> add(const btasUMTensorVarray<T, Range> &arg1,
                                 const btasUMTensorVarray<T, Range> &arg2,
                                 const Scalar factor,
                                 const Perm &perm) {
  auto result = add(arg1, arg2, factor);

  // wait to finish before switch stream
  auto stream = tls_cudastream_accessor();
  cudaStreamSynchronize(*stream);

  return permute(result, perm);
}

template <typename T, typename Range, typename Perm, typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
btasUMTensorVarray<T, Range> add(const btasUMTensorVarray<T, Range> &arg1,
                                 const btasUMTensorVarray<T, Range> &arg2,
                                 const Perm &perm) {
  auto result = add(arg1, arg2);

  // wait to finish before switch stream
  auto stream = tls_cudastream_accessor();
  cudaStreamSynchronize(*stream);

  return permute(result, perm);
}

///
/// add_to
///

template <typename T, typename Range>
btasUMTensorVarray<T, Range>& add_to(btasUMTensorVarray<T, Range> &result,
            const btasUMTensorVarray<T, Range> &arg) {
  detail::to_cuda(result);
  detail::to_cuda(arg);
  btas_tensor_add_to_cuda_impl(result, arg, T(1.0));
  return result;
}

template <typename T, typename Scalar, typename Range, typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
btasUMTensorVarray<T, Range>& add_to(btasUMTensorVarray<T, Range> &result,
            const btasUMTensorVarray<T, Range> &arg, const Scalar factor) {
  add_to(result, arg);
  btas_tensor_scale_to_cuda_impl(result, factor);
  return result;
}

///
/// dot
///
template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type dot(
    const btasUMTensorVarray<T, Range> &arg1,
    const btasUMTensorVarray<T, Range> &arg2) {
  detail::to_cuda(arg1);
  detail::to_cuda(arg2);
  return btas_tensor_dot_cuda_impl(arg1, arg2);
}

///
/// mult
///
template <typename T, typename Range>
btasUMTensorVarray<T, Range> mult(const btasUMTensorVarray<T, Range> &arg1,
                                  const btasUMTensorVarray<T, Range> &arg2) {
  detail::to_cuda(arg1);
  detail::to_cuda(arg2);
  return btas_tensor_mult_cuda_impl(arg1, arg2);
}

template <typename T, typename Scalar, typename Range, typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
btasUMTensorVarray<T, Range> mult(const btasUMTensorVarray<T, Range> &arg1,
                                  const btasUMTensorVarray<T, Range> &arg2,
                                  const Scalar factor) {
  auto result = mult(arg1, arg2);
  btas_tensor_scale_to_cuda_impl(result, factor);
  return result;
}

template <typename T, typename Range, typename Perm, typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
btasUMTensorVarray<T, Range> mult(const btasUMTensorVarray<T, Range> &arg1,
                                  const btasUMTensorVarray<T, Range> &arg2,
                                  const Perm &perm) {
  auto result = mult(arg1, arg2);

  // wait to finish before switch stream
  auto stream = tls_cudastream_accessor();
  cudaStreamSynchronize(*stream);

  return permute(result, perm);
}

template <typename T, typename Range, typename Scalar, typename Perm, typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar> && TiledArray::detail::is_permutation_v<Perm>>>
btasUMTensorVarray<T, Range> mult(const btasUMTensorVarray<T, Range> &arg1,
                                  const btasUMTensorVarray<T, Range> &arg2,
                                  const Scalar factor,
                                  const Perm &perm) {
  auto result = mult(arg1, arg2, factor);

  // wait to finish before switch stream
  auto stream = tls_cudastream_accessor();
  cudaStreamSynchronize(*stream);

  return permute(result, perm);
}

///
/// mult to
///
template <typename T, typename Range>
btasUMTensorVarray<T, Range>& mult_to(btasUMTensorVarray<T, Range> &result,
             const btasUMTensorVarray<T, Range> &arg) {
  detail::to_cuda(result);
  detail::to_cuda(arg);
  btas_tensor_mult_to_cuda_impl(result, arg);
  return result;
}

template <typename T, typename Scalar, typename Range, typename = std::enable_if_t<TiledArray::detail::is_numeric_v<Scalar>>>
btasUMTensorVarray<T, Range>& mult_to(btasUMTensorVarray<T, Range> &result,
             const btasUMTensorVarray<T, Range> &arg, const Scalar factor) {
  mult_to(result, arg);
  btas_tensor_scale_to_cuda_impl(result, factor);
  return result;
}

///
/// reduction operations
///

///
/// squared_norm
///

template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type squared_norm(
    const btasUMTensorVarray<T, Range> &arg) {
  detail::to_cuda(arg);
  return btas_tensor_squared_norm_cuda_impl(arg);
}

///
/// norm
///

template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type norm(
    const btasUMTensorVarray<T, Range> &arg) {
  detail::to_cuda(arg);
  return std::sqrt(btas_tensor_squared_norm_cuda_impl(arg));
}

///
/// trace
///
template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type trace(
    const btasUMTensorVarray<T, Range> &arg) {
  assert(false);
}

///
/// sum
///
template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type sum(
    const btasUMTensorVarray<T, Range> &arg) {
  detail::to_cuda(arg);
  return btas_tensor_sum_cuda_impl(arg);
}

///
/// product
///
template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type product(
    const btasUMTensorVarray<T, Range> &arg) {
  detail::to_cuda(arg);
  return btas_tensor_product_cuda_impl(arg);
}

///
/// max
///
template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type max(
    const btasUMTensorVarray<T, Range> &arg) {
  detail::to_cuda(arg);
  return btas_tensor_max_cuda_impl(arg);
}

///
/// abs_max
///
template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type abs_max(
    const btasUMTensorVarray<T, Range> &arg) {
  detail::to_cuda(arg);
  return btas_tensor_absmax_cuda_impl(arg);
}

///
/// min
///
template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type min(
    const btasUMTensorVarray<T, Range> &arg) {
  detail::to_cuda(arg);
  return btas_tensor_min_cuda_impl(arg);
}

///
/// abs min
///
template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type abs_min(
    const btasUMTensorVarray<T, Range> &arg) {
  detail::to_cuda(arg);
  return btas_tensor_absmin_cuda_impl(arg);
}

/// to host for UM Array
template <typename UMTensor, typename Policy>
void to_host(
    TiledArray::DistArray<TiledArray::Tile<UMTensor>, Policy> &um_array) {
  auto to_host = [](TiledArray::Tile<UMTensor> &tile) {
    CudaSafeCall(cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));
    auto &stream = detail::get_stream_based_on_range(tile.range());

    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CPU>(
        tile.tensor().storage(), stream);
  };

  auto &world = um_array.world();

  auto start = um_array.pmap()->begin();
  auto end = um_array.pmap()->end();

  for (; start != end; ++start) {
    if (!um_array.is_zero(*start)) {
      world.taskq.add(to_host, um_array.find(*start));
    }
  }

  world.gop.fence();
  CudaSafeCall(cudaDeviceSynchronize());
};

/// to device for UM Array
template <typename UMTensor, typename Policy>
void to_device(
    TiledArray::DistArray<TiledArray::Tile<UMTensor>, Policy> &um_array) {
  auto to_device = [](TiledArray::Tile<UMTensor> &tile) {
    CudaSafeCall(cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));
    auto &stream = detail::get_stream_based_on_range(tile.range());

    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CUDA>(
        tile.tensor().storage(), stream);
  };

  auto &world = um_array.world();

  auto start = um_array.pmap()->begin();
  auto end = um_array.pmap()->end();

  for (; start != end; ++start) {
    if (!um_array.is_zero(*start)) {
      world.taskq.add(to_device, um_array.find(*start));
    }
  }

  world.gop.fence();
  CudaSafeCall(cudaDeviceSynchronize());
};

/// convert array from UMTensor to TiledArray::Tensor
template <typename UMTensor, typename TATensor, typename Policy>
typename std::enable_if<!std::is_same<UMTensor, TATensor>::value,
                        TiledArray::DistArray<TATensor, Policy>>::type
um_tensor_to_ta_tensor(
    const TiledArray::DistArray<UMTensor, Policy> &um_array) {
  const auto convert_tile_memcpy = [](const UMTensor &tile) {
    TATensor result(tile.tensor().range());

    auto &stream = cudaEnv::instance()->cuda_stream_d2h();
    CudaSafeCall(
        cudaMemcpyAsync(result.data(), tile.data(),
                        tile.size() * sizeof(typename TATensor::value_type),
                        cudaMemcpyDefault, stream));
    synchronize_stream(&stream);

    return result;
  };

  const auto convert_tile_um = [](const UMTensor &tile) {
    TATensor result(tile.tensor().range());
    using std::begin;
    const auto n = tile.tensor().size();

    CudaSafeCall(cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));
    auto &stream = detail::get_stream_based_on_range(tile.range());

    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CPU>(
        tile.tensor().storage(), stream);

    std::copy_n(tile.data(), n, result.data());

    return result;
  };

  const char *use_legacy_conversion =
      std::getenv("TA_CUDA_LEGACY_UM_CONVERSION");
  auto ta_array = use_legacy_conversion
                      ? to_new_tile_type(um_array, convert_tile_um)
                      : to_new_tile_type(um_array, convert_tile_memcpy);

  um_array.world().gop.fence();
  return ta_array;
}

/// no-op if UMTensor is the same type as TATensor type
template <typename UMTensor, typename TATensor, typename Policy>
typename std::enable_if<std::is_same<UMTensor, TATensor>::value,
                        TiledArray::DistArray<UMTensor, Policy>>::type
um_tensor_to_ta_tensor(
    const TiledArray::DistArray<UMTensor, Policy> &um_array) {
  return um_array;
}

/// convert array from TiledArray::Tensor to UMTensor
template <typename UMTensor, typename TATensor, typename Policy>
typename std::enable_if<!std::is_same<UMTensor, TATensor>::value,
                        TiledArray::DistArray<UMTensor, Policy>>::type
ta_tensor_to_um_tensor(const TiledArray::DistArray<TATensor, Policy> &array) {
  auto convert_tile_memcpy = [](const TATensor &tile) {
    /// UMTensor must be wrapped into TA::Tile

    CudaSafeCall(cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));

    using Tensor = typename UMTensor::tensor_type;

    auto &stream = cudaEnv::instance()->cuda_stream_h2d();
    typename Tensor::storage_type storage;
    make_device_storage(storage, tile.range().area(), stream);
    Tensor result(tile.range(), std::move(storage));

    CudaSafeCall(
        cudaMemcpyAsync(result.data(), tile.data(),
                        tile.size() * sizeof(typename Tensor::value_type),
                        cudaMemcpyDefault, stream));

    synchronize_stream(&stream);
    return TiledArray::Tile<Tensor>(std::move(result));
  };

  auto convert_tile_um = [](const TATensor &tile) {
    /// UMTensor must be wrapped into TA::Tile

    CudaSafeCall(cudaSetDevice(cudaEnv::instance()->current_cuda_device_id()));

    using Tensor = typename UMTensor::tensor_type;
    typename Tensor::storage_type storage(tile.range().area());

    Tensor result(tile.range(), std::move(storage));

    const auto n = tile.size();

    std::copy_n(tile.data(), n, result.data());

    auto &stream = detail::get_stream_based_on_range(result.range());

    // prefetch data to GPU
    TiledArray::to_execution_space<TiledArray::ExecutionSpace::CUDA>(
        result.storage(), stream);

    return TiledArray::Tile<Tensor>(std::move(result));
  };

  const char *use_legacy_conversion =
      std::getenv("TA_CUDA_LEGACY_UM_CONVERSION");
  auto um_array = use_legacy_conversion
                      ? to_new_tile_type(array, convert_tile_um)
                      : to_new_tile_type(array, convert_tile_memcpy);

  array.world().gop.fence();
  return um_array;
}

/// no-op if array is the same as return type
template <typename UMTensor, typename TATensor, typename Policy>
typename std::enable_if<std::is_same<UMTensor, TATensor>::value,
                        TiledArray::DistArray<UMTensor, Policy>>::type
ta_tensor_to_um_tensor(const TiledArray::DistArray<UMTensor, Policy> &array) {
  return array;
}

}  // namespace TiledArray

#ifndef TILEDARRAY_HEADER_ONLY

extern template class btas::varray<double,
                                   TiledArray::cuda_um_allocator<double>>;
extern template class btas::varray<float, TiledArray::cuda_um_allocator<float>>;
extern template class btas::varray<int, TiledArray::cuda_um_allocator<int>>;
extern template class btas::varray<long, TiledArray::cuda_um_allocator<long>>;

extern template class btas::Tensor<double, TiledArray::Range,
                                   TiledArray::cuda_um_btas_varray<double>>;
extern template class btas::Tensor<float, TiledArray::Range,
                                   TiledArray::cuda_um_btas_varray<float>>;
extern template class btas::Tensor<int, TiledArray::Range,
                                   TiledArray::cuda_um_btas_varray<int>>;
extern template class btas::Tensor<long, TiledArray::Range,
                                   TiledArray::cuda_um_btas_varray<long>>;

extern template class TiledArray::Tile<btas::Tensor<
    double, TiledArray::Range, TiledArray::cuda_um_btas_varray<double>>>;
extern template class TiledArray::Tile<btas::Tensor<
    float, TiledArray::Range, TiledArray::cuda_um_btas_varray<float>>>;
extern template class TiledArray::Tile<
    btas::Tensor<int, TiledArray::Range, TiledArray::cuda_um_btas_varray<int>>>;
extern template class TiledArray::Tile<btas::Tensor<
    long, TiledArray::Range, TiledArray::cuda_um_btas_varray<long>>>;

#endif  // TILEDARRAY_HEADER_ONLY

#endif  // TILEDARRAY_HAS_CUDA

#endif  // TILEDARRAY_CUDA_CUDA_UM_TENSOR_H
