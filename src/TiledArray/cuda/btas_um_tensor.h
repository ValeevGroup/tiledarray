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

#include <TiledArray/external/btas.h>
#include <TiledArray/cuda/um_storage.h>

#ifdef TILEDARRAY_HAS_CUDA

#include <TiledArray/external/cutt.h>
#include <TiledArray/range.h>
#include <TiledArray/cuda/btas_cublas.h>
#include <TiledArray/tensor/tensor.h>
#include <TiledArray/tile.h>

namespace TiledArray {

/*
 * btas::Tensor with UM storage cuda_um_btas_varray
 */

template <typename T, typename Range = TiledArray::Range>
using btasUMTensorVarray =
    ::btas::Tensor<T, Range, TiledArray::cuda_um_btas_varray<T>>;

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

template <typename T, typename Range>
struct eval_trait<btasUMTensorVarray<T, Range>> {
  typedef btasUMTensorVarray<T, Range> type;
};

///
/// gemm
///

template <typename T, typename Scalar, typename Range>
btasUMTensorVarray<T, Range> gemm(
    const btasUMTensorVarray<T, Range> &left,
    const btasUMTensorVarray<T, Range> &right, Scalar factor,
    const TiledArray::math::GemmHelper &gemm_helper) {
  return btas_tensor_gemm_cuda_impl(left, right, factor, gemm_helper);
}

template <typename T, typename Scalar, typename Range>
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
  return btas_tensor_clone_cuda_impl(arg);
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

  return result;
}

///
/// scale
///

template <typename T, typename Range, typename Scalar>
btasUMTensorVarray<T, Range> scale(const btasUMTensorVarray<T, Range> &arg,
                                   const Scalar factor) {
  return btas_tensor_scale_cuda_impl(arg, factor);
}

template <typename T, typename Range, typename Scalar>
void scale_to(btasUMTensorVarray<T, Range> &arg, const Scalar factor) {
  btas_tensor_scale_to_cuda_impl(arg, factor);
}

template <typename T, typename Range, typename Scalar>
btasUMTensorVarray<T, Range> scale(const btasUMTensorVarray<T, Range> &arg,
                                   const Scalar factor,
                                   const TiledArray::Permutation &perm) {
  auto result = permute(arg, perm);
  scale_to(result, factor);
  return result;
}

///
/// neg
///

template <typename T, typename Range>
btasUMTensorVarray<T, Range> neg(const btasUMTensorVarray<T, Range> &arg) {
  return btas_tensor_scale_cuda_impl(arg, T(-1.0));
}

template <typename T, typename Range>
btasUMTensorVarray<T, Range> neg(const btasUMTensorVarray<T, Range> &arg,
                                 const TiledArray::Permutation &perm) {
  auto result = permute(arg, perm);
  scale_to(result, T(-1.0));
  return result;
}

template <typename T, typename Range>
void neg_to(btasUMTensorVarray<T, Range> &arg) {
  btas_tensor_scale_to_cuda_impl(arg, T(-1.0));
}

///
/// subt
///

template <typename T, typename Range>
btasUMTensorVarray<T, Range> subt(const btasUMTensorVarray<T, Range> &arg1,
                                  const btasUMTensorVarray<T, Range> &arg2) {
  return btas_tensor_subt_cuda_impl(arg1, arg2, T(1.0));
}

template <typename T, typename Scalar, typename Range>
btasUMTensorVarray<T, Range> subt(const btasUMTensorVarray<T, Range> &arg1,
                                  const btasUMTensorVarray<T, Range> &arg2,
                                  const Scalar factor) {
  auto result = subt(arg1, arg2);
  scale_to(result, factor);
  return result;
}

template <typename T, typename Range>
btasUMTensorVarray<T, Range> subt(const btasUMTensorVarray<T, Range> &arg1,
                                  const btasUMTensorVarray<T, Range> &arg2,
                                  const TiledArray::Permutation &perm) {
  auto result = subt(arg1, arg2);
  return permute(result, perm);
}

template <typename T, typename Scalar, typename Range>
btasUMTensorVarray<T, Range> subt(const btasUMTensorVarray<T, Range> &arg1,
                                  const btasUMTensorVarray<T, Range> &arg2,
                                  const Scalar factor,
                                  const TiledArray::Permutation &perm) {
  auto result = subt(arg1, arg2);
  auto permute_result = permute(result, perm);
  scale_to(permute_result, factor);
  return permute_result;
}

///
/// subt_to
///

template <typename T, typename Range>
void subt_to(btasUMTensorVarray<T, Range> &result,
             const btasUMTensorVarray<T, Range> &arg1) {
  btas_tensor_subt_to_cuda_impl(result, arg1, T(1.0));
}

template <typename T, typename Scalar, typename Range>
void subt_to(btasUMTensorVarray<T, Range> &result,
             const btasUMTensorVarray<T, Range> &arg1, const Scalar factor) {
  subt_to(result, arg1);
  scale_to(result, factor);
}

///
/// add
///

template <typename T, typename Range>
btasUMTensorVarray<T, Range> add(const btasUMTensorVarray<T, Range> &arg1,
                                 const btasUMTensorVarray<T, Range> &arg2) {
  return btas_tensor_add_cuda_impl(arg1, arg2, T(1.0));
}

template <typename T, typename Scalar, typename Range>
btasUMTensorVarray<T, Range> add(const btasUMTensorVarray<T, Range> &arg1,
                                 const btasUMTensorVarray<T, Range> &arg2,
                                 const Scalar factor) {
  auto result = add(arg1, arg2);
  scale_to(result, factor);
  return result;
}

template <typename T, typename Scalar, typename Range>
btasUMTensorVarray<T, Range> add(const btasUMTensorVarray<T, Range> &arg1,
                                 const btasUMTensorVarray<T, Range> &arg2,
                                 const Scalar factor,
                                 const TiledArray::Permutation &perm) {
  auto result = add(arg1, arg2);
  auto perm_result = permute(result, perm);
  scale_to(perm_result, factor);
  return perm_result;
}

template <typename T, typename Range>
btasUMTensorVarray<T, Range> add(const btasUMTensorVarray<T, Range> &arg1,
                                 const btasUMTensorVarray<T, Range> &arg2,
                                 const TiledArray::Permutation &perm) {
  auto result = add(arg1, arg2);
  return permute(result, perm);
}

///
/// add_to
///

template <typename T, typename Range>
void add_to(btasUMTensorVarray<T, Range> &result,
            const btasUMTensorVarray<T, Range> &arg) {
  btas_tensor_add_to_cuda_impl(result, arg, T(1.0));
}

template <typename T, typename Scalar, typename Range>
void add_to(btasUMTensorVarray<T, Range> &result,
            const btasUMTensorVarray<T, Range> &arg, const Scalar factor) {
  add_to(result, arg);
  scale_to(result, factor);
}

///
/// dot
///
template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type dot(
    const btasUMTensorVarray<T, Range> &arg1,
    const btasUMTensorVarray<T, Range> &arg2) {
  return btas_tensor_dot_cuda_impl(arg1, arg2);
}

///
/// mult
///
template <typename T, typename Range>
btasUMTensorVarray<T, Range> mult(const btasUMTensorVarray<T, Range> &arg1,
                                  const btasUMTensorVarray<T, Range> &arg2) {
  return btas_tensor_mult_cuda_impl(arg1, arg2);
}

template <typename T, typename Scalar, typename Range>
btasUMTensorVarray<T, Range> mult(const btasUMTensorVarray<T, Range> &arg1,
                                  const btasUMTensorVarray<T, Range> &arg2,
                                  const Scalar factor) {
  auto result = mult(arg1, arg2);
  scale_to(result, factor);
  return result;
}

template <typename T, typename Range>
btasUMTensorVarray<T, Range> mult(const btasUMTensorVarray<T, Range> &arg1,
                                  const btasUMTensorVarray<T, Range> &arg2,
                                  const TiledArray::Permutation &perm) {
  auto result = mult(arg1, arg2);
  return permute(result, perm);
}

template <typename T, typename Scalar, typename Range>
btasUMTensorVarray<T, Range> mult(const btasUMTensorVarray<T, Range> &arg1,
                                  const btasUMTensorVarray<T, Range> &arg2,
                                  const Scalar factor,
                                  const TiledArray::Permutation &perm) {
  auto result = mult(arg1, arg2, factor);
  return permute(result, perm);
}

///
/// mult to
///
template <typename T, typename Range>
void mult_to(btasUMTensorVarray<T, Range> &result,
             const btasUMTensorVarray<T, Range> &arg) {
  btas_tensor_mult_to_cuda_impl(result, arg);
}

template <typename T, typename Scalar, typename Range>
void mult_to(btasUMTensorVarray<T, Range> &result,
             const btasUMTensorVarray<T, Range> &arg, const Scalar factor) {
  mult_to(result, arg);
  scale_to(result, factor);
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
  return btas_tensor_squared_norm_cuda_impl(arg);
}

///
/// norm
///

template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type norm(
    const btasUMTensorVarray<T, Range> &arg) {
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
  assert(false);
}

///
/// product
///
template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type product(
    const btasUMTensorVarray<T, Range> &arg) {
  assert(false);
}

///
/// max
///
template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type max(
    const btasUMTensorVarray<T, Range> &arg) {
  assert(false);
}

///
/// abs_max
///
template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type abs_max(
    const btasUMTensorVarray<T, Range> &arg) {
  assert(false);
}

///
/// min
///
template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type min(
    const btasUMTensorVarray<T, Range> &arg) {
  assert(false);
}

///
/// min
///
template <typename T, typename Range>
typename btasUMTensorVarray<T, Range>::value_type abs_min(
    const btasUMTensorVarray<T, Range> &arg) {
  assert(false);
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
um_tensor_to_ta_tensor(const TiledArray::DistArray<UMTensor, Policy> &um_array) {
  const auto convert_tile = [](const UMTensor &tile) {
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

  auto ta_array = to_new_tile_type(um_array, convert_tile);

  um_array.world().gop.fence();
  return ta_array;
};

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
typename std::enable_if<
    !std::is_same<UMTensor, TATensor>::value,
    TiledArray::DistArray<UMTensor, Policy>>::type
ta_tensor_to_um_tensor(
    const TiledArray::DistArray<TATensor, Policy> &array) {
  auto convert_tile = [](const TATensor &tile) {
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

    return TiledArray::Tile<Tensor>(result);
  };

  auto um_array = to_new_tile_type(array, convert_tile);

  array.world().gop.fence();
  return um_array;
};

/// no-op if array is the same as return type
template <typename UMTensor, typename TATensor, typename Policy>
typename std::enable_if<
    std::is_same<UMTensor, TATensor>::value,
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
