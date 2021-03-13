/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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
 */

#ifndef TILEDARRAY_FWD_H__INCLUDED
#define TILEDARRAY_FWD_H__INCLUDED

#include <complex>

#include <btas/fwd.h>

#include <TiledArray/config.h>

namespace Eigen {  // fwd define Eigen's aligned allocator for
                   // TiledArray::Tensor
template <class>
class aligned_allocator;
}  // namespace Eigen

namespace TiledArray {

// Ranges
class Range;
class TiledRange1;
class TiledRange;

// TiledArray Policy
class DensePolicy;
class SparsePolicy;

// TiledArray Tensors
template <typename, typename>
class Tensor;

typedef Tensor<double, Eigen::aligned_allocator<double>> TensorD;
typedef Tensor<int, Eigen::aligned_allocator<int>> TensorI;
typedef Tensor<float, Eigen::aligned_allocator<float>> TensorF;
typedef Tensor<long, Eigen::aligned_allocator<long>> TensorL;
typedef Tensor<std::complex<double>,
               Eigen::aligned_allocator<std::complex<double>>>
    TensorZ;
typedef Tensor<std::complex<float>,
               Eigen::aligned_allocator<std::complex<float>>>
    TensorC;

// CUDA tensor
#ifdef TILEDARRAY_HAS_CUDA

template <class T>
class cuda_um_allocator_impl;

template <typename T, typename A = std::allocator<T>>
class default_init_allocator;

template <typename T>
using cuda_um_allocator = default_init_allocator<T, cuda_um_allocator_impl<T>>;

/// \brief a vector that lives in CUDA Unified Memory, with most operations
/// implemented on the CPU
template <typename T>
using cuda_um_btas_varray = ::btas::varray<T, TiledArray::cuda_um_allocator<T>>;

/**
 * btas::Tensor with UM storage cuda_um_btas_varray
 */
template <typename T, typename Range = TiledArray::Range>
using btasUMTensorVarray =
    ::btas::Tensor<T, Range, TiledArray::cuda_um_btas_varray<T>>;

#endif

// TiledArray Arrays
template <typename, typename>
class DistArray;

// Dense Array Typedefs
template <typename T>
using TArray = DistArray<Tensor<T, Eigen::aligned_allocator<T>>, DensePolicy>;
typedef TArray<double> TArrayD;
typedef TArray<int> TArrayI;
typedef TArray<float> TArrayF;
typedef TArray<long> TArrayL;
typedef TArray<std::complex<double>> TArrayZ;
typedef TArray<std::complex<float>> TArrayC;

// Sparse Array Typedefs
template <typename T>
using TSpArray =
    DistArray<Tensor<T, Eigen::aligned_allocator<T>>, SparsePolicy>;
typedef TSpArray<double> TSpArrayD;
typedef TSpArray<int> TSpArrayI;
typedef TSpArray<float> TSpArrayF;
typedef TSpArray<long> TSpArrayL;
typedef TSpArray<std::complex<double>> TSpArrayZ;
typedef TSpArray<std::complex<float>> TSpArrayC;

// type alias for backward compatibility: the old Array has static type,
// DistArray is rank-polymorphic
template <typename T, unsigned int = 0,
          typename Tile = Tensor<T, Eigen::aligned_allocator<T>>,
          typename Policy = DensePolicy>
using Array = DistArray<Tile, Policy>;

// vector pack of DistArray's
template <typename, typename>
class DistArrayVector;

}  // namespace TiledArray

#ifndef TILEDARRAY_DISABLE_NAMESPACE_TA
namespace TA = TiledArray;
#endif  // TILEDARRAY_DISABLE_NAMESPACE_TA

#endif  // TILEDARRAY_FWD_H__INCLUDED
