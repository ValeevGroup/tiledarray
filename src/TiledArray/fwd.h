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

#include <TiledArray/config.h>

#include <btas/fwd.h>
#include <complex>

// uncomment to import fwddecl for std::allocator
// #include <boost/container/detail/std_fwd.hpp>

// fwddecl Eigen::aligned_allocator
namespace Eigen {
template <class>
class aligned_allocator;
}  // namespace Eigen

// fwddecl host_allocator
namespace umpire {
namespace detail {
struct get_host_allocator;
struct NullLock;
template <typename Tag = void>
class MutexLock;
}  // namespace detail

template <class T, class StaticLock, typename UmpireAllocatorAccessor>
class allocator;

template <typename T, typename A = std::allocator<T>>
class default_init_allocator;

};  // namespace umpire

// fwddecl host_allocator
namespace TiledArray {
namespace detail {
struct get_host_allocator;
}  // namespace detail

namespace host {
class Env;
}
using hostEnv = host::Env;

/// pooled thread-safe host memory allocator
template <typename T>
using host_allocator = umpire::default_init_allocator<
    T, umpire::allocator<T, umpire::detail::MutexLock<hostEnv>,
                         detail::get_host_allocator>>;
}  // namespace TiledArray

namespace madness {
class World;
}

namespace TiledArray {

using madness::World;
World& get_default_world();

// Ranges
class Range;
class TiledRange1;
class TiledRange;
class BlockRange;

// TiledArray Policy
class DensePolicy;
class SparsePolicy;

// TiledArray Tensors
// can any standard-compliant allocator such as std::allocator<T>
template <typename T, typename A =
#ifndef TA_TENSOR_MEM_PROFILE
                          Eigen::aligned_allocator<T>
#else
                          host_allocator<T>
#endif
          >
class Tensor;

typedef Tensor<double> TensorD;
typedef Tensor<int> TensorI;
typedef Tensor<float> TensorF;
typedef Tensor<long> TensorL;
typedef Tensor<std::complex<double>> TensorZ;
typedef Tensor<std::complex<float>> TensorC;

#ifdef TILEDARRAY_HAS_DEVICE
namespace device {
class Env;
}
using deviceEnv = device::Env;

namespace detail {
struct get_um_allocator;
struct get_pinned_allocator;
}  // namespace detail

/// pooled thread-safe unified memory (UM) allocator for device computing
template <typename T>
using device_um_allocator = umpire::default_init_allocator<
    T, umpire::allocator<T, umpire::detail::MutexLock<deviceEnv>,
                         detail::get_um_allocator>>;

/// pooled thread-safe pinned host memory allocator for device computing
template <typename T>
using device_pinned_allocator = umpire::default_init_allocator<
    T, umpire::allocator<T, umpire::detail::MutexLock<deviceEnv>,
                         detail::get_pinned_allocator>>;

/// \brief a vector that lives in UM, with most operations
/// implemented on the CPU
template <typename T>
using device_um_btas_varray =
    ::btas::varray<T, TiledArray::device_um_allocator<T>>;

/**
 * btas::Tensor with UM storage device_um_btas_varray
 */
template <typename T, typename Range = TiledArray::Range>
using btasUMTensorVarray =
    ::btas::Tensor<T, Range, TiledArray::device_um_btas_varray<T>>;

#endif  // TILEDARRAY_HAS_DEVICE

template <typename>
class Tile;

class Permutation;
class BipartitePermutation;

namespace symmetry {
class Permutation;
}

// shapes
class DenseShape;
template <typename T = float>
class SparseShape;

// TiledArray Arrays
template <typename, typename>
class DistArray;

/// Type trait to detect dense shape types
template <typename S>
struct is_dense : public std::false_type {};

template <>
struct is_dense<DenseShape> : public std::true_type {};

template <>
struct is_dense<DensePolicy> : public std::true_type {};

template <typename Tile, typename Policy>
struct is_dense<DistArray<Tile, Policy>>
    : public is_dense<typename DistArray<Tile, Policy>::shape_type> {};

template <typename T>
constexpr const bool is_dense_v = is_dense<T>::value;

// Dense Array Typedefs
template <typename T>
using TArray = DistArray<Tensor<T>, DensePolicy>;
typedef TArray<double> TArrayD;
typedef TArray<int> TArrayI;
typedef TArray<float> TArrayF;
typedef TArray<long> TArrayL;
typedef TArray<std::complex<double>> TArrayZ;
typedef TArray<std::complex<float>> TArrayC;

// Sparse Array Typedefs
template <typename T>
using TSpArray = DistArray<Tensor<T>, SparsePolicy>;
typedef TSpArray<double> TSpArrayD;
typedef TSpArray<int> TSpArrayI;
typedef TSpArray<float> TSpArrayF;
typedef TSpArray<long> TSpArrayL;
typedef TSpArray<std::complex<double>> TSpArrayZ;
typedef TSpArray<std::complex<float>> TSpArrayC;

// type alias for backward compatibility: the old Array has static type,
// DistArray is rank-polymorphic
template <typename T, unsigned int = 0, typename Tile = Tensor<T>,
          typename Policy = DensePolicy>
using Array
    [[deprecated("use TiledArray::DistArray or TiledArray::TArray<T>")]] =
        DistArray<Tile, Policy>;

enum class HostExecutor { Thread, MADWorld, Default = MADWorld };

/// fence types
enum class Fence {
  Global,  //!< global fence (`world.gop.fence()`)
  Local,   //!< local fence (all local work done, equivalent to
           //!< `world.taskq.fence() in absence of active messages)
  No       //!< no fence
};

namespace conversions {

/// user defined conversions

/// must define
/// \code
///  To operator()(From&& from);
/// \endcode
template <typename To, typename From>
struct to;

}  // namespace conversions

/// used to indicate that block tensor expression should preserve the underlying
/// tensor's trange lobound
struct preserve_lobound_t {};

/// used to tag block tensor expression methods that preserve the underlying
/// tensor's trange lobound
inline constexpr preserve_lobound_t preserve_lobound;

}  // namespace TiledArray

#ifndef TILEDARRAY_DISABLE_NAMESPACE_TA
namespace TA = TiledArray;
#endif  // TILEDARRAY_DISABLE_NAMESPACE_TA

#endif  // TILEDARRAY_FWD_H__INCLUDED
