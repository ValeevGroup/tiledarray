/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  type_traits.h
 *  May 31, 2015
 *
 */

#ifndef TILEDARRAY_TENSOR_TYPE_TRAITS_H__INCLUDED
#define TILEDARRAY_TENSOR_TYPE_TRAITS_H__INCLUDED

#include <TiledArray/type_traits.h>
#include <type_traits>

namespace Eigen {

// Forward declarations
template <typename>
class aligned_allocator;

}  // namespace Eigen

namespace TiledArray {

// Forward declarations
class Range;
class BlockRange;
template <typename T, typename A = Eigen::aligned_allocator<T>>
class Tensor;
template <typename>
class Tile;

class Permutation;
class BipartitePermutation;

namespace symmetry {
class Permutation;
}

namespace detail {

// Forward declarations
template <typename T, typename R, typename = Tensor<std::remove_const_t<T>>>
class TensorInterface;
template <typename>
class ShiftWrapper;

// Type traits for detecting tensors and tensors of tensors.
// is_tensor_helper tests if individual types are tensors, while is_tensor
// tests a pack of types. Similarly is_tensor_of_tensor tests if
// one or more types are tensors of tensors.
// To extend the definition of tensors and tensors of tensor, add additional
// is_tensor_helper and is_tensor_of_tensor_helper (partial) specializations.
// Note: These type traits help differentiate different implementation
// functions for tensors, so a tensor of tensors is not considered a tensor.

template <typename... Ts>
struct is_tensor;
template <typename... Ts>
struct is_tensor_of_tensor;

template <typename>
struct is_tensor_helper : public std::false_type {};

template <typename T, typename A>
struct is_tensor_helper<Tensor<T, A>> : public std::true_type {};

template <typename... Args>
struct is_tensor_helper<TensorInterface<Args...>> : public std::true_type {};

template <typename T>
struct is_tensor_helper<ShiftWrapper<T>> : public is_tensor_helper<T> {};

template <typename T>
struct is_tensor_helper<ShiftWrapper<const T>> : public is_tensor_helper<T> {};

template <typename T>
struct is_tensor_helper<Tile<T>> : public is_tensor_helper<T> {};

template <typename T>
struct is_tensor_of_tensor_helper : public std::false_type {};

template <typename T, typename A>
struct is_tensor_of_tensor_helper<Tensor<T, A>> : public is_tensor_helper<T> {};

template <typename T, typename... Args>
struct is_tensor_of_tensor_helper<TensorInterface<T, Args...>>
    : public is_tensor_helper<T> {};

template <typename T>
struct is_tensor_of_tensor_helper<ShiftWrapper<T>>
    : public is_tensor_of_tensor_helper<T> {};

template <typename T>
struct is_tensor_of_tensor_helper<Tile<T>>
    : public is_tensor_of_tensor_helper<T> {};

template <>
struct is_tensor<> : public std::false_type {};

template <typename T>
struct is_tensor<T> {
  static constexpr bool value =
      is_tensor_helper<T>::value && !is_tensor_of_tensor_helper<T>::value;
};

template <typename T1, typename T2, typename... Ts>
struct is_tensor<T1, T2, Ts...> {
  static constexpr bool value =
      is_tensor<T1>::value && is_tensor<T2, Ts...>::value;
};

/// @tparam Ts a parameter pack
/// @c is_tensor_v<Ts...> is an alias for @c is_tensor<Ts...>::value
template <typename... Ts>
constexpr const bool is_tensor_v = is_tensor<Ts...>::value;

template <>
struct is_tensor_of_tensor<> : public std::false_type {};

template <typename T>
struct is_tensor_of_tensor<T> {
  static constexpr bool value = is_tensor_of_tensor_helper<T>::value;
};

template <typename T1, typename T2, typename... Ts>
struct is_tensor_of_tensor<T1, T2, Ts...> {
  static constexpr bool value =
      is_tensor_of_tensor<T1>::value && is_tensor_of_tensor<T2, Ts...>::value;
};

/// @tparam Ts a parameter pack
/// @c is_tensor_of_tensor_v<Ts...> is an alias for @c
/// is_tensor_of_tensor<Ts...>::value
template <typename... Ts>
constexpr const bool is_tensor_of_tensor_v = is_tensor_of_tensor<Ts...>::value;

template <typename T, typename Enabler = void>
struct is_ta_tensor : public std::false_type {};

template <typename T, typename A>
struct is_ta_tensor<Tensor<T, A>> : public std::true_type {};

template <typename T>
constexpr const bool is_ta_tensor_v = is_ta_tensor<T>::value;

// Test if the tensor is contiguous

template <typename T>
struct is_contiguous_range_helper : public std::false_type {};

template <>
struct is_contiguous_range_helper<Range> : public std::true_type {};

template <typename T>
struct is_contiguous_tensor_helper : public std::false_type {};

template <typename T, typename A>
struct is_contiguous_tensor_helper<Tensor<T, A>> : public std::true_type {};

template <typename T, typename R, typename OpResult>
struct is_contiguous_tensor_helper<TensorInterface<T, R, OpResult>>
    : public is_contiguous_range_helper<R> {};

template <typename T>
struct is_contiguous_tensor_helper<ShiftWrapper<T>>
    : public is_contiguous_tensor_helper<T> {};

template <typename T>
struct is_contiguous_tensor_helper<Tile<T>>
    : public is_contiguous_tensor_helper<T> {};

template <typename... Ts>
struct is_contiguous_tensor;

template <>
struct is_contiguous_tensor<> : public std::false_type {};

template <typename T>
struct is_contiguous_tensor<T> : public is_contiguous_tensor_helper<T> {};

template <typename T1, typename T2, typename... Ts>
struct is_contiguous_tensor<T1, T2, Ts...> {
  static constexpr bool value = is_contiguous_tensor_helper<T1>::value &&
                                is_contiguous_tensor<T2, Ts...>::value;
};

/// @tparam Ts a parameter pack
/// @c is_contiguous_tensor_v<Ts...> is an alias for @c
/// is_contiguous_tensor<Ts...>::value
template <typename... Ts>
constexpr const bool is_contiguous_tensor_v =
    is_contiguous_tensor<Ts...>::value;

// Test if the tensor is shifted

template <typename T>
struct is_shifted_helper : public std::false_type {};

template <typename T>
struct is_shifted_helper<ShiftWrapper<T>> : public std::true_type {};

template <typename... Ts>
struct is_shifted;

template <>
struct is_shifted<> : public std::false_type {};

template <typename T>
struct is_shifted<T> : public is_shifted_helper<T> {};

template <typename T1, typename T2, typename... Ts>
struct is_shifted<T1, T2, Ts...> {
  static constexpr bool value =
      is_shifted_helper<T1>::value && is_shifted<T2, Ts...>::value;
};

/// @tparam Ts a parameter pack
/// @c is_shifted_v<Ts...> is an alias for @c is_shifted<Ts...>::value
template <typename... Ts>
constexpr const bool is_shifted_v = is_shifted<Ts...>::value;

// check if reduce_op can reduce set of types
template <typename Enabler, typename ReduceOp, typename Result,
          typename... Args>
struct is_reduce_op_ : public std::false_type {};

template <typename ReduceOp, typename Result, typename... Args>
struct is_reduce_op_<
    std::void_t<decltype(std::declval<ReduceOp&>()(
        std::declval<Result&>(), std::declval<const Args*>()...))>,
    ReduceOp, Result, Args...> : public std::true_type {};

template <typename ReduceOp, typename Result, typename... Args>
constexpr const bool is_reduce_op_v =
    is_reduce_op_<void, ReduceOp, Result, Args...>::value;

/// detect cuda tile
#ifdef TILEDARRAY_HAS_CUDA
template <typename T>
struct is_cuda_tile : public std::false_type {};

template <typename T>
struct is_cuda_tile<Tile<T>> : public is_cuda_tile<T> {};

template <typename T, typename Op>
struct is_cuda_tile<LazyArrayTile<T, Op>>
    : public is_cuda_tile<typename LazyArrayTile<T, Op>::eval_type> {};
#endif

template <typename Tensor, typename Enabler = void>
struct default_permutation;

template <typename Tensor>
struct default_permutation<Tensor,
                           std::enable_if_t<!is_tensor_of_tensor_v<Tensor>>> {
  using type = TiledArray::Permutation;
};

template <typename Tensor>
struct default_permutation<Tensor,
                           std::enable_if_t<is_tensor_of_tensor_v<Tensor>>> {
  using type = TiledArray::BipartitePermutation;
};

template <typename Tensor>
using default_permutation_t = typename default_permutation<Tensor>::type;

template <typename T, typename Enabler = void>
struct is_permutation : public std::false_type {};

template <>
struct is_permutation<TiledArray::Permutation> : public std::true_type {};

template <>
struct is_permutation<TiledArray::BipartitePermutation>
    : public std::true_type {};

template <>
struct is_permutation<TiledArray::symmetry::Permutation>
    : public std::true_type {};

template <typename T>
static constexpr const auto is_permutation_v = is_permutation<T>::value;

template <typename T>
static constexpr const auto is_bipartite_permutation_v =
    std::is_same_v<T, TiledArray::BipartitePermutation>;

template <typename T>
static constexpr const auto is_bipartite_permutable_v =
    is_free_function_permute_anyreturn_v<
        const T&, const TiledArray::BipartitePermutation&>;

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TENSOR_TYPE_TRAITS_H__INCLUDED
