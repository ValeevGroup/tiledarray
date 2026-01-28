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

#include <TiledArray/config.h>

#include <TiledArray/fwd.h>
#include <TiledArray/type_traits.h>
#include <iterator>
#include <type_traits>

namespace Eigen {

// Forward declarations
template <typename>
class aligned_allocator;

}  // namespace Eigen

namespace TiledArray {

// Forward declarations
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

/// is true type if all `Ts...` are tensors of scalars
template <typename... Ts>
struct is_tensor;
/// is true type if all `Ts...` are tensors of tensors of scalars
template <typename... Ts>
struct is_tensor_of_tensor;
/// is true type if all `Ts...` are _nested_ tensors; a nested tensor is a
/// tensors of scalars or tensors of nested tensors
template <typename... Ts>
struct is_nested_tensor;
/// is true type if `T1`, `T2`, and `Ts...` are tensors of same nested
/// rank, i.e. they are all tensors of scalars or tensors of tensors of scalars,
/// etc. ;
/// \warning the types must be tensors, hence
/// `tensors_have_equal_nested_rank<Scalar1,Scalar2>` is false
template <typename T1, typename T2, typename... Ts>
struct tensors_have_equal_nested_rank;

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

////////////////////////////////////////////////////////////////////////////////

template <>
struct is_nested_tensor<> : public std::false_type {};

template <typename T>
struct is_nested_tensor<T> : is_tensor_helper<T> {};

template <typename T1, typename T2, typename... Ts>
struct is_nested_tensor<T1, T2, Ts...> {
  static constexpr bool value =
      is_tensor_helper<T1>::value && is_nested_tensor<T2, Ts...>::value;
};

/// @tparam Ts a parameter pack
/// @c is_nested_tensor_v<Ts...> is an alias for @c
/// is_nested_tensor<Ts...>::value
template <typename... Ts>
inline constexpr const bool is_nested_tensor_v = is_nested_tensor<Ts...>::value;

////////////////////////////////////////////////////////////////////////////////

template <typename T, typename Enabler = void>
struct is_tensor_of_tensor_helper : public std::false_type {};

template <typename T>
struct is_tensor_of_tensor_helper<
    T, std::enable_if_t<is_tensor_helper<T>::value>> {
  static constexpr bool value =
      is_tensor_helper<detail::remove_cvr_t<typename T::value_type>>::value &&
      !is_tensor_of_tensor_helper<
          detail::remove_cvr_t<typename T::value_type>>::value;
};

////////////////////////////////////////////////////////////////////////////////

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
inline constexpr const bool is_tensor_v = is_tensor<Ts...>::value;

////////////////////////////////////////////////////////////////////////////////

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
inline constexpr const bool is_tensor_of_tensor_v =
    is_tensor_of_tensor<Ts...>::value;

////////////////////////////////////////////////////////////////////////////////
/// It is sometimes desirable to distinguish between DistArrays with
/// tensor-of-scalars tiles vs tensor-of-tensors tiles.

///
/// True if the @tparam Array is a DistArray with tensor-of-scalars tile type.
/// e.g. DistArray<Tensor<double>>;
///
template <typename Array>
concept array_tos =
    is_array_v<Array> && is_tensor_v<typename Array::value_type>;

///
/// True if the @tparam Array is a DistArray with tensor-of-tensors tile type.
/// e.g. DistArray<Tensor<Tensor<double>>>;
///
template <typename Array>
concept array_tot =
    is_array_v<Array> && is_tensor_of_tensor_v<typename Array::value_type>;

////////////////////////////////////////////////////////////////////////////////

template <typename T1, typename T2, typename Enabler = void>
struct tensors_have_equal_nested_rank_helper : std::false_type {};

template <typename T1, typename T2>
struct tensors_have_equal_nested_rank_helper<
    T1, T2, std::enable_if_t<is_nested_tensor_v<T1, T2>>> {
  static constexpr bool value =
      tensors_have_equal_nested_rank_helper<
          detail::remove_cvr_t<typename T1::value_type>,
          detail::remove_cvr_t<typename T2::value_type>>::value ||
      (detail::is_numeric_v<detail::remove_cvr_t<typename T1::value_type>> &&
       detail::is_numeric_v<detail::remove_cvr_t<typename T2::value_type>>);
};

template <typename T1, typename T2>
struct tensors_have_equal_nested_rank<T1, T2>
    : tensors_have_equal_nested_rank_helper<T1, T2> {};

template <typename T1, typename T2, typename T3, typename... Ts>
struct tensors_have_equal_nested_rank<T1, T2, T3, Ts...> {
  static constexpr bool value =
      tensors_have_equal_nested_rank<T1, T2>::value &&
      tensors_have_equal_nested_rank<T2, T3, Ts...>::value;
};

/// @tparam Ts a parameter pack
/// @c tensors_have_equal_nested_rank_v<Ts...> is an alias for @c
/// tensors_have_equal_nested_rank<Ts...>::value
template <typename T1, typename T2, typename... Ts>
constexpr const bool tensors_have_equal_nested_rank_v =
    tensors_have_equal_nested_rank<T1, T2, Ts...>::value;

template <typename>
constexpr size_t nested_rank = 0;

template <typename T, typename... Ts>
constexpr size_t nested_rank<TA::Tensor<T, Ts...>> = 1 + nested_rank<T>;

template <typename T, typename... Ts>
constexpr size_t nested_rank<const TA::Tensor<T, Ts...>> =
    nested_rank<TA::Tensor<T, Ts...>>;

template <typename T, typename P>
constexpr size_t nested_rank<TA::DistArray<T, P>> = nested_rank<T>;

template <typename T, typename P>
constexpr size_t nested_rank<const TA::DistArray<T, P>> =
    nested_rank<TA::DistArray<T, P>>;

template <typename...>
constexpr size_t max_nested_rank = 0;

template <typename T>
constexpr size_t max_nested_rank<T> = nested_rank<T>;

template <typename T, typename U, typename... Us>
constexpr size_t max_nested_rank<T, U, Us...> =
    std::max(nested_rank<T>, std::max(nested_rank<U>, max_nested_rank<Us...>));

////////////////////////////////////////////////////////////////////////////////

template <typename T, typename Enabler = void>
struct is_ta_tensor : public std::false_type {};

template <typename T, typename A>
struct is_ta_tensor<Tensor<T, A>> : public std::true_type {};

template <typename T>
constexpr const bool is_ta_tensor_v = is_ta_tensor<T>::value;

////////////////////////////////////////////////////////////////////////////////

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

////////////////////////////////////////////////////////////////////////////////

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

/// detect device tile types
template <typename T>
struct is_device_tile : public std::false_type {};

template <typename T>
struct is_device_tile<Tile<T>> : public is_device_tile<T> {};

template <typename T, typename Op>
struct is_device_tile<LazyArrayTile<T, Op>>
    : public is_device_tile<typename LazyArrayTile<T, Op>::eval_type> {};

template <typename T>
static constexpr const auto is_device_tile_v = is_device_tile<T>::value;

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

template <typename T>
struct is_permutation<const T> : public is_permutation<T> {};

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
    std::is_same_v<T, TiledArray::BipartitePermutation> ||
    std::is_same_v<T, const TiledArray::BipartitePermutation>;

template <typename T>
static constexpr const auto is_bipartite_permutable_v =
    is_free_function_permute_anyreturn_v<
        const T&, const TiledArray::BipartitePermutation&>;

//
template <typename, typename = void, typename = void>
constexpr bool is_random_access_container_v{};

///
/// - The container concept is weakly tested -- any type that has
///   @c iterator typedef gets picked up.
///
/// - The iterator category must be std::random_access_iterator_tag --
///   random-access-ness is strongly tested.
///
/// Following lines compile, for example:
///
///     @c static_assert(is_random_access_container<std::vector<int>>);
///     @c static_assert(!is_random_access_container<std::list<int>>);
///
template <typename T>
constexpr bool is_random_access_container_v<
    T, std::void_t<typename T::iterator>,
    std::enable_if_t<std::is_same_v<
        typename std::iterator_traits<typename T::iterator>::iterator_category,
        std::random_access_iterator_tag>>>{true};

//
template <typename, typename = void, typename = void>
constexpr bool is_annotation_v{};

///
/// An annotation type (T) is a type that satisfies the following constraints:
///   - is_random_access_container_v<T> is true.
///   - The value type of the container T are strictly ordered. Note that T is a
///     container from the first constraint.
///
template <typename T>
constexpr bool is_annotation_v<
    T, std::void_t<typename T::value_type>,
    std::enable_if_t<is_random_access_container_v<T> &&
                     is_strictly_ordered_v<typename T::value_type>>

    >{true};

namespace {

template <typename Op, typename Lhs, typename Rhs>
using binop_result_t = std::invoke_result_t<Op, Lhs, Rhs>;

template <typename Op, typename Lhs, typename Rhs, typename = void>
constexpr bool is_binop_v{};

template <typename Op, typename Lhs, typename Rhs>
constexpr bool
    is_binop_v<Op, Lhs, Rhs, std::void_t<binop_result_t<Op, Lhs, Rhs>>>{true};

template <typename Op, typename TensorA, typename TensorB,
          typename Allocator = void,
          typename = std::enable_if_t<is_nested_tensor_v<TensorA, TensorB>>>
struct result_tensor_helper {
 private:
  using TensorA_ = std::remove_reference_t<TensorA>;
  using TensorB_ = std::remove_reference_t<TensorB>;
  using value_type_A = typename TensorA_::value_type;
  using value_type_B = typename TensorB_::value_type;
  using allocator_type_A = typename TensorA_::allocator_type;
  using allocator_type_B = typename TensorB_::allocator_type;

 public:
  using numeric_type = binop_result_t<Op, value_type_A, value_type_B>;
  using allocator_type =
      std::conditional_t<std::is_same_v<void, Allocator> &&
                             std::is_same_v<allocator_type_A, allocator_type_B>,
                         allocator_type_A, Allocator>;
  using result_type =
      std::conditional_t<std::is_same_v<void, allocator_type>,
                         TA::Tensor<numeric_type>,
                         TA::Tensor<numeric_type, allocator_type>>;
};

}  // namespace

///
/// The typedef is a complete TA::Tensor<NumericT, AllocatorT> type where
/// - NumericT is determined by Op:
///   - effectively, it is:
///     <tt> std::invoke_result_t<Op, typename TensorA::value_type, typename
///     TensorB::value_type> </tt>
///
/// - AllocatorT is
///   - the default TA::Tensor allocator if @tparam Allocator is void
///   - TensorA::allocator_type if TensorA and TensorB have the same allocator
///   type
///   - the @tparam Allocator otherwise
/// todo: constraint what @tparam Allocator
///
///
template <typename Op, typename TensorA, typename TensorB,
          typename Allocator = void,
          typename = std::enable_if_t<is_nested_tensor_v<TensorA, TensorB>>>
using result_tensor_t =
    typename result_tensor_helper<Op, TensorA, TensorB, Allocator>::result_type;

}  // namespace detail

/// Specifies how coordinates are mapped to ordinal values
/// - RowMajor: stride decreases as mode index increases
/// - ColMajor: stride increases with the mode index
/// - Other: unknown or dynamic order
enum class OrdinalType { RowMajor = -1, ColMajor = 1, Other = 0, Invalid };

namespace detail {

/// ordinal trait specifies properties of the ordinal
template <typename Ordinal, typename Enabler = void>
struct ordinal_traits;

/// TA::Range is hardwired to row-major
template <>
struct ordinal_traits<Range> {
  static constexpr const auto type = OrdinalType::RowMajor;
};

/// ordinal traits of contiguous tensors are defined by their range type
template <typename T>
struct ordinal_traits<T, std::enable_if_t<is_contiguous_tensor_v<T>>> {
  static constexpr const auto type = ordinal_traits<
      std::decay_t<decltype(std::declval<const T&>().range())>>::type;
};

}  // namespace detail

}  // namespace TiledArray

#endif  // TILEDARRAY_TENSOR_TYPE_TRAITS_H__INCLUDED
