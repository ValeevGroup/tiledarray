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
 *  untility.h
 *  Jun 1, 2015
 *
 */

#ifndef TILEDARRAY_TENSOR_UTILITY_H__INCLUDED
#define TILEDARRAY_TENSOR_UTILITY_H__INCLUDED

#include <TiledArray/block_range.h>
#include <TiledArray/range.h>
#include <TiledArray/size_array.h>
#include <TiledArray/tensor/type_traits.h>
#include <TiledArray/tiled_range1.h>
#include <TiledArray/utility.h>

namespace TiledArray {
namespace detail {

/// Create a copy of the range of the tensor

/// \tparam T The tensor type
/// \param tensor The tensor with the range to be cloned
/// \return A contiguous range with the same lower and upper bounds as the
/// range of \c tensor.
template <typename T, typename std::enable_if<
                          is_contiguous_tensor<T>::value>::type* = nullptr>
inline auto clone_range(const T& tensor) {
  return tensor.range();
}

/// Create a contiguous copy of the range of the tensor

/// \tparam T The tensor type
/// \param tensor The tensor with the range to be cloned
/// \return A contiguous range with the same lower and upper bounds as the
/// range of \c tensor.
template <typename T, typename std::enable_if<
                          !is_contiguous_tensor<T>::value>::type* = nullptr>
inline Range clone_range(const T& tensor) {
  return Range(tensor.range().lobound(), tensor.range().upbound());
}

/// Test that the ranges of a pair of tensors are congruent

/// This function tests that the rank and extent of
/// \c tensor1 are equal to those of \c tensor2.
/// \tparam T1 The first tensor type
/// \tparam T2 The second tensor type
/// \param tensor1 The first tensor to be compared
/// \param tensor2 The second tensor to be compared
/// \return \c true if the rank and extent of the two tensors equal,
/// otherwise \c false.
template <typename T1, typename T2,
          typename std::enable_if<!(is_shifted<T1>::value ||
                                    is_shifted<T2>::value)>::type* = nullptr>
inline bool is_range_congruent(const T1& tensor1, const T2& tensor2) {
  return is_congruent(tensor1.range(), tensor2.range());
}

/// Test that the ranges of a pair of permuted tensors are congruent

/// This function tests that the rank, lower bound, and upper bound of
/// \c tensor1 is equal to that of the permuted range of \c tensor2.
/// \tparam T1 The first tensor type
/// \tparam T2 The second tensor type
/// \param tensor1 The first tensor to be compared
/// \param tensor2 The second tensor to be compared
/// \param perm The permutation to be applied to \c tensor2
/// \return \c true if the rank and extent of the two tensors equal,
/// otherwise \c false.
template <typename T1, typename T2,
          typename std::enable_if<!(is_shifted<T1>::value ||
                                    is_shifted<T2>::value)>::type* = nullptr>
inline bool is_range_congruent(const T1& tensor1, const T2& tensor2,
                               const Permutation& perm) {
  return is_congruent(tensor1.range(), perm * tensor2.range());
}

/// Test that the ranges of a pair of shifted tensors are congruent

/// This function tests that the extents of the two tensors are equal. One
/// or both of the tensors may be shifted.
/// \tparam T1 The first tensor type
/// \tparam T2 The second tensor type
/// \param tensor1 The first tensor to be compared
/// \param tensor2 The second tensor to be compared
/// \return \c true if the rank and extent of the two tensors equal,
/// otherwise \c false.
template <typename T1, typename T2,
          typename std::enable_if<is_shifted<T1>::value ||
                                  is_shifted<T2>::value>::type* = nullptr>
inline bool is_range_congruent(const T1& tensor1, const T2& tensor2) {
  const auto rank1 = tensor1.range().rank();
  const auto rank2 = tensor2.range().rank();
  const auto* const extent1 = tensor1.range().extent_data();
  const auto* const extent2 = tensor2.range().extent_data();
  return (rank1 == rank2) && std::equal(extent1, extent1 + rank1, extent2);
}

/// Test that the ranges of a permuted tensor is congruent with itself

/// This function is used as the termination step for the recursive
/// \c is_range_set_congruent() function, and to handle the case of a single
/// tensor.
/// \tparam T The tensor type
/// \param perm The permutation
/// \param tensor The tensor
/// \return \c true
template <typename T>
inline constexpr bool is_range_set_congruent(const Permutation& perm,
                                             const T& tensor) {
  return true;
}

/// Test that the ranges of a permuted set of tensors are congruent

/// \tparam T1 The first tensor type
/// \tparam T2 The second tensor type
/// \tparam Ts The remaining tensor types
/// \param perm The permutation to be applied to \c tensor2 and \c tensors...
/// \param tensor1 The first tensor to be compared
/// \param tensor2 The second tensor to be compared
/// \param tensors The remaining tensor to be compared in recursive steps
/// \return \c true if all permuted tensors in the list are congruent with
/// the first tensor in the set, otherwise \c false.
template <typename T1, typename T2, typename... Ts>
inline bool is_range_set_congruent(const Permutation& perm, const T1& tensor1,
                                   const T2& tensor2, const Ts&... tensors) {
  return is_range_congruent(tensor1, tensor2, perm) &&
         is_range_set_congruent(perm, tensor1, tensors...);
}

/// Test that the ranges of a tensor is congruent with itself

/// This function is used as the termination step for the recursive
/// \c is_range_set_congruent() function, and to handle the case of a single
/// tensor.
/// \tparam T The tensor type
/// \param tensor The tensor
/// \return \c true
template <typename T>
inline constexpr bool is_range_set_congruent(const T& tensor) {
  return true;
}

/// Test that the ranges of a set of tensors are congruent

/// \tparam T1 The first tensor type
/// \tparam T2 The second tensor type
/// \tparam Ts The remaining tensor types
/// \param tensor1 The first tensor to be compared
/// \param tensor2 The second tensor to be compared
/// \param tensors The remaining tensor to be compared in recursive steps
/// \return \c true if all tensors in the list are congruent with the
/// first tensor in the set, otherwise \c false.
template <typename T1, typename T2, typename... Ts>
inline bool is_range_set_congruent(const T1& tensor1, const T2& tensor2,
                                   const Ts&... tensors) {
  return is_range_congruent(tensor1, tensor2) &&
         is_range_set_congruent(tensor1, tensors...);
}

/// Get the inner size

/// This function searches of the largest contiguous size in the range of a
/// non-contiguous tensor. At a minimum, this is equal to the size of the
/// stride-one dimension.
/// \tparam T A tensor type
/// \param tensor The tensor to be tested
/// \return The largest contiguous, inner-dimension size.
template <typename T>
inline typename T::size_type inner_size_helper(const T& tensor) {
  const auto* MADNESS_RESTRICT const stride = tensor.range().stride_data();
  const auto* MADNESS_RESTRICT const size = tensor.range().extent_data();

  int i = int(tensor.range().rank()) - 1;
  auto volume = size[i];

  for (--i; i >= 0; --i) {
    const auto stride_i = stride[i];
    const auto size_i = size[i];

    if (volume != stride_i) break;
    volume *= size_i;
  }

  return volume;
}

/// Get the inner size of two tensors

/// This function searches of the largest, common contiguous size in the
/// ranges of two non-contiguous tensors. At a minimum, this is equal to the
/// size of the stride-one dimension.
/// \tparam T1 The first tensor type
/// \tparam T2 The secont tensor type
/// \param tensor1 The first tensor to be tested
/// \param tensor2 The second tensor to be tested
/// \return The largest contiguous, inner-dimension size of the two tensors.
template <typename T1, typename T2>
inline typename T1::size_type inner_size_helper(const T1& tensor1,
                                                const T2& tensor2) {
  TA_ASSERT(is_range_congruent(tensor1, tensor2));
  const auto* MADNESS_RESTRICT const size1 = tensor1.range().extent_data();
  const auto* MADNESS_RESTRICT const stride1 = tensor1.range().stride_data();
  const auto* MADNESS_RESTRICT const size2 = tensor2.range().extent_data();
  const auto* MADNESS_RESTRICT const stride2 = tensor2.range().stride_data();

  int i = int(tensor1.range().rank()) - 1;
  auto volume1 = size1[i];
  auto volume2 = size2[i];

  for (--i; i >= 0; --i) {
    const auto stride1_i = stride1[i];
    const auto stride2_i = stride2[i];
    const auto size1_i = size1[i];
    const auto size2_i = size2[i];

    if ((volume1 != stride1_i) || (volume2 != stride2_i)) break;
    volume1 *= size1_i;
    volume2 *= size2_i;
  }

  return volume1;
}

/// Get the inner size of two tensors

/// This function searches of the largest, common contiguous size in the
/// ranges of two non-contiguous tensors. At a minimum, this is equal to the
/// size of the stride-one dimension.
/// \tparam T1 The first tensor type
/// \tparam T2 The secont tensor type
/// \param tensor1 The first tensor to be tested
/// \return The largest contiguous, inner-dimension size.
template <
    typename T1, typename T2,
    typename std::enable_if<!is_contiguous_tensor<T1>::value &&
                            is_contiguous_tensor<T2>::value>::type* = nullptr>
inline typename T1::size_type inner_size(const T1& tensor1, const T2&) {
  return inner_size_helper(tensor1);
}

/// Get the inner size of two tensors

/// This function searches of the largest, common contiguous size in the
/// ranges of two non-contiguous tensors. At a minimum, this is equal to the
/// size of the stride-one dimension.
/// \tparam T1 The first tensor type
/// \tparam T2 The secont tensor type
/// \param tensor2 The second tensor to be tested
/// \return The largest contiguous, inner-dimension size.
template <
    typename T1, typename T2,
    typename std::enable_if<is_contiguous_tensor<T1>::value &&
                            !is_contiguous_tensor<T2>::value>::type* = nullptr>
inline typename T1::size_type inner_size(const T1&, const T2& tensor2) {
  return inner_size_helper(tensor2);
}

/// Get the inner size of two tensors

/// This function searches of the largest, common contiguous size in the
/// ranges of two non-contiguous tensors. At a minimum, this is equal to the
/// size of the stride-one dimension.
/// \tparam T1 The first tensor type
/// \tparam T2 The secont tensor type
/// \param tensor1 The first tensor to be tested
/// \param tensor2 The second tensor to be tested
/// \return The largest common, contiguous inner-dimension size of the two
/// tensors.
template <
    typename T1, typename T2,
    typename std::enable_if<!is_contiguous_tensor<T1>::value &&
                            !is_contiguous_tensor<T2>::value>::type* = nullptr>
inline typename T1::size_type inner_size(const T1& tensor1, const T2& tensor2) {
  return inner_size_helper(tensor1, tensor2);
}

/// Get the inner size

/// This function searches of the largest contiguous size in the range of a
/// non-contiguous tensor. At a minimum, this is equal to the size of the
/// stride-one dimension.
/// \tparam T A tensor type
/// \param tensor The tensor to be tested
/// \return The largest contiguous, inner-dimension size.
template <typename T, typename std::enable_if<
                          !is_contiguous_tensor<T>::value>::type* = nullptr>
inline typename T::size_type inner_size(const T& tensor) {
  return inner_size_helper(tensor);
}

/// Test for empty tensors in an empty list

/// This function is used as the termination step for the recursive empty()
/// function. It also handles the case where there are no tensors in the
/// list.
/// \return \c false
inline constexpr bool empty() { return false; }

/// Test for empty tensors

/// \tparam T1 The first tensor type
/// \tparam Ts The remaining tensor types
/// \param tensor1 The first tensor to test
/// \param tensors The remaining tensors to test
/// \return \c true if one _or_ more tensors are empty
template <typename T1, typename... Ts>
inline bool empty(const T1& tensor1, const Ts&... tensors) {
  return tensor1.empty() || empty(tensors...);
}

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TENSOR_UTILITY_H__INCLUDED
