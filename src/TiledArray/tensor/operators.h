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
 *  operators.h
 *  Jun 16, 2015
 *
 */

#ifndef TILEDARRAY_TENSOR_OPERATORS_H__INCLUDED
#define TILEDARRAY_TENSOR_OPERATORS_H__INCLUDED

#include <TiledArray/tensor/type_traits.h>

namespace TiledArray {

// Tensor arithmetic operators

/// Tensor plus operator

/// Add two tensors
/// \tparam T1 The left-hand tensor type
/// \tparam T2 The right-hand tensor type
/// \param left The left-hand tensor argument
/// \param right The right-hand tensor argument
/// \return A tensor where element \c i is equal to <tt>left[i] + right[i]</tt>
template <typename T1, typename T2,
          typename = std::enable_if_t<detail::tensors_have_equal_nested_rank_v<
              detail::remove_cvr_t<T1>, detail::remove_cvr_t<T2>>>>
inline decltype(auto) operator+(T1&& left, T2&& right) {
  return add(std::forward<T1>(left), std::forward<T2>(right));
}

/// Tensor minus operator

/// Subtract two tensors
/// \tparam T1 The left-hand tensor type
/// \tparam T2 The right-hand tensor type
/// \param left The left-hand tensor argument
/// \param right The right-hand tensor argument
/// \return A tensor where element \c i is equal to <tt>left[i] - right[i]</tt>
template <typename T1, typename T2,
          typename = std::enable_if_t<detail::tensors_have_equal_nested_rank_v<
              detail::remove_cvr_t<T1>, detail::remove_cvr_t<T2>>>>
inline decltype(auto) operator-(T1&& left, T2&& right) {
  return subt(std::forward<T1>(left), std::forward<T2>(right));
}

/// Tensor multiplication operator

/// Element-wise multiplication of two tensors
/// \tparam T1 The left-hand tensor type
/// \tparam T2 The right-hand tensor type
/// \param left The left-hand tensor argument
/// \param right The right-hand tensor argument
/// \return A tensor where element \c i is equal to <tt>left[i] * right[i]</tt>
template <
    typename T1, typename T2,
    typename std::enable_if<detail::is_nested_tensor_v<
        detail::remove_cvr_t<T1>, detail::remove_cvr_t<T2>>>::type* = nullptr>
inline decltype(auto) operator*(T1&& left, T2&& right) {
  return mult(std::forward<T1>(left), std::forward<T2>(right));
}

/// Create a copy of \c left that is scaled by \c right

/// Scale a tensor
/// \tparam T The left-hand tensor type
/// \tparam N Numeric type
/// \param left The left-hand tensor argument
/// \param right The right-hand scalar argument
/// \return A tensor where element \c i is equal to <tt> left[i] * right </tt>
template <typename T, typename N,
          typename std::enable_if<
              detail::is_nested_tensor_v<detail::remove_cvr_t<T>> &&
              detail::is_numeric_v<N>>::type* = nullptr>
inline decltype(auto) operator*(T&& left, N right) {
  return scale(std::forward<T>(left), right);
}

/// Create a copy of \c right that is scaled by \c left

/// \tparam N A numeric type
/// \tparam T The right-hand tensor type
/// \param left The left-hand scalar argument
/// \param right The right-hand tensor argument
/// \return A tensor where element \c i is equal to <tt> left * right[i] </tt>
template <
    typename N, typename T,
    typename std::enable_if<
        detail::is_numeric_v<N> &&
        detail::is_nested_tensor_v<detail::remove_cvr_t<T>>>::type* = nullptr>
inline decltype(auto) operator*(N left, T&& right) {
  return scale(std::forward<T>(right), left);
}

/// Create a negated copy of \c arg

/// \tparam T The element type of \c arg
/// \param arg The argument tensor
/// \return A tensor where element \c i is equal to \c -arg[i]
template <typename T, typename std::enable_if<
                          detail::is_tensor<detail::remove_cvr_t<T>>::value ||
                          detail::is_tensor_of_tensor<
                              detail::remove_cvr_t<T>>::value>::type* = nullptr>
inline decltype(auto) operator-(T&& arg) {
  return neg(std::forward<T>(arg));
}

/// Create a permuted copy of \c arg

/// \tparam T The argument tensor type
/// \param perm The permutation to be applied to \c arg
/// \param arg The argument tensor to be permuted
template <typename T, typename std::enable_if<
                          detail::is_tensor<detail::remove_cvr_t<T>>::value ||
                          detail::is_tensor_of_tensor<
                              detail::remove_cvr_t<T>>::value>::type* = nullptr>
inline decltype(auto) operator*(const Permutation& perm, T&& arg) {
  return permute(std::forward<T>(arg), perm);
}

/// Tensor plus operator

/// Add the elements of \c right to that of \c left
/// \tparam T1 The left-hand tensor type
/// \tparam T2 The right-hand tensor type
/// \param left The left-hand tensor argument
/// \param right The right-hand tensor argument
/// \return A tensor where element \c i is equal to <tt>left[i] + right[i]</tt>
template <typename T1, typename T2,
          typename std::enable_if<
              detail::is_tensor<detail::remove_cvr_t<T1>, T2>::value ||
              detail::is_tensor_of_tensor<detail::remove_cvr_t<T1>,
                                          T2>::value>::type* = nullptr>
inline decltype(auto) operator+=(T1&& left, const T2& right) {
  return add_to(std::forward<T1>(left), right);
}

/// Tensor minus operator

/// Subtract the elements of \c right from that of \c left
/// \tparam T1 The left-hand tensor type
/// \tparam T2 The right-hand tensor type
/// \param left The left-hand tensor argument
/// \param right The right-hand tensor argument
/// \return A reference to \c left
template <typename T1, typename T2,
          typename std::enable_if<
              detail::is_tensor<detail::remove_cvr_t<T1>, T2>::value ||
              detail::is_tensor_of_tensor<detail::remove_cvr_t<T1>,
                                          T2>::value>::type* = nullptr>
inline decltype(auto) operator-=(T1&& left, const T2& right) {
  return subt_to(std::forward<T1>(left), right);
}

/// In place tensor multiplication

/// Multiply the elements of left by that of right
/// \tparam T1 The left-hand tensor type
/// \tparam T2 The right-hand tensor type
/// \param left The left-hand tensor argument
/// \param right The right-hand tensor argument
/// \return A reference to \c left
template <typename T1, typename T2,
          typename std::enable_if<
              detail::is_tensor<detail::remove_cvr_t<T1>, T2>::value ||
              detail::is_tensor_of_tensor<detail::remove_cvr_t<T1>,
                                          T2>::value>::type* = nullptr>
inline decltype(auto) operator*=(T1&& left, const T2& right) {
  return mult_to(std::forward<T1>(left), right);
}

/// In place tensor add constant

/// Scale the elements of \c left by \c right
/// \tparam T The left-hand tensor type
/// \tparam N Numeric type
/// \param left The left-hand tensor argument
/// \param right The right-hand scalar argument
/// \return A reference to \c left
template <typename T, typename N,
          typename std::enable_if<
              (detail::is_tensor<detail::remove_cvr_t<T>>::value ||
               detail::is_tensor_of_tensor<detail::remove_cvr_t<T>>::value) &&
              detail::is_numeric_v<N>>::type* = nullptr>
inline decltype(auto) operator+=(T&& left, N right) {
  return add_to(std::forward<T>(left), right);
}

/// In place tensor subtract constant

/// Scale the elements of \c left by \c right
/// \tparam T The left-hand tensor type
/// \tparam N Numeric type
/// \param left The left-hand tensor argument
/// \param right The right-hand scalar argument
/// \return A reference to \c left
template <typename T, typename N,
          typename std::enable_if<
              (detail::is_tensor<detail::remove_cvr_t<T>>::value ||
               detail::is_tensor_of_tensor<detail::remove_cvr_t<T>>::value) &&
              detail::is_numeric_v<N>>::type* = nullptr>
inline decltype(auto) operator-=(T&& left, N right) {
  return subt_to(std::forward<T>(left), right);
}

/// In place tensor scale

/// Scale the elements of \c left by \c right
/// \tparam T The left-hand tensor type
/// \tparam N Numeric type
/// \param left The left-hand tensor argument
/// \param right The right-hand scalar argument
/// \return A reference to \c left
template <typename T, typename N,
          typename std::enable_if<
              (detail::is_tensor<detail::remove_cvr_t<T>>::value ||
               detail::is_tensor_of_tensor<detail::remove_cvr_t<T>>::value) &&
              detail::is_numeric_v<N>>::type* = nullptr>
inline decltype(auto) operator*=(T&& left, N right) {
  return scale_to(std::forward<T>(left), right);
}

}  // namespace TiledArray

#endif  // TILEDARRAY_TENSOR_OPERATORS_H__INCLUDED
