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

#include <TiledArray/tensor/print.h>
#include <TiledArray/tensor/type_traits.h>

namespace TiledArray {

// Tensor arithmetic operators
//
// The element-wise tensor+tensor, tensor-tensor, tensor*tensor, and unary
// negation operators live in @c operators_body.ipp so they can be re-injected
// into @c namespace btas via the same file (see @c external/btas.h). Operators
// that involve TiledArray-specific types (Permutation, ostream, scalar mixing,
// in-place compounds) remain below. The @c detail::ta_ops_match_tensor
// predicate they use is declared in <TiledArray/tensor/type_traits.h>.
#include <TiledArray/tensor/operators_body.ipp>

/// Tensor plus number operator

/// Adds a number to a tensor
/// \tparam T1 A tensor type
/// \param tensor The tensor argument
/// \param number The number argument
/// \return A tensor where element \c i is equal to <tt>tensor[i] + number</tt>
template <typename T1,
          typename = std::enable_if_t<
              TA::detail::is_nested_tensor_v<TA::detail::remove_cvr_t<T1>> &&
              !TA::is_tensor_view_v<TA::detail::remove_cvr_t<T1>>>>
inline decltype(auto) operator+(
    T1&& tensor, TA::detail::numeric_t<TA::detail::remove_cvr_t<T1>> number) {
  return std::forward<T1>(tensor).add(number);
}

/// Number plus Tensor operator

/// Adds a number to a tensor
/// \tparam T1 A tensor type
/// \param number The number argument
/// \param tensor The tensor argument
/// \return A tensor where element \c i is equal to <tt>tensor[i] + number</tt>
template <typename T1,
          typename = std::enable_if_t<
              TA::detail::is_nested_tensor_v<TA::detail::remove_cvr_t<T1>> &&
              !TA::is_tensor_view_v<TA::detail::remove_cvr_t<T1>>>>
inline decltype(auto) operator+(
    TA::detail::numeric_t<TA::detail::remove_cvr_t<T1>> number, T1&& tensor) {
  return std::forward<T1>(tensor).add(number);
}

/// Tensor minus number operator

/// Subtracts a number from a tensor
/// \tparam T1 A tensor type
/// \param tensor The tensor argument
/// \param number The number argument
/// \return A tensor where element \c i is equal to <tt>tensor[i] - number</tt>
template <typename T1,
          typename = std::enable_if_t<
              TA::detail::is_nested_tensor_v<TA::detail::remove_cvr_t<T1>> &&
              !TA::is_tensor_view_v<TA::detail::remove_cvr_t<T1>>>>
inline decltype(auto) operator-(
    T1&& tensor, TA::detail::numeric_t<TA::detail::remove_cvr_t<T1>> number) {
  return std::forward<T1>(tensor).subt(number);
}

/// In place tensor add constant

/// Scale the elements of \c left by \c right
/// \tparam T The left-hand tensor type
/// \tparam N Numeric type
/// \param left The left-hand tensor argument
/// \param right The right-hand scalar argument
/// \return A reference to \c left
template <
    typename T, typename N,
    typename std::enable_if<
        (TA::detail::is_tensor<TA::detail::remove_cvr_t<T>>::value ||
         TA::detail::is_tensor_of_tensor<TA::detail::remove_cvr_t<T>>::value) &&
        TA::detail::is_numeric_v<N>>::type* = nullptr>
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
template <
    typename T, typename N,
    typename std::enable_if<
        (TA::detail::is_tensor<TA::detail::remove_cvr_t<T>>::value ||
         TA::detail::is_tensor_of_tensor<TA::detail::remove_cvr_t<T>>::value) &&
        TA::detail::is_numeric_v<N>>::type* = nullptr>
inline decltype(auto) operator-=(T&& left, N right) {
  return subt_to(std::forward<T>(left), right);
}

/// Tensor output operator (non-contiguous TensorInterface-style views; the
/// contiguous-tensor case is provided by @c operators_body.ipp above for both
/// TiledArray::Tensor and btas::Tensor)

/// Output tensor \c t to the output stream, \c os .
/// \tparam T The tensor type
/// \param os The output stream
/// \param t The tensor to be output
/// \return A reference to the output stream
template <typename Char, typename CharTraits, typename T,
          typename std::enable_if<
              TA::detail::is_tensor<T>::value &&
              !TA::detail::is_contiguous_tensor<T>::value>::type* = nullptr>
inline std::basic_ostream<Char, CharTraits>& operator<<(
    std::basic_ostream<Char, CharTraits>& os, const T& t) {
  const auto stride = inner_size(t);
  const auto volume = t.range().volume();

  auto tensor_print_range =
      [&os, stride](typename T::const_pointer MADNESS_RESTRICT const t_data) {
        for (decltype(t.range().volume()) i = 0ul; i < stride; ++i)
          os << t_data[i] << " ";
      };

  os << t.range() << " { ";

  for (decltype(t.range().volume()) i = 0ul; i < volume; i += stride)
    tensor_print_range(t.data() + t.range().ordinal(i));

  os << "}\n";

  return os;
}

}  // namespace TiledArray

#endif  // TILEDARRAY_TENSOR_OPERATORS_H__INCLUDED
