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

/// Tensor plus Tensor operator

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

/// Tensor plus number operator

/// Adds a number to a tensor
/// \tparam T1 A tensor type
/// \param tensor The tensor argument
/// \param number The number argument
/// \return A tensor where element \c i is equal to <tt>tensor[i] + number</tt>
template <typename T1, typename = std::enable_if_t<detail::is_nested_tensor_v<
                           detail::remove_cvr_t<T1>>>>
inline decltype(auto) operator+(
    T1&& tensor, detail::numeric_t<detail::remove_cvr_t<T1>> number) {
  return std::forward<T1>(tensor).add(number);
}

/// Number plus Tensor operator

/// Adds a number to a tensor
/// \tparam T1 A tensor type
/// \param number The number argument
/// \param tensor The tensor argument
/// \return A tensor where element \c i is equal to <tt>tensor[i] + number</tt>
template <typename T1, typename = std::enable_if_t<detail::is_nested_tensor_v<
                           detail::remove_cvr_t<T1>>>>
inline decltype(auto) operator+(
    detail::numeric_t<detail::remove_cvr_t<T1>> number, T1&& tensor) {
  return std::forward<T1>(tensor).add(number);
}

/// Tensor minus Tensor operator

/// Subtracts two tensors
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

/// Tensor minus number operator

/// Subtracts a number from a tensor
/// \tparam T1 A tensor type
/// \param tensor The tensor argument
/// \param number The number argument
/// \return A tensor where element \c i is equal to <tt>tensor[i] - number</tt>
template <typename T1, typename = std::enable_if_t<detail::is_nested_tensor_v<
                           detail::remove_cvr_t<T1>>>>
inline decltype(auto) operator-(
    T1&& tensor, detail::numeric_t<detail::remove_cvr_t<T1>> number) {
  return std::forward<T1>(tensor).subt(number);
}

/// Element-wise multiplication operator for Tensors

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

/// Tensor output operator

/// Output tensor \c t to the output stream, \c os .
/// \tparam T The tensor type
/// \param os The output stream
/// \param t The tensor to be output
/// \return A reference to the output stream
template <
    typename Char, typename CharTraits, typename T,
    typename std::enable_if<detail::is_nested_tensor_v<T> &&
                            detail::is_contiguous_tensor_v<T>>::type* = nullptr>
inline std::basic_ostream<Char, CharTraits>& operator<<(
    std::basic_ostream<Char, CharTraits>& os, const T& t) {
  os << t.range() << " {\n";
  const auto n = t.range().volume();
  std::size_t offset = 0ul;
  std::size_t nbatch = 1;
  if constexpr (detail::has_member_function_nbatch_anyreturn_v<T>)
    nbatch = t.nbatch();
  const auto more_than_1_batch = nbatch > 1;
  for (auto b = 0ul; b != nbatch; ++b) {
    if (more_than_1_batch) {
      os << "  [batch " << b << "]{\n";
    }
    if constexpr (detail::is_tensor_v<T>) {  // tensor of scalars
      detail::NDArrayPrinter{}.print(
          t.data() + offset, t.range().rank(), t.range().extent_data(),
          t.range().stride_data(), os, more_than_1_batch ? 4 : 2);
    } else {  // tensor of tensors, need to annotate each element by its index
      for (auto&& idx : t.range()) {  // Loop over inner tensors
        const auto& inner_t = *(t.data() + offset + t.range().ordinal(idx));
        os << "  " << idx << ":";
        detail::NDArrayPrinter{}.print(inner_t.data(), inner_t.range().rank(),
                                       inner_t.range().extent_data(),
                                       inner_t.range().stride_data(), os,
                                       more_than_1_batch ? 6 : 4);
        os << "\n";
      }
    }
    if (more_than_1_batch) {
      os << "\n  }";
      if (b + 1 != nbatch) os << "\n";  // not last batch
    }
    offset += n;
  }
  os << "\n}\n";

  return os;
}

/// Tensor output operator

/// Output tensor \c t to the output stream, \c os .
/// \tparam T The tensor type
/// \param os The output stream
/// \param t The tensor to be output
/// \return A reference to the output stream
template <typename Char, typename CharTraits, typename T,
          typename std::enable_if<
              detail::is_tensor<T>::value &&
              !detail::is_contiguous_tensor<T>::value>::type* = nullptr>
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
