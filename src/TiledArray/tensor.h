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
 *  tensor.h
 *  Jun 16, 2015
 *
 */

#ifndef TILEDARRAY_TENSOR_H__INCLUDED
#define TILEDARRAY_TENSOR_H__INCLUDED

#include <TiledArray/block_range.h>

#include <TiledArray/tensor/tensor.h>

#include <TiledArray/tensor/print.h>
#include <TiledArray/tensor/tensor_interface.h>
#include <TiledArray/tensor/tensor_map.h>

#include <TiledArray/tile_op/tile_interface.h>

#include <TiledArray/tensor/operators.h>
#include <TiledArray/tensor/shift_wrapper.h>

namespace TiledArray {

// Template aliases for TensorInterface objects

template <typename T>
using TensorView = detail::TensorInterface<T, BlockRange>;

template <typename T>
using TensorConstView =
    detail::TensorInterface<typename std::add_const<T>::type, BlockRange>;

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

#endif  // TILEDARRAY_SRC_TILEDARRAY_TENSOR_H__INCLUDED
