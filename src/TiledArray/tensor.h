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
template <typename T, typename std::enable_if<detail::is_tensor<T>::value &&
                                              detail::is_contiguous_tensor<
                                                  T>::value>::type* = nullptr>
inline std::ostream& operator<<(std::ostream& os, const T& t) {
  os << t.range() << " { ";
  const auto n = t.range().volume();
  std::size_t offset = 0ul;
  const auto more_than_1_batch = t.batch_size() > 1;
  for (auto b = 0ul; b != t.batch_size(); ++b) {
    if (more_than_1_batch) {
      os << "[batch " << b << "]{ ";
    }
    for (auto ord = 0ul; ord < n; ++ord) {
      os << t.data()[offset + ord] << " ";
    }
    if (more_than_1_batch) {
      os << "} ";
    }
    offset += n;
  }
  os << "}";

  return os;
}

/// Tensor output operator

/// Output tensor \c t to the output stream, \c os .
/// \tparam T The tensor type
/// \param os The output stream
/// \param t The tensor to be output
/// \return A reference to the output stream
template <typename T, typename std::enable_if<detail::is_tensor<T>::value &&
                                              !detail::is_contiguous_tensor<
                                                  T>::value>::type* = nullptr>
inline std::ostream& operator<<(std::ostream& os, const T& t) {
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

  os << "}";

  return os;
}

template <typename T,
          typename = std::enable_if_t<detail::is_tensor_of_tensor_v<T>>>
inline std::ostream& operator<<(std::ostream& os, const T& t) {
  os << t.range() << " {" << std::endl;  // Outer tensor's range
  for (auto idx : t.range()) {           // Loop over inner tensors
    const auto& inner_t = t(idx);
    os << "  " << idx << ":" << inner_t << std::endl;
  }
  os << "}";  // End outer tensor
  return os;
}

}  // namespace TiledArray

#endif  // TILEDARRAY_SRC_TILEDARRAY_TENSOR_H__INCLUDED
