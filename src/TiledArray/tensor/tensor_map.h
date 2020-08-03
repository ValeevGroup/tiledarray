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
 *  tensor_map.h
 *  Jun 16, 2015
 *
 */

#ifndef TILEDARRAY_TENSOR_TENSOR_MAP_H__INCLUDED
#define TILEDARRAY_TENSOR_TENSOR_MAP_H__INCLUDED

#include <initializer_list>
#include <type_traits>

#include <TiledArray/range.h>
#include <TiledArray/tensor/type_traits.h>

namespace TiledArray {

namespace detail {

template <typename, typename, typename>
class TensorInterface;

}  // namespace detail

template <typename T, typename Range_ = Range, typename OpResult = Tensor<T>>
using TensorMap = detail::TensorInterface<T, Range_, OpResult>;

template <typename T, typename Range_ = Range, typename OpResult = Tensor<T>>
using TensorConstMap =
    detail::TensorInterface<typename std::add_const<T>::type, Range_, OpResult>;

template <typename T, typename Range_ = Range, typename OpResult = Tensor<T>,
          typename Index>
inline TensorMap<T, Range_, OpResult> make_map(T* const data,
                                               const Index& lower_bound,
                                               const Index& upper_bound) {
  return TensorMap<T, Range_, OpResult>(Range_(lower_bound, upper_bound), data);
}

template <typename T, typename Range_ = Range, typename OpResult = Tensor<T>>
inline TensorMap<T, Range_, OpResult> make_map(
    T* const data, const std::initializer_list<std::size_t>& lower_bound,
    const std::initializer_list<std::size_t>& upper_bound) {
  return TensorMap<T, Range_, OpResult>(Range_(lower_bound, upper_bound), data);
}

template <typename T, typename Range_, typename OpResult = Tensor<T>>
inline TensorMap<T, std::decay_t<Range_>, OpResult> make_map(T* const data,
                                                             Range_&& range) {
  return TensorMap<T, std::decay_t<Range_>>(std::forward<Range_>(range), data);
}

template <typename T, typename Range_ = Range, typename OpResult = Tensor<T>,
          typename Index>
inline TensorConstMap<T, Range_, OpResult> make_map(const T* const data,
                                                    const Index& lower_bound,
                                                    const Index& upper_bound) {
  return TensorConstMap<T, Range_, OpResult>(Range_(lower_bound, upper_bound),
                                             data);
}

template <typename T, typename Range_ = Range, typename OpResult = Tensor<T>>
inline TensorConstMap<T, Range_, OpResult> make_map(
    const T* const data, const std::initializer_list<std::size_t>& lower_bound,
    const std::initializer_list<std::size_t>& upper_bound) {
  return TensorConstMap<T, Range_, OpResult>(Range_(lower_bound, upper_bound),
                                             data);
}

template <typename T, typename Range_, typename OpResult = Tensor<T>>
inline TensorConstMap<T, std::decay_t<Range_>, OpResult> make_map(
    const T* const data, Range_&& range) {
  return TensorConstMap<T, std::decay_t<Range_>, OpResult>(
      std::forward<Range_>(range), data);
}

template <typename T, typename Range_ = Range, typename OpResult = Tensor<T>,
          typename Index>
inline TensorConstMap<T, Range_, OpResult> make_const_map(
    const T* const data, const Index& lower_bound, const Index& upper_bound) {
  return TensorConstMap<T, Range_, OpResult>(Range_(lower_bound, upper_bound),
                                             data);
}

template <typename T, typename Range_ = Range, typename OpResult = Tensor<T>>
inline TensorConstMap<T, Range_, OpResult> make_const_map(
    const T* const data, const std::initializer_list<std::size_t>& lower_bound,
    const std::initializer_list<std::size_t>& upper_bound) {
  return TensorConstMap<T, Range_, OpResult>(Range_(lower_bound, upper_bound),
                                             data);
}

template <typename T, typename Range_, typename OpResult = Tensor<T>>
inline TensorConstMap<T, std::decay_t<Range_>, OpResult> make_const_map(
    const T* const data, Range_&& range) {
  return TensorConstMap<T, std::decay_t<Range_>, OpResult>(
      std::forward<Range_>(range), data);
}

template <typename T, typename Range_, typename OpResult = Tensor<T>,
          typename Index>
inline TensorConstMap<T, Range_, OpResult> make_const_map(
    T* const data, const Index& lower_bound, const Index& upper_bound) {
  return TensorConstMap<T, Range_, OpResult>(Range_(lower_bound, upper_bound),
                                             const_cast<const T*>(data));
}

template <typename T, typename Range_, typename OpResult = Tensor<T>>
inline TensorConstMap<T, Range_, OpResult> make_const_map(
    T* const data, const std::initializer_list<std::size_t>& lower_bound,
    const std::initializer_list<std::size_t>& upper_bound) {
  return TensorConstMap<T, Range_, OpResult>(Range_(lower_bound, upper_bound),
                                             const_cast<const T*>(data));
}

template <typename T, typename Range_, typename OpResult = Tensor<T>>
inline TensorConstMap<T, std::decay_t<Range_>, OpResult> make_const_map(
    T* const data, Range_&& range) {
  return TensorConstMap<T, std::decay_t<Range_>, OpResult>(
      std::forward<Range_>(range), const_cast<const T*>(data));
}

/// For reusing map without allocating new ranges . . . maybe.
template <typename T, typename Range_, typename OpResult, typename Index1,
          typename Index2>
inline void remap(TensorMap<T, Range_, OpResult>& map, T* const data,
                  const Index1& lower_bound, const Index2& upper_bound) {
  map.range_.resize(lower_bound, upper_bound);
  map.data_ = data;
}

template <typename T, typename Range_, typename OpResult, typename Index1,
          typename Index2,
          typename = std::enable_if_t<!std::is_const<T>::value>>
inline void remap(TensorConstMap<T, Range_, OpResult>& map, T* const data,
                  const Index1& lower_bound, const Index2& upper_bound) {
  map.range_.resize(lower_bound, upper_bound);
  map.data_ = const_cast<const T*>(data);
}

template <typename T, typename Range_, typename OpResult, typename Index1,
          typename Index2>
inline void remap(TensorMap<T, Range_, OpResult>& map, T* const data,
                  const std::initializer_list<Index1>& lower_bound,
                  const std::initializer_list<Index2>& upper_bound) {
  map.range_.resize(lower_bound, upper_bound);
  map.data_ = data;
}

template <typename T, typename Range_, typename OpResult, typename Index1,
          typename Index2,
          typename = std::enable_if_t<!std::is_const<T>::value>>
inline void remap(TensorConstMap<T, Range_, OpResult>& map, T* const data,
                  const std::initializer_list<Index1>& lower_bound,
                  const std::initializer_list<Index2>& upper_bound) {
  map.range_.resize(lower_bound, upper_bound);
  map.data_ = const_cast<const T*>(data);
}

}  // namespace TiledArray

#endif  // TILEDARRAY_TENSOR_TENSOR_MAP_H__INCLUDED
