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

#include <TiledArray/tensor/tensor.h>
#include <TiledArray/tensor/tensor_interface.h>
#include <TiledArray/tensor/shift_wrapper.h>

namespace TiledArray {

  // Template aliases for TensorInterface objects

  template <typename T>
  using TensorView =
      detail::TensorInterface<T, BlockRange>;

  template <typename T>
  using TensorConstView =
      detail::TensorInterface<typename std::add_const<T>::type, BlockRange>;


  template <typename T>
  using TensorMap =
      detail::TensorInterface<T, Range>;

  template <typename T>
  using TensorConstMap =
      detail::TensorInterface<typename std::add_const<T>::type, Range>;

  template <typename T, typename Index>
  inline TensorMap<T> make_map(T* const data, const Index& lower_bound,
      const Index& upper_bound)
  { return TensorMap<T>(Range(lower_bound, upper_bound), data); }

  template <typename T, typename Index>
  inline TensorMap<T> make_map(T* const data,
      const std::initializer_list<Index>& lower_bound,
      const std::initializer_list<Index>& upper_bound)
  { return TensorMap<T>(Range(lower_bound, upper_bound), data); }

  template <typename T>
  inline TensorMap<T> make_map(T* const data, const Range& range)
  { return TensorMap<T>(range, data); }

  template <typename T, typename Index>
  inline TensorConstMap<T> make_map(const T* const data, const Index& lower_bound,
      const Index& upper_bound)
  { return TensorConstMap<T>(Range(lower_bound, upper_bound), data); }


  template <typename T, typename Index>
  inline TensorConstMap<T> make_map(const T* const data,
      const std::initializer_list<Index>& lower_bound,
      const std::initializer_list<Index>& upper_bound)
  { return TensorConstMap<T>(Range(lower_bound, upper_bound), data); }

  template <typename T>
  inline TensorConstMap<T> make_map(const T* const data, const Range& range)
  { return TensorConstMap<T>(range, data); }


  template <typename T, typename Index>
  inline TensorConstMap<T> make_const_map(const T* const data, const Index& lower_bound,
      const Index& upper_bound)
  { return TensorConstMap<T>(Range(lower_bound, upper_bound), data); }


  template <typename T, typename Index>
  inline TensorConstMap<T> make_const_map(const T* const data,
      const std::initializer_list<Index>& lower_bound,
      const std::initializer_list<Index>& upper_bound)
  { return TensorConstMap<T>(Range(lower_bound, upper_bound), data); }

  template <typename T>
  inline TensorConstMap<T> make_const_map(const T* const data, const Range& range)
  { return TensorConstMap<T>(range, data); }

} // namespace TiledArray

#endif // TILEDARRAY_SRC_TILEDARRAY_TENSOR_H__INCLUDED
