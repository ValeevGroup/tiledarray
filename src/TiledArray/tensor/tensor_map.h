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

namespace TiledArray {

  class Range;

  namespace detail {

    template <typename, typename> class TensorInterface;

  }  // namespace detail

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

  template <typename T>
  inline TensorMap<T> make_map(T* const data,
      const std::initializer_list<std::size_t>& lower_bound,
      const std::initializer_list<std::size_t>& upper_bound)
  { return TensorMap<T>(Range(lower_bound, upper_bound), data); }

  template <typename T>
  inline TensorMap<T> make_map(T* const data, const Range& range)
  { return TensorMap<T>(range, data); }

  template <typename T, typename Index>
  inline TensorConstMap<T> make_map(const T* const data, const Index& lower_bound,
      const Index& upper_bound)
  { return TensorConstMap<T>(Range(lower_bound, upper_bound), data); }


  template <typename T>
  inline TensorConstMap<T> make_map(const T* const data,
      const std::initializer_list<std::size_t>& lower_bound,
      const std::initializer_list<std::size_t>& upper_bound)
  { return TensorConstMap<T>(Range(lower_bound, upper_bound), data); }

  template <typename T>
  inline TensorConstMap<T> make_map(const T* const data, const Range& range)
  { return TensorConstMap<T>(range, data); }


  template <typename T, typename Index>
  inline TensorConstMap<T> make_const_map(const T* const data, const Index& lower_bound,
      const Index& upper_bound)
  { return TensorConstMap<T>(Range(lower_bound, upper_bound), data); }


  template <typename T>
  inline TensorConstMap<T> make_const_map(const T* const data,
      const std::initializer_list<std::size_t>& lower_bound,
      const std::initializer_list<std::size_t>& upper_bound)
  { return TensorConstMap<T>(Range(lower_bound, upper_bound), data); }

  template <typename T>
  inline TensorConstMap<T> make_const_map(const T* const data, const Range& range)
  { return TensorConstMap<T>(range, data); }

  template <typename T, typename Index>
  inline TensorConstMap<T> make_const_map(T* const data, const Index& lower_bound,
      const Index& upper_bound)
  { return TensorConstMap<T>(Range(lower_bound, upper_bound), const_cast<const T*>(data)); }


  template <typename T>
  inline TensorConstMap<T> make_const_map(T* const data,
      const std::initializer_list<std::size_t>& lower_bound,
      const std::initializer_list<std::size_t>& upper_bound)
  { return TensorConstMap<T>(Range(lower_bound, upper_bound), const_cast<const T*>(data)); }

  template <typename T>
  inline TensorConstMap<T> make_const_map(T* const data, const Range& range)
  { return TensorConstMap<T>(range, const_cast<const T*>(data)); }

  /// For reusing map without allocating new ranges . . . maybe. 
  template <typename T, typename Index>
  void remap(TensorMap<T> &map, T* data, const Index &lower_bound, 
          const Index &upper_bound)
  {
      map.range_.resize(lower_bound, upper_bound);
      map.data_ = data;
  }

  template <typename T, typename Index>
  void remap(TensorConstMap<T>& map, T* data, const Index& lower_bound,
             const Index& upper_bound) {
    map.range_.resize(lower_bound, upper_bound);
    map.data_ = const_cast<const T*>(data);
   }

  template <typename T>
  void remap(TensorMap<T> &map, T* const data,
          const std::initializer_list<std::size_t> &lower_bound,
          const std::initializer_list<std::size_t> &upper_bound)
  {
      map.range_.resize(lower_bound, upper_bound);
      map.data_ = data;
  }

  template <typename T>
  void remap(TensorConstMap<T>& map, T* const data,
             const std::initializer_list<std::size_t>& lower_bound,
             const std::initializer_list<std::size_t>& upper_bound) {
    map.range_.resize(lower_bound, upper_bound);
    map.data_ = const_cast<const T*>(data);
   }

} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_TENSOR_MAP_H__INCLUDED
