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

namespace TiledArray {

  // Forward declarations
  template <typename, typename> class Tensor;
  template <typename> class TensorView;

  namespace detail {

    // Forward declarations
    template <typename> class ShiftWrapper;


    // Test if the object is a tensor

    template <typename>
    struct is_tensor : public std::false_type { };

    template <typename T, typename A>
    struct is_tensor<Tensor<T, A> > : public std::true_type { };

    template <typename T>
    struct is_tensor<TensorView<T> > : public std::true_type { };

    template <typename T>
    struct is_tensor<ShiftWrapper<T> > : public is_tensor<T> { };

    // Test if the tensor is contiguous

    template <typename T>
    struct is_contiguous_tensor : public std::false_type { };

    template <typename T, typename A>
    struct is_contiguous_tensor<Tensor<T, A> > : public std::true_type { };

    template <typename T>
    struct is_contiguous_tensor<ShiftWrapper<T> > :
        public is_contiguous_tensor<T> { };


    // Test if the tensor is shifted

    template <typename T>
    struct is_shifted : public std::false_type { };

    template <typename T>
    struct is_shifted<ShiftWrapper<T> > : public std::true_type{ };

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_TYPE_TRAITS_H__INCLUDED
