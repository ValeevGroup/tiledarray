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

#include <type_traits>

namespace TiledArray {

  // Forward declarations
  template <typename, typename> class Tensor;
  template <typename> class TensorView;

  namespace detail {

    // Forward declarations
    template <typename> class ShiftWrapper;


    // Type traits for detecting tensors and tensors of tensors.
    // is_tensor_helper tests if individual types are tensors, while is_tensor
    // tests zero or more tensor types. Similarly is_tensor_of_tensor tests if
    // one or more types are tensors of tensors.
    // To extend the definition of tensors and tensors of tensor, add additional
    // is_tensor_helper and is_tensor_of_tensor_helper (partial) specializations.
    // Note: These type traits help differentiate different implementation
    // functions for tensors, so a tensor of tensors is not considered a tensor.

    template <typename...Ts> struct is_tensor;
    template <typename...Ts> struct is_tensor_of_tensor;

    template <typename>
    struct is_tensor_helper : public std::false_type { };

    template <typename T, typename A>
    struct is_tensor_helper<Tensor<T, A> > : public std::true_type { };

    template <typename T>
    struct is_tensor_helper<TensorView<T> > : public std::true_type { };

    template <typename T>
    struct is_tensor_helper<ShiftWrapper<T> > : public is_tensor_helper<T> { };

    template <typename T>
    struct is_tensor_of_tensor_helper : public std::false_type { };

    template <typename T, typename A>
    struct is_tensor_of_tensor_helper<Tensor<T, A> > :
        public is_tensor_helper<T> { };

    template <typename T>
    struct is_tensor_of_tensor_helper<TensorView<T> > :
        public is_tensor_helper<T> { };

    template <typename T>
    struct is_tensor_of_tensor_helper<ShiftWrapper<T> > :
      public is_tensor_of_tensor_helper<T> { };


    template <> struct is_tensor<> : public std::false_type { };

    template <typename T>
    struct is_tensor<T> {
      static constexpr bool value = is_tensor_helper<T>::value
                                 && ! is_tensor_of_tensor_helper<T>::value;
    };

    template <typename T1, typename T2, typename... Ts>
    struct is_tensor<T1, T2, Ts...> {
      static constexpr bool value = is_tensor<T1>::value
                                 && is_tensor<T2, Ts...>::value;
    };


    template <> struct is_tensor_of_tensor<> : public std::false_type { };

    template <typename T>
    struct is_tensor_of_tensor<T> {
      static constexpr bool value = is_tensor_of_tensor_helper<T>::value;
    };

    template <typename T1, typename T2, typename... Ts>
    struct is_tensor_of_tensor<T1, T2, Ts...> {
      static constexpr bool value = is_tensor_of_tensor<T1>::value
                                 && is_tensor_of_tensor<T2, Ts...>::value;
    };


    // Test if the tensor is contiguous

    template <typename T>
    struct is_contiguous_tensor_helper : public std::false_type { };

    template <typename T, typename A>
    struct is_contiguous_tensor_helper<Tensor<T, A> > : public std::true_type { };

    template <typename T>
    struct is_contiguous_tensor_helper<ShiftWrapper<T> > :
        public is_contiguous_tensor_helper<T> { };



    template <typename...Ts> struct is_contiguous_tensor;

    template <> struct is_contiguous_tensor<> : public std::false_type { };

    template <typename T>
    struct is_contiguous_tensor<T> : public is_contiguous_tensor_helper<T> { };

    template <typename T1, typename T2, typename... Ts>
    struct is_contiguous_tensor<T1, T2, Ts...> {
      static constexpr bool value = is_contiguous_tensor_helper<T1>::value
                                 && is_contiguous_tensor<T2, Ts...>::value;
    };

    // Test if the tensor is shifted

    template <typename T>
    struct is_shifted_helper : public std::false_type { };

    template <typename T>
    struct is_shifted_helper<ShiftWrapper<T> > : public std::true_type{ };


    template <typename...Ts> struct is_shifted;

    template <> struct is_shifted<> : public std::false_type { };

    template <typename T>
    struct is_shifted<T> : public is_shifted_helper<T> { };

    template <typename T1, typename T2, typename... Ts>
    struct is_shifted<T1, T2, Ts...> {
      static constexpr bool value = is_shifted_helper<T1>::value
                                 && is_shifted<T2, Ts...>::value;
    };

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_TYPE_TRAITS_H__INCLUDED
