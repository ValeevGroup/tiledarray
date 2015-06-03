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
 *  untility.h
 *  Jun 1, 2015
 *
 */

#ifndef TILEDARRAY_TENSOR_UTILITY_H__INCLUDED
#define TILEDARRAY_TENSOR_UTILITY_H__INCLUDED

#include <TiledArray/utility.h>
#include <TiledArray/range.h>
#include <TiledArray/block_range.h>
#include <TiledArray/size_array.h>
#include <TiledArray/tensor/type_traits.h>

namespace TiledArray {
  namespace detail {

    /// Get the inner size

    /// \return The largest contiguous, inner-dimension size.
    template <typename T>
    inline typename T::size_type inner_size_helper(const T& tensor) {
      const auto* restrict const stride = data(tensor.range().weight());
      const auto* restrict const size = data(tensor.range().size());

      int i = int(tensor.range().rank()) - 1;
      auto volume = size[i];

      for(--i; i >= 0; --i) {
        const auto stride_i = stride[i];
        const auto size_i = size[i];

        if(volume != stride_i)
          break;
        volume *= size_i;
      }

      return volume;
    }

    template <typename T1, typename T2>
    inline typename T1::size_type inner_size_helper(const T1& tensor1, const T2& tensor2) {
      const auto* restrict const size1   = data(tensor1.range().size());
      const auto* restrict const stride1 = data(tensor1.range().weight());
      const auto* restrict const size2   = data(tensor2.range().size());
      const auto* restrict const stride2 = data(tensor2.range().weight());

      int i = int(tensor1.range().rank()) - 1;
      auto volume1 = size1[i];
      auto volume2 = size2[i];

      for(--i; i >= 0; --i) {
        const auto stride1_i = stride1[i];
        const auto stride2_i = stride2[i];
        const auto size1_i = size1[i];
        const auto size2_i = size2[i];

        if((volume1 != stride1_i) || (volume2 != stride2_i))
          break;
        volume1 *= size1_i;
        volume2 *= size2_i;
      }

      return volume1;
    }

    template <typename T1, typename T2,
        enable_if_t<is_tensor<T1>::value && is_tensor<T2>::value
                 && ! is_contiguous_tensor<T1>::value
                 && is_contiguous_tensor<T2>::value>* = nullptr>
    inline typename T1::size_type inner_size(const T1& tensor1, const T2&) {
      return inner_size_helper(tensor1);
    }


    template <typename T1, typename T2,
        enable_if_t<is_tensor<T1>::value && is_tensor<T2>::value
                 && is_contiguous_tensor<T1>::value
                 && ! is_contiguous_tensor<T2>::value>* = nullptr>
    inline typename T1::size_type inner_size(const T1&, const T2& tensor2) {
      return inner_size_helper(tensor2);
    }

    template <typename T1, typename T2,
        enable_if_t<is_tensor<T1>::value && is_tensor<T2>::value
                 && ! is_contiguous_tensor<T1>::value
                 && ! is_contiguous_tensor<T2>::value>* = nullptr>
    inline typename T1::size_type inner_size(const T1& tensor1, const T2& tensor2) {
      return inner_size_helper(tensor1, tensor2);
    }

    template <typename T,
        enable_if_t<is_tensor<T>::value
                 && ! is_contiguous_tensor<T>::value>* = nullptr>
    inline typename T::size_type inner_size(const T& tensor) {
      return inner_size_helper(tensor);
    }


    template <typename T,
        enable_if_t<is_tensor<T>::value
                 && is_contiguous_tensor<T>::value>* = nullptr>
    inline auto clone_range(const T& tensor) -> decltype(tensor.range())
    { return tensor.range(); }

    template <typename T,
        enable_if_t<is_tensor<T>::value
                 && ! is_contiguous_tensor<T>::value>* = nullptr>
    inline Range clone_range(const T& tensor) {
      const auto rank = tensor.range().rank();
      const auto* const lobound = tensor.range().start();
      const auto* const upbound = tensor.range().finish();
      SizeArray<decltype(*lobound)>
          lower_bound(lobound, lobound + rank);
      SizeArray<decltype(*upbound)>
          upper_bound(upbound, upbound + rank);
      return Range(lower_bound, upper_bound);
    }

    template <typename T1, typename T2,
        enable_if_t<is_tensor<T1>::value && is_tensor<T2>::value
                 && ! (is_shifted<T1>::value || is_shifted<T2>::value)>* = nullptr>
    inline bool is_range_congruent(const T1& tensor1, const T2& tensor2) {
      return tensor1.range() == tensor2.range();
    }

    template <typename T1, typename T2,
        enable_if_t<is_tensor<T1>::value && is_tensor<T2>::value
                 && ! (is_shifted<T1>::value || is_shifted<T2>::value)>* = nullptr>
    inline bool is_range_congruent(const T1& tensor1, const T2& tensor2, const Permutation& perm) {
      return tensor1.range() == (perm * tensor2.range());
    }

    template <typename T1, typename T2,
        enable_if_t<is_tensor<T1>::value && is_tensor<T2>::value
                 && (is_shifted<T1>::value || is_shifted<T2>::value)>* = nullptr>
    inline bool is_range_congruent(const T1& tensor1, const T2& tensor2) {
      const auto rank1 = tensor1.range().rank();
      const auto rank2 = tensor2.range().rank();
      const auto* const extent1 = tensor1.range().size();
      const auto* const extent2 = tensor2.range().size();
      return (rank1 == rank2) && std::equal(extent1, extent1 + rank1, extent2);
    }


  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_UTILITY_H__INCLUDED
