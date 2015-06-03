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
 *  binary.h
 *  Jun 1, 2015
 *
 */

#ifndef TILEDARRAY_TENSOR_UNARY_H__INCLUDED
#define TILEDARRAY_TENSOR_UNARY_H__INCLUDED

#include <TiledArray/tensor/utility.h>
#include <TiledArray/tensor/permute.h>

namespace TiledArray {

  template <typename, typename> class Tensor;
  template <typename> class TensorView;
  template <typename> class ShiftWrapper;

  namespace detail {

    // Unary implementation functions for tensors with contiguous memory layout

    template <typename T, typename Op,
        enable_if_t<is_tensor<T>::value
                 && is_contiguous_tensor<T>::value>* = nullptr>
    inline Tensor<typename T::value_type,
        Eigen::aligned_allocator<typename T::value_type> >
    unary(const T& tensor, Op&& op) {
      TA_ASSERT(! tensor.empty());

      typedef Tensor<typename T::value_type,
          Eigen::aligned_allocator<typename T::value_type> > result_type;

      result_type result(clone_range(tensor));
      const auto volume = tensor.range().volume();

      math::vector_op(std::forward<Op>(op), volume, result.data(),
          tensor.data());

      return result;
    }

    template <typename T, typename InputOp, typename OutputOp,
        enable_if_t<is_tensor<T>::value
                 && is_contiguous_tensor<T>::value>* = nullptr>
    inline Tensor<typename T::value_type,
        Eigen::aligned_allocator<typename T::value_type> >
    unary(const T& tensor, const Permutation& perm, InputOp&& input_op,
        OutputOp&& output_op)
    {
      TA_ASSERT(! tensor.empty());

      typedef Tensor<typename T::value_type,
          Eigen::aligned_allocator<typename T::value_type> > result_type;

      result_type result(perm * clone_range(tensor));

      permute(std::forward<InputOp>(input_op), std::forward<OutputOp>(output_op),
          result, perm, tensor);

      return result;
    }


    template <typename T, typename Op,
        enable_if_t<is_tensor<T>::value
                 && is_contiguous_tensor<T>::value>* = nullptr>
    inline void inplace_unary(T& tensor, Op&& op) {
      TA_ASSERT(! tensor.empty());

      const auto volume = tensor.range().volume();

      math::vector_op(std::forward<Op>(op), volume, tensor.data());
    }



    // Unary implementation functions for tensors with non-contiguous memory layout

    template <typename T, typename Op,
        enable_if_t<is_tensor<T>::value
                 && ! is_contiguous_tensor<T>::value>* = nullptr>
    inline Tensor<typename T::value_type,
        Eigen::aligned_allocator<typename T::value_type> >
    unary(const T& tensor, Op&& op) {
      TA_ASSERT(! tensor.empty());

      typedef Tensor<typename T::value_type,
          Eigen::aligned_allocator<typename T::value_type> > result_type;

      result_type result(clone_range(tensor));
      const auto stride = inner_size(tensor);
      const auto volume = tensor.range().volume();

      for(decltype(volume) i = 0ul; i < volume; i += stride)
        math::vector_op(std::forward<Op>(op), stride, result.data() + i,
          tensor.data() + tensor.range().ord(i));

      return result;
    }

    template <typename T, typename Op,
        enable_if_t<is_tensor<T>::value
                 && ! is_contiguous_tensor<T>::value>* = nullptr>
    inline void inplace_binary(T& tensor, Op&& op) {
      TA_ASSERT(! tensor.empty());

      typedef Tensor<typename T::value_type,
          Eigen::aligned_allocator<typename T::value_type> > result_type;

      const auto stride = inner_size(tensor);
      const typename T::size_type volume = tensor.range().volume();

      for(typename T::size_type i = 0ul; i < volume; i += stride)
        math::vector_op(std::forward<Op>(op), stride,
          tensor.data() + tensor.range().ord(i));
    }

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_UNARY_H__INCLUDED
