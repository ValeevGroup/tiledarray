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

#ifndef TILEDARRAY_TENSOR_BINARY_H__INCLUDED
#define TILEDARRAY_TENSOR_BINARY_H__INCLUDED

#include <TiledArray/tensor/utility.h>
#include <TiledArray/tensor/permute.h>
#include <TiledArray/math/eigen.h>

namespace TiledArray {

  template <typename, typename> class Tensor;

  namespace detail {

    // Binary implementation functions for tensors with contiguous memory layout

    template <typename T1, typename T2, typename Op,
        enable_if_t<is_tensor<T1>::value && is_tensor<T2>::value
               && is_contiguous_tensor<T1>::value
               && is_contiguous_tensor<T2>::value>* = nullptr>
    inline Tensor<typename T1::value_type,
        Eigen::aligned_allocator<typename T1::value_type> >
    binary(const T1& tensor1, const T2& tensor2, Op&& op) {
      TA_ASSERT(! tensor1.empty());
      TA_ASSERT(! tensor2.empty());
      TA_ASSERT(is_range_congruent(tensor1, tensor2));

      typedef Tensor<typename T1::value_type,
          Eigen::aligned_allocator<typename T1::value_type> > result_type;

      result_type result(clone_range(tensor1));
      const auto volume = tensor1.range().volume();

      math::vector_op(std::forward<Op>(op), volume, result.data(),
          tensor1.data(), tensor2.data());

      return result;
    }

    template <typename T1, typename T2, typename InputOp, typename OutputOp,
        enable_if_t<is_tensor<T1>::value && is_tensor<T2>::value
               && is_contiguous_tensor<T1>::value
               && is_contiguous_tensor<T2>::value>* = nullptr>
    inline Tensor<typename T1::value_type,
        Eigen::aligned_allocator<typename T1::value_type> >
    binary(const T1& tensor1, const T2& tensor2,
        const Permutation& perm, InputOp&& input_op, OutputOp&& output_op)
    {
      TA_ASSERT(! tensor1.empty());
      TA_ASSERT(! tensor2.empty());
      TA_ASSERT(is_range_congruent(tensor1, tensor2));
      TA_ASSERT(perm);
      TA_ASSERT(perm.dim() == tensor1.range().rank());

      typedef Tensor<typename T1::value_type,
          Eigen::aligned_allocator<typename T1::value_type> > result_type;

      result_type result(perm * clone_range(tensor1));

      permute(std::forward<InputOp>(input_op), std::forward<OutputOp>(output_op),
          result, perm, tensor1, tensor2);

      return result;
    }


    template <typename T1, typename T2, typename Op,
        enable_if_t<is_tensor<T1>::value && is_tensor<T2>::value
               && is_contiguous_tensor<T1>::value
               && is_contiguous_tensor<T2>::value>* = nullptr>
    inline void inplace_binary(T1& tensor1, const T2& tensor2, Op&& op) {
      TA_ASSERT(! tensor1.empty());
      TA_ASSERT(! tensor2.empty());
      TA_ASSERT(is_range_congruent(tensor1, tensor2));

      const auto volume = tensor1.range().volume();

      math::vector_op(std::forward<Op>(op), volume, tensor1.data(),
          tensor2.data());
    }


    template <typename InputOp, typename OutputOp, typename T1, typename T2,
        enable_if_t<is_tensor<T1>::value && is_tensor<T2>::value
               && is_contiguous_tensor<T1>::value
               && is_contiguous_tensor<T2>::value>* = nullptr>
    inline void inplace_binary(T1& tensor1, const T2& tensor2,
        const Permutation& perm, InputOp&& input_op, OutputOp&& output_op)
    {
      TA_ASSERT(! tensor1.empty());
      TA_ASSERT(! tensor2.empty());
      TA_ASSERT(is_range_congruent(tensor1, tensor2, perm));

      permute(std::forward<InputOp>(input_op), std::forward<OutputOp>(output_op),
          tensor1, perm, tensor2);
    }

    // Binary impl functions with strided access -------------------------------

    template <typename T1, typename T2, typename Op,
        enable_if_t<is_tensor<T1>::value && is_tensor<T2>::value
               && ! (is_contiguous_tensor<T1>::value
               && is_contiguous_tensor<T2>::value)>* = nullptr>
    inline Tensor<typename T1::value_type,
        Eigen::aligned_allocator<typename T1::value_type> >
    binary(const T1& tensor1, const T2& tensor2, Op&& op) {
      TA_ASSERT(! tensor1.empty());
      TA_ASSERT(! tensor2.empty());
      TA_ASSERT(is_range_congruent(tensor1, tensor2));

      typedef Tensor<typename T1::value_type,
          Eigen::aligned_allocator<typename T1::value_type> > result_type;

      result_type result(clone_range(tensor1));
      const auto stride = inner_size(tensor1, tensor2);
      const auto volume = tensor1.range().volume();

      for(decltype(volume) i = 0ul; i < volume; i += stride)
        math::vector_op(std::forward<Op>(op), stride, result.data() + i,
          tensor1.data() + tensor1.range().ord(i),
          tensor2.data() + tensor2.range().ord(i));

      return result;
    }

    template <typename T1, typename T2, typename Op,
        enable_if_t<is_tensor<T1>::value && is_tensor<T2>::value
               && ! (is_contiguous_tensor<T1>::value
               && is_contiguous_tensor<T2>::value)>* = nullptr>
    inline void inplace_binary(T1& tensor1, const T2& tensor2, Op&& op) {
      TA_ASSERT(! tensor1.empty());
      TA_ASSERT(! tensor2.empty());
      TA_ASSERT(is_range_congruent(tensor1, tensor2));

      const auto stride = inner_size(tensor1, tensor2);
      const typename T1::size_type volume = tensor1.range().volume();

      for(typename T1::size_type i = 0ul; i < volume; i += stride)
        math::vector_op(std::forward<Op>(op), stride,
          tensor1.data() + tensor1.range().ord(i),
          tensor2.data() + tensor2.range().ord(i));
    }

  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_BINARY_H__INCLUDED
