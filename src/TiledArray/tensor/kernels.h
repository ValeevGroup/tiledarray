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

    // Vector operation implementation functions for tensors with contiguous
    // memory layout

    template <typename Op, typename T1, typename... Ts,
        enable_if_t<is_tensor<T1, Ts...>::value
               && is_contiguous_tensor<T1, Ts...>::value>* = nullptr>
    inline Tensor<typename T1::value_type,
        Eigen::aligned_allocator<typename T1::value_type> >
    tensor_op(Op&& op, const T1& tensor1, const Ts&... tensors) {
      TA_ASSERT(! empty(tensor1, tensors...));
      TA_ASSERT(is_range_set_congruent(tensor1, tensors...));

      typedef Tensor<typename T1::value_type,
          Eigen::aligned_allocator<typename T1::value_type> > result_type;

      result_type result(clone_range(tensor1));
      const auto volume = tensor1.range().volume();

      math::vector_op(std::forward<Op>(op), volume, result.data(),
          tensor1.data(), tensors.data()...);

      return result;
    }

    template <typename Op, typename T1, typename... Ts,
        enable_if_t<is_tensor<T1, Ts...>::value
                 && is_contiguous_tensor<T1, Ts...>::value>* = nullptr>
    inline void inplace_tensor_op(Op&& op, T1& tensor1, const Ts&... tensors) {
      TA_ASSERT(! empty(tensor1, tensors...));
      TA_ASSERT(is_range_set_congruent(tensor1, tensors...));

      const auto volume = tensor1.range().volume();

      math::vector_op(std::forward<Op>(op), volume, tensor1.data(),
          tensors.data()...);
    }



    template <typename InputOp, typename OutputOp, typename T1, typename... Ts,
        enable_if_t<is_tensor<T1, Ts...>::value
               && is_contiguous_tensor<T1, Ts...>::value>* = nullptr>
    inline Tensor<typename T1::value_type,
        Eigen::aligned_allocator<typename T1::value_type> >
    tensor_op(InputOp&& input_op, OutputOp&& output_op, const Permutation& perm,
        const T1& tensor1, const Ts&... tensors)
    {
      TA_ASSERT(! empt(tensor1, tensors...));
      TA_ASSERT(is_range_set_congruent(tensor1, tensors...));
      TA_ASSERT(perm);
      TA_ASSERT(perm.dim() == tensor1.range().rank());

      typedef Tensor<typename T1::value_type,
          Eigen::aligned_allocator<typename T1::value_type> > result_type;

      result_type result(perm * clone_range(tensor1));

      permute(std::forward<InputOp>(input_op), std::forward<OutputOp>(output_op),
          result, perm, tensor1, tensors...);

      return result;
    }

    template <typename InputOp, typename OutputOp, typename T1, typename T2, typename... Ts,
        enable_if_t<is_tensor<T1, T2, Ts...>::value
               && is_contiguous_tensor<T1, T2, Ts...>::value>* = nullptr>
    inline void inplace_tensor_op(InputOp&& input_op, OutputOp&& output_op,
        const Permutation& perm, T1& tensor1, const T2& tensor2, const Ts&... tensors)
    {
      TA_ASSERT(! empty(tensor1, tensor2, tensors...));
      TA_ASSERT(is_range_congruent(tensor1, tensor2, perm));
      TA_ASSERT(is_range_set_congruent(tensor2, tensors...));
      TA_ASSERT(perm);
      TA_ASSERT(perm.dim() == tensor1.range().rank());

      permute(std::forward<InputOp>(input_op), std::forward<OutputOp>(output_op),
          tensor1, perm, tensor2, tensors...);
    }


    // Vector operation implementation functions for tensors with non-contiguous
    // memory layout

    template <typename Op, typename T1, typename... Ts,
        enable_if_t<is_tensor<T1, Ts...>::value
               && ! (is_contiguous_tensor<T1, Ts...>::value)>* = nullptr>
    inline Tensor<typename T1::value_type,
        Eigen::aligned_allocator<typename T1::value_type> >
    tensor_op(Op&& op, const T1& tensor1, const Ts&... tensors) {
      TA_ASSERT(! empty(tensor1, tensors...));
      TA_ASSERT(is_range_congruent(tensor1, tensors...));

      typedef Tensor<typename T1::value_type,
          Eigen::aligned_allocator<typename T1::value_type> > result_type;

      result_type result(clone_range(tensor1));
      const auto stride = inner_size(tensor1, tensors...);
      const auto volume = tensor1.range().volume();

      for(decltype(tensor1.range().volume()) i = 0ul; i < volume; i += stride)
        math::vector_op(op, stride, result.data() + i,
          tensor1.data() + tensor1.range().ord(i),
          (tensors.data() + tensors.range().ord(i))...);

      return result;
    }

    template <typename Op, typename T1, typename... Ts,
        enable_if_t<is_tensor<T1, Ts...>::value
               && ! (is_contiguous_tensor<T1, Ts...>::value)>* = nullptr>
    inline void inplace_tensor_op(Op&& op, T1& tensor1, const Ts&... tensors) {
      TA_ASSERT(! empty(tensor1, tensors...));
      TA_ASSERT(is_range_congruent(tensor1, tensors...));

      const auto stride = inner_size(tensor1, tensors...);
      const typename T1::size_type volume = tensor1.range().volume();

      for(typename T1::size_type i = 0ul; i < volume; i += stride)
        math::vector_op(op, stride,
          tensor1.data() + tensor1.range().ord(i),
          (tensors.data() + tensors.range().ord(i))...);
    }


    // Tensor initialization functions for argument tensors with contiguous
    // memory layout

    template <typename Op, typename T1, typename... Ts>
    struct uninitialized_wrapper_op {
    private:
      Op op_;

    public:
      uninitialized_wrapper_op() = delete;
      uninitialized_wrapper_op(const uninitialized_wrapper_op&) = default;
      uninitialized_wrapper_op(uninitialized_wrapper_op&&) = default;
      ~uninitialized_wrapper_op() = default;
      uninitialized_wrapper_op& operator=(const uninitialized_wrapper_op&) = delete;
      uninitialized_wrapper_op& operator=(uninitialized_wrapper_op&&) = delete;

      uninitialized_wrapper_op(Op&& op) : op_(std::forward<Op>(op)) { }

      void operator()(typename T1::pointer restrict result,
          typename Ts::const_reference restrict... ts)
      { new(result) typename T1::value_type(op_(ts...)); };
    };

    /// Initialize tensor with contiguous tensor arguments

    /// This function initializes the elements of \c tensor1 with the result of
    /// \c op(tensors[i]...)
    /// \pre The memory of \c tensor1 has been allocated but not initialized.
    /// \tparam Op The element initialization operation type
    /// \tparam T1 The result tensor type
    /// \tparam Ts The argument tensor types
    /// \param op The result tensor element initialization operation
    /// \param tensor1 The result tensor
    /// \param tensors The argument tensors
    template <typename Op, typename T1, typename... Ts,
        enable_if_t<is_tensor<T1, Ts...>::value
               && is_contiguous_tensor<T1, Ts...>::value>* = nullptr>
    inline void tensor_init(Op&& op, T1& tensor1, const Ts&... tensors) {
      TA_ASSERT(! empty(tensor1, tensors...));
      TA_ASSERT(is_range_set_congruent(tensor1, tensors...));

      const auto volume = tensor1.range().volume();
      uninitialized_wrapper_op<Op, T1, Ts...> wrapper_op(std::forward<Op>(op));

      math::vector_ptr_op(wrapper_op, volume, tensor1.data(), tensors.data()...);
    }


    /// Initialize tensor with permuted tensor arguments

    /// This function initializes the elements of \c tensor1 with the result of
    /// \c op(tensors[i]...)
    /// \pre The memory of \c tensor1 has been allocated but not initialized.
    /// \tparam Op The element initialization operation type
    /// \tparam T1 The result tensor type
    /// \tparam Ts The argument tensor types
    /// \param op The result tensor element initialization operation
    /// \param perm The permutation that will be applied to tensor2
    /// \param tensor1 The result tensor
    /// \param tensors The argument tensors
    template <typename Op, typename T1, typename... Ts,
        enable_if_t<is_tensor<T1, Ts...>::value
               && is_contiguous_tensor<T1, Ts...>::value>* = nullptr>
    inline void tensor_init(Op&& op, const Permutation& perm, T1& tensor1,
        const Ts&... tensors)
    {
      TA_ASSERT(! empty(tensor1, tensors...));
      TA_ASSERT(is_range_set_congruent(perm, tensor1, tensors...));
      TA_ASSERT(perm);
      TA_ASSERT(perm.dim() == tensor1.range().rank());

      auto output_op = [=] (typename T1::pointer restrict result,
          typename T1::const_reference  restrict temp)
          { new(result) typename T1::value_type(temp); };

      permute(std::forward<Op>(op), output_op, tensor1, perm, tensors...);
    }

    /// Initialize tensor with one or more non-contiguous tensor arguments

    /// This function initializes the elements of \c tensor1 with the result of
    /// \c op(tensors[i]...)
    /// \pre The memory of \c tensor1 has been allocated but not initialized.
    /// \tparam Op The element initialization operation type
    /// \tparam T1 The result tensor type
    /// \tparam Ts The argument tensor types
    /// \param op The result tensor element initialization operation
    /// \param tensor1 The result tensor
    /// \param tensors The argument tensors
    template <typename Op, typename T1, typename... Ts,
        enable_if_t<is_tensor<T1>::value && is_tensor<Ts...>::value
                 && is_contiguous_tensor<T1>::value
                 && ! is_contiguous_tensor<Ts...>::value>* = nullptr>
    inline void tensor_init(Op&& op, T1& tensor1, const Ts&... tensors) {
      TA_ASSERT(! empty(tensor1, tensors...));
      TA_ASSERT(is_range_set_congruent(tensor1, tensors...));

      const auto stride = inner_size(tensors...);
      const auto volume = tensor1.range().volume();
      uninitialized_wrapper_op<Op, T1, Ts...> wrapper_op(std::forward<Op>(op));

      for(decltype(volume) i = 0ul; i < volume; i += stride)
        math::vector_ptr_op(wrapper_op, stride, tensor1.data() + i,
            (tensors.data() + tensors.range().ord(i))...);
    }


  }  // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_BINARY_H__INCLUDED
