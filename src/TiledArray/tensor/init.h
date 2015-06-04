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
 *  init.h
 *  Jun 3, 2015
 *
 */

#ifndef TILEDARRAY_TENSOR_INIT_H__INCLUDED
#define TILEDARRAY_TENSOR_INIT_H__INCLUDED

#include <TiledArray/tensor/utility.h>
#include <TiledArray/tensor/permute.h>
#include <TiledArray/math/eigen.h>

namespace TiledArray {
  namespace detail {

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

      math::vector_ptr_op(uninitialized_wrapper_op<Op, T1, Ts...>(std::forward<Op>(op)),
          volume, tensor1.data(), tensors.data()...);
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
        enable_if_t<is_tensor<T1>::value && is_tensor<Ts...>::value
               && is_contiguous_tensor<T1>::value
               && is_contiguous_tensor<Ts...>::value>* = nullptr>
    inline void tensor_init(Op&& op, const Permutation& perm, T1& tensor1,
        const Ts&... tensors)
    {
      TA_ASSERT(! empty(tensor1, tensors...));
      TA_ASSERT(is_range_set_congruent(perm, tensor1, tensors...));

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

      for(decltype(volume) i = 0ul; i < volume; i += stride)
        math::vector_ptr_op(uninitialized_wrapper_op<Op, T1, Ts...>(std::forward<Op>(op)),
            stride, tensor1.data() + i, (tensors.data() + tensors.range().ord(i))...);
    }

  }  // namespace detail
} // namespace TiledArray


#endif // TILEDARRAY_TENSOR_INIT_H__INCLUDED
