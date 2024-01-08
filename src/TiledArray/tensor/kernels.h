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
 *  kernels.h
 *  Jun 1, 2015
 *
 */

#ifndef TILEDARRAY_TENSOR_KENERLS_H__INCLUDED
#define TILEDARRAY_TENSOR_KENERLS_H__INCLUDED

#include <TiledArray/tensor/permute.h>
#include <TiledArray/tensor/utility.h>
#include <TiledArray/util/vector.h>

namespace TiledArray {

template <typename, typename>
class Tensor;

namespace detail {

/// customization point transform functionality to tensor class T, useful for
/// nonintrusive extension of T to be usable as element type T in Tensor<T>
template <typename T>
struct transform;

// -------------------------------------------------------------------------
// Tensor kernel operations that generate a new tensor

/// Tensor operations with contiguous data

/// This function transforms argument tensors applying a callable directly
/// (i.e., tensor-wise as \c result=op(tensor1,tensors...) ),
/// or by lowering to the elements (i.e., element-wise as
/// \c result[i]=op(tensor1[i],tensors[i]...)  )
/// \tparam TR The tensor result type
/// \tparam Op A callable used to produce TR when called with the argument
/// tensors, or produce TR's elements when called with the argument tensor's
/// elements
/// \tparam T1 The first argument tensor type
/// \tparam Ts The remaining argument tensor types
/// \param op The result tensor element initialization operation
/// \param tensor1 The first argument tensor
/// \param tensors The remaining argument tensors
template <typename TR, typename Op, typename T1, typename... Ts,
          typename = std::enable_if_t<
              detail::is_nested_tensor_v<TR, T1, Ts...> ||
              std::is_invocable_r_v<TR, Op, const T1&, const Ts&...>>>
inline TR tensor_op(Op&& op, const T1& tensor1, const Ts&... tensors) {
  if constexpr (std::is_invocable_r_v<TR, Op, const T1&, const Ts&...>) {
    return std::forward<Op>(op)(tensor1, tensors...);
  } else {
    static_assert(detail::is_nested_tensor_v<TR, T1, Ts...>);
    return TiledArray::detail::transform<TR>()(std::forward<Op>(op), tensor1,
                                               tensors...);
  }
  abort();  // unreachable
}

/// Tensor permutation operations with contiguous data

/// This function transforms argument tensors applying a callable directly
/// (i.e., tensor-wise as \c result=op(perm,tensor1,tensors...) ),
/// or by lowering to the elements (i.e., element-wise as
/// \c result[i]=op(perm,tensor1[i],tensors[i]...)  )
/// \tparam TR The tensor result type
/// \tparam Op A callable used to produce TR when called with the argument
/// tensors, or produce TR's elements when called with the argument tensor's
/// elements
/// \tparam T1 The result tensor type
/// \tparam Ts The argument tensor types
/// \param[in] op The operation that is used to compute the result
/// value from the input arguments
/// \param[in] perm The permutation applied to the argument tensors
/// \param[in] tensor1 The first argument tensor
/// \param[in] tensors The remaining argument tensors
template <typename TR, typename Op, typename T1, typename... Ts,
          typename std::enable_if<
              is_nested_tensor_v<T1, Ts...> &&
              is_contiguous_tensor<T1, Ts...>::value>::type* = nullptr>
inline TR tensor_op(Op&& op, const Permutation& perm, const T1& tensor1,
                    const Ts&... tensors) {
  if constexpr (std::is_invocable_r_v<TR, Op, const Permutation&, const T1&,
                                      const Ts&...>) {
    return std::forward<Op>(op)(perm, tensor1, tensors...);
  } else {
    return TiledArray::detail::transform<TR>()(std::forward<Op>(op), perm,
                                               tensor1, tensors...);
  }
}

/// provides transform functionality to class \p T, useful for nonintrusive
/// extension of a tensor type \p T to be usable as element type \p T in
/// \c Tensor<T>
/// \tparam T a tensor type
/// \note The default implementation
/// constructs T, then computes it by coiterating over elements of the argument
/// tensors and transforming with the transform \c Op .
/// This should be specialized for classes like TiledArray::Tensor that
/// already include the appropriate transform constructors already
template <typename T>
struct transform {
  /// creates a result tensor in which element \c i is obtained by \c
  /// op(tensor[i], tensors[i]...)
  template <typename Op, typename Tensor, typename... Tensors>
  T operator()(Op&& op, Tensor&& tensor, Tensors&&... tensors) const {
    TA_ASSERT(!empty(tensor, tensors...));
    TA_ASSERT(is_range_set_congruent(tensor, tensors...));

    const auto& range = tensor.range();
    T result(range);
    this->operator()(result, std::forward<Op>(op), std::forward<Tensor>(tensor),
                     std::forward<Tensors>(tensors)...);
    return result;
  }

  /// an in-place version of above
  /// \note result  must be already allocated
  template <typename Op, typename Tensor, typename... Tensors>
  void operator()(T& result, Op&& op, Tensor&& tensor,
                  Tensors&&... tensors) const {
    TA_ASSERT(!empty(result, tensor, tensors...));
    TA_ASSERT(is_range_set_congruent(result, tensor, tensors...));

    const auto& range = result.range();
    for (auto&& i : range)
      result[std::forward<decltype(i)>(i)] = std::forward<Op>(op)(
          std::forward<Tensor>(tensor)[std::forward<decltype(i)>(i)],
          std::forward<Tensors>(tensors)[std::forward<decltype(i)>(i)]...);
  }

  template <typename Op, typename Tensor, typename... Tensors>
  T operator()(Op&& op, const Permutation& perm, Tensor&& tensor,
               Tensors&&... tensors) const {
    TA_ASSERT(!empty(tensor, tensors...));
    TA_ASSERT(is_range_set_congruent(tensor, tensors...));
    TA_ASSERT(perm);
    TA_ASSERT(perm.size() == tensor.range().rank());

    const auto& range = tensor.range();
    T result(perm ^ range);
    this->operator()(result, std::forward<Op>(op), perm,
                     std::forward<Tensor>(tensor),
                     std::forward<Tensors>(tensors)...);
    return result;
  }

  template <typename Op, typename Tensor, typename... Tensors>
  void operator()(T& result, Op&& op, const Permutation& perm, Tensor&& tensor,
                  Tensors&&... tensors) const {
    TA_ASSERT(!empty(result, tensor, tensors...));
    TA_ASSERT(is_range_congruent(result, tensor, perm));
    TA_ASSERT(is_range_set_congruent(tensor, tensors...));
    TA_ASSERT(perm);
    TA_ASSERT(perm.size() == tensor.range().rank());

    const auto& range = tensor.range();
    for (auto&& i : range)
      result[perm ^ std::forward<decltype(i)>(i)] = std::forward<Op>(op)(
          std::forward<Tensor>(tensor)[std::forward<decltype(i)>(i)],
          std::forward<Tensors>(tensors)[std::forward<decltype(i)>(i)]...);
  }
};

// -------------------------------------------------------------------------
// Tensor kernel operations with in-place memory operations

/// In-place tensor operations with contiguous data

/// This function sets the elements of \c result with the result of
/// \c op(tensors[i]...)
/// \tparam Op The element initialization operation type
/// \tparam TR The result tensor type
/// \tparam Ts The remaining argument tensor types
/// \param[in] op The result tensor element initialization operation
/// \param[in,out] result The result tensor
/// \param[in] tensors The argument tensors
template <typename Op, typename TR, typename... Ts,
          typename std::enable_if<
              is_tensor<TR, Ts...>::value &&
              is_contiguous_tensor<TR, Ts...>::value>::type* = nullptr>
inline void inplace_tensor_op(Op&& op, TR& result, const Ts&... tensors) {
  TA_ASSERT(!empty(result, tensors...));
  TA_ASSERT(is_range_set_congruent(result, tensors...));

  const auto volume = result.range().volume();

  math::inplace_vector_op(std::forward<Op>(op), volume, result.data(),
                          tensors.data()...);
}

/// In-place tensor of tensors operations with contiguous data

/// This function sets the elements of \c result with the result of
/// \c op(tensors[i]...)
/// \tparam Op The element initialization operation type
/// \tparam TR The result tensor type
/// \tparam Ts The remaining argument tensor types
/// \param[in] op The result tensor element initialization operation
/// \param[in,out] result The result tensor
/// \param[in] tensors The argument tensors
template <typename Op, typename TR, typename... Ts,
          typename std::enable_if<
              !is_tensor_v<TR, Ts...> &&
              is_contiguous_tensor<TR, Ts...>::value>::type* = nullptr>
inline void inplace_tensor_op(Op&& op, TR& result, const Ts&... tensors) {
  TA_ASSERT(!empty(result, tensors...));
  TA_ASSERT(is_range_set_congruent(result, tensors...));

  const auto volume = result.range().volume();

  for (decltype(result.range().volume()) ord = 0ul; ord < volume; ++ord) {
    if constexpr (std::is_invocable_r_v<void, Op, typename TR::value_type&,
                                        typename Ts::value_type...>)
      op(result.at_ordinal(ord), tensors.at_ordinal(ord)...);
    else
      inplace_tensor_op(op, result.at_ordinal(ord), tensors.at_ordinal(ord)...);
  }
}

/// In-place tensor permutation operations with contiguous data

/// This function sets the \c i -th element of \c result with the result of
/// \c op(tensor1[i],tensors[i]...)
/// The expected signature of the input operations is:
/// \code
/// Result::value_type op(const T1::value_type, const Ts::value_type...)
/// \endcode
/// The expected signature of the output operations is:
/// \code
/// void op(TR::value_type*, const TR::value_type)
/// \endcode
/// \tparam InputOp The input operation type
/// \tparam OutputOp The output operation type
/// \tparam TR The result tensor type
/// \tparam T1 The first argument tensor type
/// \tparam Ts The remaining argument tensor types
/// \param[in] input_op The operation that is used to generate the output
/// value from the input arguments
/// \param[in] output_op The operation that is used to set the value of the
/// result tensor given the element pointer and the result value
/// \param[in] perm The permutation applied to the argument tensors
/// \param[in,out] result The result tensor
/// \param[in] tensor1 The first argument tensor
/// \param[in] tensors The remaining argument tensors
template <typename InputOp, typename OutputOp, typename TR, typename T1,
          typename... Ts,
          typename std::enable_if<
              is_tensor<TR, T1, Ts...>::value &&
              is_contiguous_tensor<TR, T1, Ts...>::value>::type* = nullptr>
inline void inplace_tensor_op(InputOp&& input_op, OutputOp&& output_op,
                              const Permutation& perm, TR& result,
                              const T1& tensor1, const Ts&... tensors) {
  TA_ASSERT(!empty(result, tensor1, tensors...));
  TA_ASSERT(is_range_congruent(result, tensor1, perm));
  TA_ASSERT(is_range_set_congruent(tensor1, tensors...));
  TA_ASSERT(perm);
  TA_ASSERT(perm.size() == tensor1.range().rank());

  permute(std::forward<InputOp>(input_op), std::forward<OutputOp>(output_op),
          result, perm, tensor1, tensors...);
}

/// In-place tensor of tensors permutation operations with contiguous data

/// This function sets the \c i -th element of \c result with the result of
/// \c op(tensor1[i], tensors[i]...)
/// The expected signature of the input operations is:
/// \code
/// Result::value_type op(const T1::value_type::value_type, const
/// Ts::value_type::value_type...)
/// \endcode
/// The expected signature of the output
/// operations is:
/// \code
/// void op(TR::value_type::value_type*, const
/// TR::value_type::value_type)
/// \endcode
/// \tparam InputOp The input operation type
/// \tparam OutputOp The output operation type
/// \tparam TR The result tensor type
/// \tparam T1 The first argument tensor type
/// \tparam Ts The remaining argument tensor types
/// \param[in] input_op The operation that is used to
/// generate the output value from the input arguments
/// \param[in] output_op The operation that is used to set the value
/// of the result tensor given the element pointer and the result value
/// \param[in] perm The permutation applied to the argument tensors
/// \param[in,out] result The result tensor
/// \param[in] tensor1 The first argument tensor
/// \param[in] tensors The remaining argument tensors
template <typename InputOp, typename OutputOp, typename TR, typename T1,
          typename... Ts,
          typename std::enable_if<
              is_tensor_of_tensor<TR, T1, Ts...>::value &&
              is_contiguous_tensor<TR, T1, Ts...>::value>::type* = nullptr>
inline void inplace_tensor_op(InputOp&& input_op, OutputOp&& output_op,
                              const Permutation& perm, TR& result,
                              const T1& tensor1, const Ts&... tensors) {
  TA_ASSERT(!empty(result, tensor1, tensors...));
  TA_ASSERT(is_range_congruent(result, tensor1, perm));
  TA_ASSERT(is_range_set_congruent(tensor1, tensors...));
  TA_ASSERT(perm);
  TA_ASSERT(perm.size() == tensor1.range().rank());

  auto wrapper_input_op =
      [&input_op](typename T1::const_reference MADNESS_RESTRICT value1,
                  typename Ts::const_reference MADNESS_RESTRICT... values) ->
      typename T1::value_type {
        return tensor_op<TR::value_type>(std::forward<InputOp>(input_op),
                                         value1, values...);
      };

  auto wrapper_output_op =
      [&output_op](typename T1::pointer MADNESS_RESTRICT const result_value,
                   const typename TR::value_type value) {
        inplace_tensor_op(std::forward<OutputOp>(output_op), *result_value,
                          value);
      };

  permute(std::move(wrapper_input_op), std::move(wrapper_output_op), result,
          perm, tensor1, tensors...);
}

/// In-place tensor operations with non-contiguous data

/// This function sets the \c i -th element of \c result with the result of
/// \c op(tensors[i]...)
/// \tparam Op The element initialization operation type
/// \tparam TR The result tensor type
/// \tparam Ts The remaining argument tensor types
/// \param[in] op The result tensor element initialization operation
/// \param[in,out] result The result tensor
/// \param[in] tensors The argument tensors
template <typename Op, typename TR, typename... Ts,
          typename std::enable_if<
              is_tensor<TR, Ts...>::value &&
              !(is_contiguous_tensor<TR, Ts...>::value)>::type* = nullptr>
inline void inplace_tensor_op(Op&& op, TR& result, const Ts&... tensors) {
  TA_ASSERT(!empty(result, tensors...));
  TA_ASSERT(is_range_set_congruent(result, tensors...));

  const auto volume = result.range().volume();

  if constexpr (detail::has_member_function_data_anyreturn_v<TR> &&
                (detail::has_member_function_data_anyreturn_v<Ts> && ...)) {
    const auto stride = inner_size(result, tensors...);
    for (std::decay_t<decltype(volume)> i = 0ul; i < volume; i += stride)
      math::inplace_vector_op(std::forward<Op>(op), stride,
                              result.data() + result.range().ordinal(i),
                              (tensors.data() + tensors.range().ordinal(i))...);
  } else {  // if 1+ tensor lacks data() must iterate over individual elements
    auto& result_rng = result.range();
    using signed_idx_t = Range::index_difference_type;
    auto result_lobound = signed_idx_t(result_rng.lobound());
    for (auto&& idx : result_rng) {
      using namespace container::operators;
      std::forward<Op>(op)(
          result[idx], (tensors[idx - result_lobound +
                                signed_idx_t(tensors.range().lobound())])...);
    }
  }
}

/// In-place tensor of tensors operations with non-contiguous data

/// This function sets the \c i -th element of \c result with the result of
/// \c op(tensors[i]...)
/// \tparam Op The element initialization operation type
/// \tparam TR The result tensor type
/// \tparam Ts The remaining argument tensor types
/// \param[in] op The result tensor element initialization operation
/// \param[in,out] result The result tensor
/// \param[in] tensors The argument tensors
template <typename Op, typename TR, typename... Ts,
          typename std::enable_if<
              is_tensor_of_tensor<TR, Ts...>::value &&
              !(is_contiguous_tensor<TR, Ts...>::value)>::type* = nullptr>
inline void inplace_tensor_op(Op&& op, TR& result, const Ts&... tensors) {
  TA_ASSERT(!empty(result, tensors...));
  TA_ASSERT(is_range_set_congruent(result, tensors...));

  const auto volume = result.range().volume();

  if constexpr (detail::has_member_function_data_anyreturn_v<TR> &&
                (detail::has_member_function_data_anyreturn_v<Ts> && ...)) {
    const auto stride = inner_size(result, tensors...);
    auto inplace_tensor_range =
        [&op, stride](
            typename TR::pointer MADNESS_RESTRICT const result_data,
            typename Ts::const_pointer MADNESS_RESTRICT const... tensors_data) {
          for (decltype(result.range().volume()) i = 0ul; i < stride; ++i)
            inplace_tensor_op(op, result_data[i], tensors_data[i]...);
        };

    for (std::decay_t<decltype(volume)> ord = 0ul; ord < volume; ord += stride)
      inplace_tensor_range(result.data() + result.range().ordinal(ord),
                           (tensors.data() + tensors.range().ordinal(ord))...);
  } else {  // if 1+ tensor lacks data() must iterate over individual elements
    auto& result_rng = result.range();
    using signed_idx_t = Range::index_difference_type;
    auto result_lobound = signed_idx_t(result_rng.lobound());
    for (auto&& idx : result_rng) {
      using namespace container::operators;
      std::forward<Op>(op)(
          result[idx], (tensors[idx - result_lobound +
                                signed_idx_t(tensors.range().lobound())])...);
    }
  }
}

// -------------------------------------------------------------------------
// Tensor initialization functions for argument tensors with contiguous
// memory layout

/// Initialize tensor with contiguous tensor arguments

/// This function initializes the \c i -th element of \c result with the result
/// of \c op(tensors[i]...)
/// \pre The memory of \c tensor1 has been allocated but not initialized.
/// \tparam Op The element initialization operation type
/// \tparam TR The result tensor type
/// \tparam Ts The argument tensor types
/// \param[in] op The result tensor element initialization operation
/// \param[out] result The result tensor
/// \param[in] tensors The argument tensors
template <typename Op, typename TR, typename... Ts,
          typename std::enable_if<
              is_tensor<TR, Ts...>::value &&
              is_contiguous_tensor<TR, Ts...>::value>::type* = nullptr>
inline void tensor_init(Op&& op, TR& result, const Ts&... tensors) {
  TA_ASSERT(!empty(result, tensors...));
  TA_ASSERT(is_range_set_congruent(result, tensors...));

  const auto volume = result.range().volume();

  auto wrapper_op = [&op](typename TR::pointer MADNESS_RESTRICT result,
                          typename Ts::const_reference MADNESS_RESTRICT... ts) {
    new (result) typename TR::value_type(std::forward<Op>(op)(ts...));
  };

  math::vector_ptr_op(std::move(wrapper_op), volume, result.data(),
                      tensors.data()...);
}

/// Initialize nested tensor with contiguous tensor arguments

/// This function initializes the \c i -th element of \c result with the result
/// of \c op(tensors[i]...)
/// \pre The memory of \c tensor1 has been allocated but not initialized.
/// \tparam Op The element initialization operation type
/// \tparam TR The result tensor type
/// \tparam Ts The argument tensor types
/// \param[in] op The result tensor element initialization operation
/// \param[out] result The result tensor
/// \param[in] tensors The argument tensors
template <
    typename Op, typename TR, typename... Ts,
    typename std::enable_if<(is_nested_tensor<TR, Ts...>::value &&
                             !is_tensor<TR, Ts...>::value) &&
                            is_contiguous_tensor<TR>::value>::type* = nullptr>
inline void tensor_init(Op&& op, TR& result, const Ts&... tensors) {
  TA_ASSERT(!empty(result, tensors...));
  TA_ASSERT(is_range_set_congruent(result, tensors...));

  const auto volume = result.range().volume();

  if constexpr (std::is_invocable_r_v<TR, Op, const Ts&...>) {
    result = std::forward<Op>(op)(tensors...);
  } else {
    for (decltype(result.range().volume()) ord = 0ul; ord < volume; ++ord) {
      new (result.data() + ord) typename TR::value_type(
          tensor_op<typename TR::value_type>(op, tensors.at_ordinal(ord)...));
    }
  }
}

/// Initialize tensor with permuted tensor arguments

/// This function initializes the \c i -th element of \c result with the result
/// of \c op(tensor1[i], tensors[i]...)
/// \pre The memory of \c result has been
/// allocated but not initialized.
/// \tparam Op The element initialization operation type
/// \tparam TR The result tensor type
/// \tparam T1 The first argument tensor type
/// \tparam Ts The argument tensor types
/// \param[in] op The result tensor element initialization operation
/// \param[in] perm The permutation that will be applied to tensor2
/// \param[out] result The result tensor
/// \param[in] tensor1 The first argument tensor
/// \param[in] tensors The argument tensors
template <
    typename Op, typename TR, typename T1, typename... Ts,
    typename std::enable_if<is_tensor<TR, T1, Ts...>::value>::type* = nullptr>
inline void tensor_init(Op&& op, const Permutation& perm, TR& result,
                        const T1& tensor1, const Ts&... tensors) {
  TA_ASSERT(!empty(result, tensor1, tensors...));
  TA_ASSERT(is_range_set_congruent(perm, result, tensor1, tensors...));
  TA_ASSERT(perm);
  TA_ASSERT(perm.size() == result.range().rank());

  auto output_op = [](typename TR::pointer MADNESS_RESTRICT result,
                      typename TR::const_reference MADNESS_RESTRICT temp) {
    new (result) typename TR::value_type(temp);
  };

  permute(std::forward<Op>(op), std::move(output_op), result, perm, tensor1,
          tensors...);
}

/// Initialize tensor of tensors with permuted tensor arguments

/// This function initializes the \c i -th element of \c result with the result
/// of \c op(tensor1[i], tensors[i]...)
/// \pre The memory of \c result has been allocated but not initialized.
/// \tparam Op The element initialization operation type
/// \tparam Perm A permutation type
/// \tparam TR The result tensor type
/// \tparam T1 The first argument tensor type
/// \tparam Ts The argument tensor types
/// \param[in] op The result tensor element initialization operation
/// \param[out] result The result tensor
/// \param[in] tensor1 The first argument tensor
/// \param[in] tensors The argument tensors
template <
    typename Op, typename TR, typename T1, typename... Ts,
    typename std::enable_if<is_nested_tensor<TR, T1, Ts...>::value &&
                            !is_tensor<TR, T1, Ts...>::value>::type* = nullptr>
inline void tensor_init(Op&& op, const Permutation& perm, TR& result,
                        const T1& tensor1, const Ts&... tensors) {
  TA_ASSERT(!empty(result, tensor1, tensors...));
  TA_ASSERT(is_range_set_congruent(perm, result, tensor1, tensors...));
  TA_ASSERT(perm);
  TA_ASSERT(perm.size() == result.range().rank());

  auto output_op = [](typename TR::pointer MADNESS_RESTRICT result,
                      typename TR::const_reference MADNESS_RESTRICT temp) {
    new (result) typename TR::value_type(temp);
  };
  auto tensor_input_op =
      [&op](typename T1::const_reference MADNESS_RESTRICT value1,
            typename Ts::const_reference MADNESS_RESTRICT... values) ->
      typename TR::value_type {
        return tensor_op<typename TR::value_type>(std::forward<Op>(op), value1,
                                                  values...);
      };

  permute(std::move(tensor_input_op), output_op, result, perm, tensor1,
          tensors...);
}

/// Initialize tensor with one or more non-contiguous tensor arguments

/// This function initializes the \c i -th element of \c result with the result
/// of \c op(tensor1[i], tensors[i]...)
/// \pre The memory of \c tensor1 has been allocated but not initialized.
/// \tparam Op The element initialization operation type
/// \tparam T1 The result tensor type
/// \tparam Ts The argument tensor types
/// \param[in] op The result tensor element initialization operation
/// \param[out] result The result tensor
/// \param[in] tensor1 The first argument tensor
/// \param[in] tensors The argument tensors
template <
    typename Op, typename TR, typename T1, typename... Ts,
    typename std::enable_if<
        is_tensor<TR, T1, Ts...>::value && is_contiguous_tensor<TR>::value &&
        !is_contiguous_tensor<T1, Ts...>::value>::type* = nullptr>
inline void tensor_init(Op&& op, TR& result, const T1& tensor1,
                        const Ts&... tensors) {
  TA_ASSERT(!empty(result, tensor1, tensors...));
  TA_ASSERT(is_range_set_congruent(result, tensor1, tensors...));

  const auto volume = tensor1.range().volume();

  auto wrapper_op = [&op](typename TR::pointer MADNESS_RESTRICT result_ptr,
                          const typename T1::value_type value1,
                          const typename Ts::value_type... values) {
    new (result_ptr) typename T1::value_type(op(value1, values...));
  };

  if constexpr (detail::has_member_function_data_anyreturn_v<TR> &&
                (detail::has_member_function_data_anyreturn_v<Ts> && ...)) {
    const auto stride = inner_size(tensor1, tensors...);
    for (decltype(tensor1.range().volume()) ord = 0ul; ord < volume;
         ord += stride)
      math::vector_ptr_op(wrapper_op, stride, result.data() + ord,
                          (tensor1.data() + tensor1.range().ordinal(ord)),
                          (tensors.data() + tensors.range().ordinal(ord))...);
  } else {  // if 1+ tensor lacks data() must iterate over individual elements
    auto& result_rng = result.range();
    using signed_idx_t = Range::index_difference_type;
    auto result_lobound = signed_idx_t(result_rng.lobound());
    for (auto&& idx : result_rng) {
      using namespace container::operators;
      const signed_idx_t relidx = idx - result_lobound;
      wrapper_op(
          &(result[idx]),
          tensor1[relidx + signed_idx_t(tensor1.range().lobound())],
          (tensors[relidx + signed_idx_t(tensors.range().lobound())])...);
    }
  }
}

/// Initialize tensor with one or more non-contiguous tensor arguments

/// This function initializes the \c i -th element of \c result with the result
/// of \c op(tensor1[i],tensors[i]...)
/// \pre The memory of \c tensor1 has been
/// allocated but not initialized.
/// \tparam Op The element initialization operation type
/// \tparam T1 The result tensor type
/// \tparam Ts The argument tensor types
/// \param[in] op The result tensor element initialization operation
/// \param[out] result The result tensor
/// \param[in] tensor1 The first
/// argument tensor
/// \param[in] tensors The argument tensors
template <typename Op, typename TR, typename T1, typename... Ts,
          typename std::enable_if<
              is_tensor_of_tensor<TR, T1, Ts...>::value &&
              is_contiguous_tensor<TR>::value &&
              !is_contiguous_tensor<T1, Ts...>::value>::type* = nullptr>
inline void tensor_init(Op&& op, TR& result, const T1& tensor1,
                        const Ts&... tensors) {
  TA_ASSERT(!empty(result, tensor1, tensors...));
  TA_ASSERT(is_range_set_congruent(result, tensor1, tensors...));

  const auto volume = tensor1.range().volume();

  if constexpr (detail::has_member_function_data_anyreturn_v<TR> &&
                (detail::has_member_function_data_anyreturn_v<Ts> && ...)) {
    const auto stride = inner_size(tensor1, tensors...);
    auto inplace_tensor_range =
        [&op, stride](
            typename TR::pointer MADNESS_RESTRICT const result_data,
            typename T1::const_pointer MADNESS_RESTRICT const tensor1_data,
            typename Ts::const_pointer MADNESS_RESTRICT const... tensors_data) {
          for (std::decay_t<decltype(volume)> i = 0ul; i < stride; ++i)
            new (result_data + i)
                typename TR::value_type(tensor_op<typename TR::value_type>(
                    op, tensor1_data[i], tensors_data[i]...));
        };

    for (std::decay_t<decltype(volume)> ord = 0ul; ord < volume; ord += stride)
      inplace_tensor_range(result.data() + ord,
                           (tensor1.data() + tensor1.range().ordinal(ord)),
                           (tensors.data() + tensors.range().ordinal(ord))...);
  } else {
    auto& result_rng = result.range();
    using signed_idx_t = Range::index_difference_type;
    auto result_lobound = signed_idx_t(result_rng.lobound());
    for (auto&& idx : result_rng) {
      using namespace container::operators;
      const signed_idx_t relidx = idx - result_lobound;

      new (&(result[idx]))
          typename TR::value_type(tensor_op<typename TR::value_type>(
              op, tensor1[relidx + signed_idx_t(tensor1.range().lobound())],
              (tensors[relidx + signed_idx_t(tensors.range().lobound())])...));
    }
  }
}

// -------------------------------------------------------------------------
// Reduction kernels for argument tensors

/// Reduction operation for contiguous tensors

/// Perform an element-wise reduction of the tensors by
/// executing <tt>join_op(result, reduce_op(result, &tensor1[i],
/// &tensors[i]...))</tt> for each \c i in the index range of \c tensor1 .
/// \c result is initialized to \c identity . If `HAVE_INTEL_TBB` is defined,
/// the reduction will be executed in an undefined order, otherwise will
/// execute in the order of increasing \c i .
/// \tparam ReduceOp The element-wise reduction operation type
/// \tparam JoinOp The result operation type
/// \tparam Identity A type that can be used as an argument to ReduceOp
/// \tparam T1 The first argument tensor type
/// \tparam Ts The argument tensor types
/// \param reduce_op The element-wise reduction operation
/// \param identity The initial value for the reduction and the result
/// \param tensor1 The first tensor to be reduced
/// \param tensors The other tensors to be reduced
/// \return The reduced value of the tensor(s)
template <
    typename ReduceOp, typename JoinOp, typename Identity, typename T1,
    typename... Ts,
    typename std::enable_if_t<
        is_tensor<T1, Ts...>::value && is_contiguous_tensor<T1, Ts...>::value &&
        !is_reduce_op_v<std::decay_t<ReduceOp>, std::decay_t<Identity>,
                        std::decay_t<T1>, std::decay_t<Ts>...>>* = nullptr>
auto tensor_reduce(ReduceOp&& reduce_op, JoinOp&& join_op, Identity&& identity,
                   const T1& tensor1, const Ts&... tensors) {
  TA_ASSERT(!empty(tensor1, tensors...));
  TA_ASSERT(is_range_set_congruent(tensor1, tensors...));

  const auto volume = [&tensor1]() {
    if constexpr (detail::has_total_size_v<T1>)
      return tensor1.total_size();
    else
      return tensor1.size();
  }();

  auto init = std::forward<Identity>(identity);
  math::reduce_op(std::forward<ReduceOp>(reduce_op),
                  std::forward<JoinOp>(join_op), init, volume, init,
                  tensor1.data(), tensors.data()...);

  return init;
}

/// Reduction operation for tensors

/// Perform tensor-wise reduction of the tensors by
/// executing <tt>reduce_op(result, &tensor1, &tensors...)</tt>.
/// \c result is initialized to \c identity .
/// \tparam ReduceOp The tensor-wise reduction operation type
/// \tparam JoinOp The result operation type
/// \tparam Scalar A scalar type
/// \tparam T1 The first argument tensor type
/// \tparam Ts The argument tensor types
/// \param reduce_op The element-wise reduction operation
/// \param identity The initial value for the reduction and the result
/// \param tensor1 The first tensor to be reduced
/// \param tensors The other tensors to be reduced
/// \return The reduced value of the tensor(s)
template <
    typename ReduceOp, typename JoinOp, typename Scalar, typename T1,
    typename... Ts,
    typename std::enable_if_t<
        is_tensor<T1, Ts...>::value && is_contiguous_tensor<T1, Ts...>::value &&
        is_reduce_op_v<std::decay_t<ReduceOp>, std::decay_t<Scalar>,
                       std::decay_t<T1>, std::decay_t<Ts>...>>* = nullptr>
auto tensor_reduce(ReduceOp&& reduce_op, JoinOp&& join_op, Scalar identity,
                   const T1& tensor1, const Ts&... tensors) {
  reduce_op(identity, &tensor1, &tensors...);
  return identity;
}

/// Reduction operation for contiguous tensors of tensors

/// Perform reduction of the tensor-of-tensors' elements by
/// executing <tt>join_op(result, reduce_op(tensor1[i], tensors[i]...))</tt> for
/// each \c i in the index range of \c tensor1 . \c result is initialized to
/// \c identity . This will execute serially, in the order of increasing
/// \c i (each element's reduction can however be executed in parallel,
/// depending on the element type).
/// \tparam ReduceOp The tensor-wise reduction operation type
/// \tparam JoinOp The result operation type
/// \tparam Scalar A scalar type
/// \tparam T1 The first argument tensor type
/// \tparam Ts The argument tensor types
/// \param reduce_op The element-wise reduction operation
/// \param join_op The result join operation
/// \param identity The initial value for the reduction and the result
/// \param tensor1 The first tensor to be reduced
/// \param tensors The other tensors to be reduced
/// \return The reduced value of the tensor(s)
template <typename ReduceOp, typename JoinOp, typename Identity, typename T1,
          typename... Ts,
          typename std::enable_if<
              is_tensor_of_tensor<T1, Ts...>::value &&
              is_contiguous_tensor<T1, Ts...>::value>::type* = nullptr>
auto tensor_reduce(ReduceOp&& reduce_op, JoinOp&& join_op,
                   const Identity& identity, const T1& tensor1,
                   const Ts&... tensors) {
  TA_ASSERT(!empty(tensor1, tensors...));
  TA_ASSERT(is_range_set_congruent(tensor1, tensors...));

  const auto volume = [&tensor1]() {
    if constexpr (detail::has_total_size_v<T1>)
      return tensor1.total_size();
    else
      return tensor1.size();
  }();

  auto result = identity;
  for (std::remove_cv_t<decltype(volume)> ord = 0ul; ord < volume; ++ord) {
    auto temp = tensor_reduce(reduce_op, join_op, identity, tensor1.data()[ord],
                              tensors.data()[ord]...);
    join_op(result, temp);
  }

  return result;
}

/// Reduction operation for non-contiguous tensors

/// Perform an element-wise reduction of the tensors by
/// executing <tt>join_op(result, reduce_op(tensor1[i], tensors[i]...))</tt> for
/// each \c i in the index range of \c tensor1 . \c result is initialized to
/// \c identity . This will execute serially, in the order of increasing
/// \c i (each element-wise reduction can however be executed in parallel,
/// depending on the element type).
/// \tparam ReduceOp The element-wise reduction operation type
/// \tparam JoinOp The result operation type
/// \tparam Scalar A scalar type
/// \tparam T1 The first argument tensor type
/// \tparam Ts The argument tensor types
/// \param reduce_op The element-wise reduction operation
/// \param join_op The result join operation
/// \param identity The initial value for the reduction and the result
/// \param tensor1 The first tensor to be reduced
/// \param tensors The other tensors to be reduced
/// \return The reduced value of the tensor(s)
template <typename ReduceOp, typename JoinOp, typename Identity, typename T1,
          typename... Ts,
          typename std::enable_if<
              is_tensor<T1, Ts...>::value &&
              !is_contiguous_tensor<T1, Ts...>::value>::type* = nullptr>
auto tensor_reduce(ReduceOp&& reduce_op, JoinOp&& join_op,
                   const Identity& identity, const T1& tensor1,
                   const Ts&... tensors) {
  TA_ASSERT(!empty(tensor1, tensors...));
  TA_ASSERT(is_range_set_congruent(tensor1, tensors...));

  const auto volume = [&tensor1]() {
    if constexpr (detail::has_total_size_v<T1>)
      return tensor1.total_size();
    else
      return tensor1.size();
  }();

  auto result = identity;
  if constexpr (detail::has_member_function_data_anyreturn_v<T1> &&
                (detail::has_member_function_data_anyreturn_v<Ts> && ...)) {
    const auto stride = inner_size(tensor1, tensors...);
    for (std::decay_t<decltype(volume)> ord = 0ul; ord < volume;
         ord += stride) {
      auto temp = identity;
      math::reduce_op(reduce_op, join_op, identity, stride, temp,
                      tensor1.data() + tensor1.range().ordinal(ord),
                      (tensors.data() + tensors.range().ordinal(ord))...);
      join_op(result, temp);
    }
  } else {  // if 1+ tensor lacks data() must iterate over individual elements
    // TA_ASSERT(tensor1.nbatch() == 1); // todo: assert the same for the
    // remaining tensors
    auto& t1_rng = tensor1.range();
    using signed_idx_t = Range::index_difference_type;
    auto t1_lobound = signed_idx_t(t1_rng.lobound());
    for (auto&& idx : t1_rng) {
      using namespace container::operators;
      signed_idx_t relidx = idx - t1_lobound;
      reduce_op(result, tensor1[idx],
                (tensors[idx - t1_lobound +
                         signed_idx_t(tensors.range().lobound())])...);
    }
  }

  return result;
}

/// Reduction operation for non-contiguous tensors of tensors.

/// Perform an element-wise reduction of the tensors by
/// executing <tt>join_op(result, reduce_op(tensor1[i], tensors[i]...))</tt> for
/// each \c i in the index range of \c tensor1 . \c result is initialized to
/// \c identity . This will execute serially, in the order of increasing
/// \c i (each element-wise reduction can however be executed in parallel,
/// depending on the element type).
/// \tparam ReduceOp The element-wise reduction operation type
/// \tparam JoinOp The result operation type
/// \tparam Scalar A scalar type
/// \tparam T1 The first argument tensor type
/// \tparam Ts The argument tensor types
/// \param reduce_op The element-wise reduction operation
/// \param join_op The result join operation
/// \param identity The initial value for the reduction and the result
/// \param tensor1 The first tensor to be reduced
/// \param tensors The other tensors to be reduced
/// \return The reduced value of the tensor(s)
template <typename ReduceOp, typename JoinOp, typename Scalar, typename T1,
          typename... Ts,
          typename std::enable_if<
              is_tensor_of_tensor<T1, Ts...>::value &&
              !is_contiguous_tensor<T1, Ts...>::value>::type* = nullptr>
Scalar tensor_reduce(ReduceOp&& reduce_op, JoinOp&& join_op,
                     const Scalar identity, const T1& tensor1,
                     const Ts&... tensors) {
  TA_ASSERT(!empty(tensor1, tensors...));
  TA_ASSERT(is_range_set_congruent(tensor1, tensors...));
  // TA_ASSERT(tensor1.nbatch() == 1); // todo: assert the same for the
  // remaining tensors

  const auto volume = [&tensor1]() {
    if constexpr (detail::has_total_size_v<T1>)
      return tensor1.total_size();
    else
      return tensor1.size();
  }();

  Scalar result = identity;

  if constexpr (detail::has_member_function_data_anyreturn_v<T1> &&
                (detail::has_member_function_data_anyreturn_v<Ts> && ...)) {
    const auto stride = inner_size(tensor1, tensors...);
    auto tensor_reduce_range =
        [&reduce_op, &join_op, &identity, stride](
            Scalar& MADNESS_RESTRICT result,
            typename T1::const_pointer MADNESS_RESTRICT const tensor1_data,
            typename Ts::const_pointer MADNESS_RESTRICT const... tensors_data) {
          for (std::remove_cv_t<decltype(volume)> i = 0ul; i < stride; ++i) {
            Scalar temp = tensor_reduce(reduce_op, join_op, identity,
                                        tensor1_data[i], tensors_data[i]...);
            join_op(result, temp);
          }
        };

    for (std::decay_t<decltype(volume)> ord = 0ul; ord < volume;
         ord += stride) {
      Scalar temp = tensor_reduce_range(
          result, tensor1.data() + tensor1.range().ordinal(ord),
          (tensors.data() + tensors.range().ordinal(ord))...);
      join_op(result, temp);
    }
  } else {  // if 1+ tensor lacks data() must iterate over individual elements
    auto& t1_rng = tensor1.range();
    using signed_idx_t = Range::index_difference_type;
    auto t1_lobound = signed_idx_t(t1_rng.lobound());
    for (auto&& idx : t1_rng) {
      using namespace container::operators;
      signed_idx_t relidx = idx - t1_lobound;

      Scalar temp =
          tensor_reduce(reduce_op, join_op, identity, tensor1[idx],
                        (tensors[idx - t1_lobound +
                                 signed_idx_t(tensors.range().lobound())])...);
      join_op(result, temp);
    }
  }

  return result;
}

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TENSOR_KENERLS_H__INCLUDED
