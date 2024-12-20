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

#include <TiledArray/einsum/index.h>
#include <TiledArray/math/gemm_helper.h>
#include <TiledArray/tensor/permute.h>
#include <TiledArray/tensor/utility.h>
#include <TiledArray/util/vector.h>

namespace TiledArray {

template <typename, typename>
class Tensor;

namespace detail {

// -------------------------------------------------------------------------
// Tensor GEMM

/// Contract two tensors

/// GEMM is limited to matrix like contractions. For example, the following
/// contractions are supported:
/// \code
/// C[a,b] = A[a,i,j] * B[i,j,b]
/// C[a,b] = A[a,i,j] * B[b,i,j]
/// C[a,b] = A[i,j,a] * B[i,j,b]
/// C[a,b] = A[i,j,a] * B[b,i,j]
///
/// C[a,b,c,d] = A[a,b,i,j] * B[i,j,c,d]
/// C[a,b,c,d] = A[a,b,i,j] * B[c,d,i,j]
/// C[a,b,c,d] = A[i,j,a,b] * B[i,j,c,d]
/// C[a,b,c,d] = A[i,j,a,b] * B[c,d,i,j]
/// \endcode
/// Notice that in the above contractions, the inner and outer indices of
/// the arguments for exactly two contiguous groups in each tensor and that
/// each group is in the same order in all tensors. That is, the indices of
/// the tensors must fit the one of the following patterns:
/// \code
/// C[M...,N...] = A[M...,K...] * B[K...,N...]
/// C[M...,N...] = A[M...,K...] * B[N...,K...]
/// C[M...,N...] = A[K...,M...] * B[K...,N...]
/// C[M...,N...] = A[K...,M...] * B[N...,K...]
/// \endcode
/// This allows use of optimized BLAS functions to evaluate tensor
/// contractions. Tensor contractions that do not fit this pattern require
/// one or more tensor permutation so that the tensors fit the required
/// pattern.
/// \tparam U The left-hand tensor element type
/// \tparam AU The left-hand tensor allocator type
/// \tparam V The right-hand tensor element type
/// \tparam AV The right-hand tensor allocator type
/// \tparam W The type of the scaling factor
/// \param left The left-hand tensor that will be contracted
/// \param right The right-hand tensor that will be contracted
/// \param factor The contraction result will be scaling by this value, then
/// accumulated into \c this \param gemm_helper The *GEMM operation meta data
/// \return A reference to \c this
/// \note if this is uninitialized, i.e., if \c this->empty()==true will
/// this is equivalent to
/// \code
///   return (*this = left.gemm(right, factor, gemm_helper));
/// \endcode
template <typename Alpha, typename... As, typename... Bs, typename Beta,
          typename... Cs>
void gemm(Alpha alpha, const Tensor<As...>& A, const Tensor<Bs...>& B,
          Beta beta, Tensor<Cs...>& C, const math::GemmHelper& gemm_helper) {
  static_assert(!detail::is_tensor_of_tensor_v<Tensor<As...>, Tensor<Bs...>,
                                               Tensor<Cs...>>,
                "TA::Tensor<T,Allocator>::gemm without custom element op is "
                "only applicable to "
                "plain tensors");
  {
    // Check that tensor C is not empty and has the correct rank
    TA_ASSERT(!C.empty());
    TA_ASSERT(C.range().rank() == gemm_helper.result_rank());

    // Check that the arguments are not empty and have the correct ranks
    TA_ASSERT(!A.empty());
    TA_ASSERT(A.range().rank() == gemm_helper.left_rank());
    TA_ASSERT(!B.empty());
    TA_ASSERT(B.range().rank() == gemm_helper.right_rank());

    TA_ASSERT(A.nbatch() == 1);
    TA_ASSERT(B.nbatch() == 1);
    TA_ASSERT(C.nbatch() == 1);

    // Check that the outer dimensions of left match the corresponding
    // dimensions in result
    TA_ASSERT(gemm_helper.left_result_congruent(A.range().extent_data(),
                                                C.range().extent_data()));
    TA_ASSERT(ignore_tile_position() ||
              gemm_helper.left_result_congruent(A.range().lobound_data(),
                                                C.range().lobound_data()));
    TA_ASSERT(ignore_tile_position() ||
              gemm_helper.left_result_congruent(A.range().upbound_data(),
                                                C.range().upbound_data()));

    // Check that the outer dimensions of right match the corresponding
    // dimensions in result
    TA_ASSERT(gemm_helper.right_result_congruent(B.range().extent_data(),
                                                 C.range().extent_data()));
    TA_ASSERT(ignore_tile_position() ||
              gemm_helper.right_result_congruent(B.range().lobound_data(),
                                                 C.range().lobound_data()));
    TA_ASSERT(ignore_tile_position() ||
              gemm_helper.right_result_congruent(B.range().upbound_data(),
                                                 C.range().upbound_data()));

    // Check that the inner dimensions of left and right match
    TA_ASSERT(gemm_helper.left_right_congruent(A.range().extent_data(),
                                               B.range().extent_data()));
    TA_ASSERT(ignore_tile_position() ||
              gemm_helper.left_right_congruent(A.range().lobound_data(),
                                               B.range().lobound_data()));
    TA_ASSERT(ignore_tile_position() ||
              gemm_helper.left_right_congruent(A.range().upbound_data(),
                                               B.range().upbound_data()));

    // Compute gemm dimensions
    using integer = TiledArray::math::blas::integer;
    integer m, n, k;
    gemm_helper.compute_matrix_sizes(m, n, k, A.range(), B.range());

    // Get the leading dimension for left and right matrices.
    const integer lda = std::max(
        integer{1},
        (gemm_helper.left_op() == TiledArray::math::blas::NoTranspose ? k : m));
    const integer ldb = std::max(
        integer{1},
        (gemm_helper.right_op() == TiledArray::math::blas::NoTranspose ? n
                                                                       : k));

    // may need to split gemm into multiply + accumulate for tracing purposes
#ifdef TA_ENABLE_TILE_OPS_LOGGING
    {
      using numeric_type = typename Tensor<Cs...>::numeric_type;
      using T = numeric_type;
      const bool twostep =
          TiledArray::TileOpsLogger<T>::get_instance().gemm &&
          TiledArray::TileOpsLogger<T>::get_instance().gemm_print_contributions;
      std::unique_ptr<T[]> data_copy;
      size_t tile_volume;
      if (twostep) {
        tile_volume = C.range().volume();
        data_copy = std::make_unique<T[]>(tile_volume);
        std::copy(C.data(), C.data() + tile_volume, data_copy.get());
      }
      non_distributed::gemm(gemm_helper.left_op(), gemm_helper.right_op(), m, n,
                            k, alpha, A.data(), lda, B.data(), ldb,
                            twostep ? numeric_type(0) : beta, C.data(), n);

      if (TiledArray::TileOpsLogger<T>::get_instance_ptr() != nullptr &&
          TiledArray::TileOpsLogger<T>::get_instance().gemm) {
        auto& logger = TiledArray::TileOpsLogger<T>::get_instance();
        auto apply = [](auto& fnptr, const Range& arg) {
          return fnptr ? fnptr(arg) : arg;
        };
        auto tformed_left_range =
            apply(logger.gemm_left_range_transform, A.range());
        auto tformed_right_range =
            apply(logger.gemm_right_range_transform, B.range());
        auto tformed_result_range =
            apply(logger.gemm_result_range_transform, C.range());
        if ((!logger.gemm_result_range_filter ||
             logger.gemm_result_range_filter(tformed_result_range)) &&
            (!logger.gemm_left_range_filter ||
             logger.gemm_left_range_filter(tformed_left_range)) &&
            (!logger.gemm_right_range_filter ||
             logger.gemm_right_range_filter(tformed_right_range))) {
          logger << "TA::Tensor::gemm+: left=" << tformed_left_range
                 << " right=" << tformed_right_range
                 << " result=" << tformed_result_range << std::endl;
          if (TiledArray::TileOpsLogger<T>::get_instance()
                  .gemm_print_contributions) {
            if (!TiledArray::TileOpsLogger<T>::get_instance()
                     .gemm_printer) {  // default printer
              // must use custom printer if result's range transformed
              if (!logger.gemm_result_range_transform)
                logger << C << std::endl;
              else
                logger << make_map(C.data(), tformed_result_range) << std::endl;
            } else {
              TiledArray::TileOpsLogger<T>::get_instance().gemm_printer(
                  *logger.log, tformed_left_range, A.data(),
                  tformed_right_range, B.data(), tformed_right_range, C.data(),
                  C.nbatch());
            }
          }
        }
      }

      if (twostep) {
        for (size_t v = 0; v != tile_volume; ++v) {
          C.data()[v] += data_copy[v];
        }
      }
    }
#else   // TA_ENABLE_TILE_OPS_LOGGING
    const integer ldc = std::max(integer{1}, n);
    math::blas::gemm(gemm_helper.left_op(), gemm_helper.right_op(), m, n, k,
                     alpha, A.data(), lda, B.data(), ldb, beta, C.data(), ldc);
#endif  // TA_ENABLE_TILE_OPS_LOGGING
  }
}

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

  auto volume = result.total_size();
  for (decltype(volume) ord = 0; ord < volume; ++ord) {
    if constexpr (is_tensor_of_tensor_v<TR>)
      if (result.data()[ord].range().volume() == 0) continue;
    if constexpr (is_tensor_of_tensor_v<Ts...>)
      if (((tensors.data()[ord].range().volume() == 0) || ...)) continue;
    if constexpr (std::is_invocable_r_v<void, Op, typename TR::value_type&,
                                        typename Ts::value_type...>)
      op(result.data()[ord], tensors.data()[ord]...);
    else
      inplace_tensor_op(op, result.data()[ord], tensors.data()[ord]...);
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
          for (decltype(result.range().volume()) i = 0ul; i < stride; ++i) {
            if constexpr (std::is_invocable_v<
                              std::remove_reference_t<Op>,
                              typename std::remove_reference_t<TR>::value_type&,
                              typename std::remove_reference_t<
                                  Ts>::value_type const&...>) {
              std::forward<Op>(op)(result_data[i], tensors_data[i]...);
            } else {
              inplace_tensor_op(op, result_data[i], tensors_data[i]...);
            }
          }
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
    typename std::enable_if<
        (is_nested_tensor<TR, Ts...>::value && !is_tensor<TR, Ts...>::value) &&
        is_contiguous_tensor<TR, Ts...>::value>::type* = nullptr>
inline void tensor_init(Op&& op, TR& result, const Ts&... tensors) {
  TA_ASSERT(!empty(result, tensors...));
  TA_ASSERT(is_range_set_congruent(result, tensors...));

  if constexpr (std::is_invocable_r_v<TR, Op, const Ts&...>) {
    result = std::forward<Op>(op)(tensors...);
  } else {
    const auto volume = result.total_size();
    for (std::remove_cv_t<decltype(volume)> ord = 0ul; ord < volume; ++ord) {
      new (result.data() + ord) typename TR::value_type(
          tensor_op<typename TR::value_type>(op, (*(tensors.data() + ord))...));
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
    if (tensor1.data()[ord].range().volume() == 0 ||
        ((tensors.data()[ord].range().volume() == 0) || ...))
      continue;
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

///
/// todo: constraint ResultTensorAllocator type so that non-sensical Allocators
/// are prohibited
///
template <typename ResultTensorAllocator = void, typename TensorA,
          typename TensorB, typename Annot,
          typename = std::enable_if_t<is_tensor_v<TensorA, TensorB> &&
                                      is_annotation_v<Annot>>>
auto tensor_contract(TensorA const& A, Annot const& aA, TensorB const& B,
                     Annot const& aB, Annot const& aC) {
  using Result = result_tensor_t<std::multiplies<>, TensorA, TensorB,
                                 ResultTensorAllocator>;

  using Indices = ::Einsum::index::Index<typename Annot::value_type>;
  using Permutation = ::Einsum::index::Permutation;
  using ::Einsum::index::permutation;

  // Check that the ranks of the tensors match that of the annotation.
  TA_ASSERT(A.range().rank() == aA.size());
  TA_ASSERT(B.range().rank() == aB.size());

  struct {
    Indices  //
        A,   // indices of A
        B,   // indices of B
        C,   // indices of C (target indices)
        h,   // Hadamard indices (aA intersection aB intersection aC)
        e,   // external indices (aA symmetric difference aB)
        i;   // internal indices ((aA intersection aB) set difference aC)
  } const indices{aA,
                  aB,
                  aC,
                  (indices.A & indices.B & indices.C),
                  (indices.A ^ indices.B),
                  ((indices.A & indices.B) - indices.h)};

  TA_ASSERT(!indices.h && "Hadamard indices not supported");
  TA_ASSERT(indices.e && "Dot product not supported");

  struct {
    Indices A, B, C;
  } const blas_layout{(indices.A - indices.B) | indices.i,
                      indices.i | (indices.B - indices.A), indices.e};

  struct {
    Permutation A, B, C;
  } const perm{permutation(indices.A, blas_layout.A),
               permutation(indices.B, blas_layout.B),
               permutation(indices.C, blas_layout.C)};

  struct {
    bool A, B, C;
  } const do_perm{indices.A != blas_layout.A, indices.B != blas_layout.B,
                  indices.C != blas_layout.C};

  math::GemmHelper gemm_helper{blas::Op::NoTrans, blas::Op::NoTrans,
                               static_cast<unsigned int>(indices.e.size()),
                               static_cast<unsigned int>(indices.A.size()),
                               static_cast<unsigned int>(indices.B.size())};

  // initialize result with the correct extents
  Result result;
  {
    using Index = typename Indices::value_type;
    using Extent = std::remove_cv_t<
        typename decltype(std::declval<Range>().extent())::value_type>;
    using ExtentMap = ::Einsum::index::IndexMap<Index, Extent>;

    // Map tensor indices to their extents.
    // Note that whether the contracting indices have matching extents is
    // implicitly checked here by the pipe(|) operator on ExtentMap.

    ExtentMap extent = (ExtentMap{indices.A, A.range().extent()} |
                        ExtentMap{indices.B, B.range().extent()});

    container::vector<Extent> rng;
    rng.reserve(indices.e.size());
    for (auto&& ix : indices.e) {
      // assuming ix _exists_ in extent
      rng.emplace_back(extent[ix]);
    }
    result = Result{TA::Range(rng)};
  }

  using Numeric = typename Result::numeric_type;

  // call gemm
  gemm(Numeric{1},                         //
       do_perm.A ? A.permute(perm.A) : A,  //
       do_perm.B ? B.permute(perm.B) : B,  //
       Numeric{0}, result, gemm_helper);

  return do_perm.C ? result.permute(perm.C.inv()) : result;
}

template <typename TensorA, typename TensorB, typename Annot,
          typename = std::enable_if_t<is_tensor_v<TensorA, TensorB> &&
                                      is_annotation_v<Annot>>>
auto tensor_hadamard(TensorA const& A, Annot const& aA, TensorB const& B,
                     Annot const& aB, Annot const& aC) {
  using ::Einsum::index::Permutation;
  using ::Einsum::index::permutation;
  using Indices = ::Einsum::index::Index<typename Annot::value_type>;

  struct {
    Permutation  //
        AB,      // permutes A to B
        AC,      // permutes A to C
        BC;      // permutes B to C
  } const perm{permutation(Indices(aA), Indices(aB)),
               permutation(Indices(aA), Indices(aC)),
               permutation(Indices(aB), Indices(aC))};

  struct {
    bool no_perm, perm_to_c, perm_a, perm_b;
  } const do_this{
      perm.AB.is_identity() && perm.AC.is_identity() && perm.BC.is_identity(),
      perm.AB.is_identity(),  //
      perm.BC.is_identity(),  //
      perm.AC.is_identity()};

  if (do_this.no_perm) {
    return A.mult(B);
  } else if (do_this.perm_to_c) {
    return A.mult(B, perm.AC);
  } else if (do_this.perm_a) {
    auto pA = A.permute(perm.AC);
    pA.mult_to(B);
    return pA;
  } else if (do_this.perm_b) {
    auto pB = B.permute(perm.BC);
    pB.mult_to(A);
    return pB;
  } else {
    auto pA = A.permute(perm.AC);
    return pA.mult_to(B.permute(perm.BC));
  }
}

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TENSOR_KENERLS_H__INCLUDED
