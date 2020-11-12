/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 *  contract_reduce.h
 *  Oct 9, 2013
 *
 */

#ifndef TILEDARRAY_TILE_OP_CONTRACT_REDUCE_H__INCLUDED
#define TILEDARRAY_TILE_OP_CONTRACT_REDUCE_H__INCLUDED

#include <TiledArray/math/gemm_helper.h>
#include <TiledArray/permutation.h>
#include <TiledArray/tensor/complex.h>
#include <TiledArray/tile_op/tile_interface.h>
#include <TiledArray/util/function.h>
#include "../tile_interface/add.h"
#include "../tile_interface/permute.h"

namespace TiledArray {
namespace detail {

/// Contract and (sum) reduce base

/// This implementation class is used to provide shallow copy semantics for
/// ContractReduce. It encodes a binary tensor contraction mapped to a GEMM, as
/// well as the sum reduction and post-processing.
/// \tparam Result The result
/// tile type
/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand
/// tile type
/// \tparam Scalar The scaling factor type
template <typename Result, typename Left, typename Right, typename Scalar>
class ContractReduceBase {
 public:
  typedef ContractReduceBase<Result, Left, Right, Scalar>
      ContractReduceBase_;                    ///< This class type
  typedef const Left& first_argument_type;    ///< The left tile type
  typedef const Right& second_argument_type;  ///< The right tile type
  typedef Result result_type;                 ///< The result type
  typedef Scalar scalar_type;                 ///< The scaling factor type

  using left_value_type = typename Left::value_type;
  using right_value_type = typename Right::value_type;
  using result_value_type = typename Result::value_type;
  using element_mult_op_type = result_value_type(const left_value_type&,
                                                 const right_value_type&);

  static_assert(
      TiledArray::detail::is_tensor_v<left_value_type> ==
              TiledArray::detail::is_tensor_v<right_value_type> &&
          TiledArray::detail::is_tensor_v<left_value_type> ==
              TiledArray::detail::is_tensor_v<result_value_type>,
      "ContractReduce can only handle plain tensors or nested tensors "
      "(tensors-of-tensors); mixed contractions are not supported");
  static constexpr bool plain_tensors =
      !(TiledArray::detail::is_tensor_v<left_value_type> &&
        TiledArray::detail::is_tensor_v<right_value_type> &&
        TiledArray::detail::is_tensor_v<result_value_type>);

 private:
  struct Impl {
    template <
        typename Perm = BipartitePermutation,
        typename ElementMultOp = TiledArray::function_ref<element_mult_op_type>,
        typename = std::enable_if_t<
            TiledArray::detail::is_permutation_v<Perm> &&
            std::is_invocable_r_v<
                result_value_type, std::remove_reference_t<ElementMultOp>,
                const left_value_type&, const right_value_type&>>>
    Impl(const madness::cblas::CBLAS_TRANSPOSE left_op,
         const madness::cblas::CBLAS_TRANSPOSE right_op,
         const scalar_type alpha, const unsigned int result_rank,
         const unsigned int left_rank, const unsigned int right_rank,
         const Perm& perm = {}, ElementMultOp&& elem_mult_op = {})
        : gemm_helper_(left_op, right_op, result_rank, left_rank, right_rank),
          alpha_(alpha),
          perm_(perm),
          element_mult_op_(std::forward<ElementMultOp>(elem_mult_op)) {}

    math::GemmHelper gemm_helper_;  ///< Gemm helper object
    scalar_type alpha_;  ///< Scaling factor applied to the contraction of
                         ///< the left- and right-hand arguments
    BipartitePermutation perm_;  ///< Permutation that is applied to the final
                                 ///< result tensor

    /// type-erased reference to custom element multiply op
    /// \note the lifetime is managed by the callee!
    TiledArray::function_ref<element_mult_op_type> element_mult_op_;
  };

  std::shared_ptr<Impl> pimpl_;

 public:
  // Compiler generated defaults are fine

  ContractReduceBase() = default;
  ContractReduceBase(const ContractReduceBase_&) = default;
  ContractReduceBase(ContractReduceBase_&&) = default;
  ~ContractReduceBase() = default;
  ContractReduceBase_& operator=(const ContractReduceBase_&) = default;
  ContractReduceBase_& operator=(ContractReduceBase_&&) = default;

  /// Construct contract/reduce functor

  /// \tparam Perm a permutation type
  /// \tparam ElementOp a callable with signature element_mult_op_type
  /// \param left_op The left-hand BLAS matrix operation
  /// \param right_op The right-hand BLAS matrix operation
  /// \param alpha The scaling factor applied to the contracted tiles
  /// \param result_rank The rank of the result tensor
  /// \param left_rank The rank of the left-hand tensor
  /// \param right_rank The rank of the right-hand tensor
  /// \param perm The permutation to be applied to the result tensor
  ///        (default = no permute)
  /// \param element_mult_op The element multiply op
  template <
      typename Perm = BipartitePermutation,
      typename ElementMultOp = TiledArray::function_ref<element_mult_op_type>,
      typename = std::enable_if_t<
          TiledArray::detail::is_permutation_v<Perm> &&
          std::is_invocable_r_v<
              result_value_type, std::remove_reference_t<ElementMultOp>,
              const left_value_type&, const right_value_type&>>>
  ContractReduceBase(const madness::cblas::CBLAS_TRANSPOSE left_op,
                     const madness::cblas::CBLAS_TRANSPOSE right_op,
                     const scalar_type alpha, const unsigned int result_rank,
                     const unsigned int left_rank,
                     const unsigned int right_rank, const Perm& perm = {},
                     ElementMultOp&& elem_mult_op = {})
      : pimpl_(std::make_shared<Impl>(
            left_op, right_op, alpha, result_rank, left_rank, right_rank, perm,
            std::forward<ElementMultOp>(elem_mult_op))) {}

  /// Gemm meta data accessor

  /// \return A const reference to the gemm helper object
  const math::GemmHelper& gemm_helper() const {
    TA_ASSERT(pimpl_);
    return pimpl_->gemm_helper_;
  }

  /// Permutation accessor

  /// \return A const reference to the permutation for this operation
  const BipartitePermutation& perm() const {
    TA_ASSERT(pimpl_);
    return pimpl_->perm_;
  }

  /// Scaling factor accessor

  /// \return The scaling factor for this operation
  scalar_type factor() const {
    TA_ASSERT(pimpl_);
    return pimpl_->alpha_;
  }

  /// Element multiply op accessor

  /// \return A const reference to the element multiply op function_ref
  const auto& element_mult_op() const {
    TA_ASSERT(pimpl_);
    return pimpl_->element_mult_op_;
  }

  //-------------- these are only used for unit tests -----------------

  /// Compute the number of contracted ranks

  /// \return The number of ranks that are summed by this operation
  unsigned int num_contract_ranks() const {
    TA_ASSERT(pimpl_);
    return pimpl_->gemm_helper_.num_contract_ranks();
  }

  /// Result rank accessor

  /// \return The rank of the result tile
  unsigned int result_rank() const {
    TA_ASSERT(pimpl_);
    return pimpl_->gemm_helper_.result_rank();
  }

  /// Left-hand argument rank accessor

  /// \return The rank of the left-hand tile
  unsigned int left_rank() const {
    TA_ASSERT(pimpl_);
    return pimpl_->gemm_helper_.left_rank();
  }

  /// Right-hand argument rank accessor

  /// \return The rank of the right-hand tile
  unsigned int right_rank() const {
    TA_ASSERT(pimpl_);
    return pimpl_->gemm_helper_.right_rank();
  }

};  // class ContractReduceBase

/// Contract and (sum) reduce operation

/// This encodes a binary tensor contraction mapped to a GEMM, as well as the
/// sum reduction and post-processing.
/// \tparam Result The result tile type
/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Scalar The scaling factor type
template <typename Result, typename Left, typename Right, typename Scalar>
class ContractReduce : public ContractReduceBase<Result, Left, Right, Scalar> {
 public:
  typedef ContractReduce<Result, Left, Right, Scalar>
      ContractReduce_;  ///< This class type
  typedef ContractReduceBase<Result, Left, Right, Scalar>
      ContractReduceBase_;  ///< This class type
  typedef typename ContractReduceBase_::first_argument_type
      first_argument_type;  ///< The left tile type
  typedef typename ContractReduceBase_::second_argument_type
      second_argument_type;    ///< The right tile type
  typedef Result result_type;  ///< The result tile type.
  typedef Scalar scalar_type;

  using typename ContractReduceBase_::element_mult_op_type;
  using typename ContractReduceBase_::left_value_type;
  using typename ContractReduceBase_::result_value_type;
  using typename ContractReduceBase_::right_value_type;

  // Compiler generated defaults are fine. N.B. this is shallow-copy.

  ContractReduce() = default;
  ContractReduce(const ContractReduce_&) = default;
  ContractReduce(ContractReduce_&&) = default;
  ~ContractReduce() = default;
  ContractReduce_& operator=(const ContractReduce_&) = default;
  ContractReduce_& operator=(ContractReduce_&&) = default;

  /// Construct contract/reduce functor

  /// \tparam Perm a permutation type
  /// \tparam ElementOp a callable with signature element_mult_op_type
  /// \param left_op The left-hand BLAS matrix operation
  /// \param right_op The right-hand BLAS matrix operation
  /// \param alpha The scaling factor applied to the contracted tiles
  /// \param result_rank The rank of the result tensor
  /// \param left_rank The rank of the left-hand tensor
  /// \param right_rank The rank of the right-hand tensor
  /// \param perm The permutation to be applied to the result tensor
  ///        (default = no permute)
  /// \param element_mult_op The element multiply op
  template <
      typename Perm = BipartitePermutation,
      typename ElementMultOp = TiledArray::function_ref<element_mult_op_type>,
      typename = std::enable_if_t<
          TiledArray::detail::is_permutation_v<Perm> &&
          std::is_invocable_r_v<
              result_value_type, std::remove_reference_t<ElementMultOp>,
              const left_value_type&, const right_value_type&>>>
  ContractReduce(const madness::cblas::CBLAS_TRANSPOSE left_op,
                 const madness::cblas::CBLAS_TRANSPOSE right_op,
                 const scalar_type alpha, const unsigned int result_rank,
                 const unsigned int left_rank, const unsigned int right_rank,
                 const Perm& perm = {}, ElementMultOp&& elem_mult_op = {})
      : ContractReduceBase_(left_op, right_op, alpha, result_rank, left_rank,
                            right_rank, perm,
                            std::forward<ElementMultOp>(elem_mult_op)) {}

  /// Create a result type object

  /// Initialize a result object for subsequent reductions
  result_type operator()() const { return result_type(); }

  /// Post processing step
  result_type operator()(const result_type& temp) const {
    using TiledArray::empty;
    TA_ASSERT(!empty(temp));

    if (!ContractReduceBase_::perm()) return temp;

    TiledArray::Permute<result_type, result_type> permute;
    return permute(temp, ContractReduceBase_::perm());
  }

  /// Reduce two result objects

  /// Add \c arg to \c result .
  /// \param[in,out] result The result object that will be the reduction
  /// target
  /// \param[in] arg The argument that will be added to \c result
  void operator()(result_type& result, const result_type& arg) const {
    using TiledArray::add_to;
    add_to(result, arg);
  }

  /// Contract a pair of tiles and add to a target tile

  /// Contract \c left and \c right and add the result to \c result.
  /// \param[in,out] result The result object that will be the reduction
  /// target
  /// \param[in] left The left-hand tile to be contracted
  /// \param[in] right The right-hand tile to be contracted
  void operator()(result_type& result, const first_argument_type& left,
                  const second_argument_type& right) const {
    if constexpr (!ContractReduceBase_::plain_tensors) {
      TA_ASSERT(this->element_mult_op());
      // not yet implemented
      using TiledArray::empty;
      using TiledArray::gemm;
      if (empty(result))
        //        result = gemm(left, right, ContractReduceBase_::factor(),
        //                      ContractReduceBase_::gemm_helper(),
        //                      this->element_mult_op());
        ;
      else
        //        gemm(result, left, right, ContractReduceBase_::factor(),
        //             ContractReduceBase_::gemm_helper(),
        //             this->element_mult_op());
        ;
      abort();
    } else {  // plain tensors
      TA_ASSERT(!this->element_mult_op());
      using TiledArray::empty;
      using TiledArray::gemm;
      if (empty(result))
        result = gemm(left, right, ContractReduceBase_::factor(),
                      ContractReduceBase_::gemm_helper());
      else
        gemm(result, left, right, ContractReduceBase_::factor(),
             ContractReduceBase_::gemm_helper());
    }
  }

};  // class ContractReduce

/// Contract and (sum) reduce operation

/// This encodes a binary tensor contraction mapped to a GEMM, as well as the
/// sum reduction and post-processing.
/// \tparam Result The result tile type
/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
template <typename Result, typename Left, typename Right>
class ContractReduce<Result, Left, Right,
                     TiledArray::detail::ComplexConjugate<void>>
    : public ContractReduceBase<Result, Left, Right,
                                TiledArray::detail::ComplexConjugate<void>> {
 public:
  typedef ContractReduce<Result, Left, Right,
                         TiledArray::detail::ComplexConjugate<void>>
      ContractReduce_;  ///< This class type
  typedef ContractReduceBase<Result, Left, Right,
                             TiledArray::detail::ComplexConjugate<void>>
      ContractReduceBase_;  ///< This class type
  typedef typename ContractReduceBase_::first_argument_type
      first_argument_type;  ///< The left tile type
  typedef typename ContractReduceBase_::second_argument_type
      second_argument_type;  ///< The right tile type
  typedef decltype(gemm(std::declval<Left>(), std::declval<Right>(), 1,
                        std::declval<math::GemmHelper>()))
      result_type;  ///< The result tile type.
  typedef TiledArray::detail::ComplexConjugate<void> scalar_type;

  using typename ContractReduceBase_::element_mult_op_type;
  using typename ContractReduceBase_::left_value_type;
  using typename ContractReduceBase_::result_value_type;
  using typename ContractReduceBase_::right_value_type;

  // Compiler generated defaults are fine. N.B. This has shallow copy semantics.

  ContractReduce() = default;
  ContractReduce(const ContractReduce_&) = default;
  ContractReduce(ContractReduce_&&) = default;
  ~ContractReduce() = default;
  ContractReduce_& operator=(const ContractReduce_&) = default;
  ContractReduce_& operator=(ContractReduce_&&) = default;

  /// Construct contract/reduce functor

  /// \tparam ElementOp a callable with signature element_mult_op_type
  /// \param left_op The left-hand BLAS matrix operation
  /// \param right_op The right-hand BLAS matrix operation
  /// \param alpha The scaling factor applied to the contracted tiles
  /// \param result_rank The rank of the result tensor
  /// \param left_rank The rank of the left-hand tensor
  /// \param right_rank The rank of the right-hand tensor
  /// \param perm The permutation to be applied to the result tensor
  ///        (default = no permute)
  /// \param element_mult_op The element multiply op
  template <
      typename Perm = BipartitePermutation,
      typename ElementMultOp = TiledArray::function_ref<element_mult_op_type>,
      typename = std::enable_if_t<
          TiledArray::detail::is_permutation_v<Perm> &&
          std::is_invocable_r_v<
              result_value_type, std::remove_reference_t<ElementMultOp>,
              const left_value_type&, const right_value_type&>>>
  ContractReduce(const madness::cblas::CBLAS_TRANSPOSE left_op,
                 const madness::cblas::CBLAS_TRANSPOSE right_op,
                 const scalar_type alpha, const unsigned int result_rank,
                 const unsigned int left_rank, const unsigned int right_rank,
                 const Perm& perm = {}, ElementMultOp&& elem_mult_op = {})
      : ContractReduceBase_(left_op, right_op, alpha, result_rank, left_rank,
                            right_rank, perm,
                            std::forward<ElementMultOp>(elem_mult_op)) {}

  /// Create a result type object

  /// Initialize a result object for subsequent reductions
  result_type operator()() const { return result_type(); }

  /// Post processing step
  result_type operator()(result_type& temp) const {
    using TiledArray::empty;
    TA_ASSERT(!empty(temp));

    if (!ContractReduceBase_::perm()) {
      using TiledArray::conj_to;
      return conj_to(temp);
    }

    using TiledArray::conj;
    return conj(temp, ContractReduceBase_::perm());
  }

  /// Reduce two result objects

  /// Add \c arg to \c result .
  /// \param[in,out] result The result object that will be the reduction
  /// target
  /// \param[in] arg The argument that will be added to \c result
  void operator()(result_type& result, const result_type& arg) const {
    using TiledArray::add_to;
    add_to(result, arg);
  }

  /// Contract a pair of tiles and add to a target tile

  /// Contract \c left and \c right and add the result to \c result.
  /// \param[in,out] result The result object that will be the reduction
  /// target
  /// \param[in] left The left-hand tile to be contracted
  /// \param[in] right The right-hand tile to be contracted
  void operator()(result_type& result, const first_argument_type& left,
                  const second_argument_type& right) const {
    if constexpr (!ContractReduceBase_::plain_tensors) {
      TA_ASSERT(this->element_mult_op());
      // not yet implemented
      abort();
    } else {
      TA_ASSERT(!this->element_mult_op());
      using TiledArray::empty;
      using TiledArray::gemm;
      if (empty(result))
        result = gemm(left, right, 1, ContractReduceBase_::gemm_helper());
      else
        gemm(result, left, right, 1, ContractReduceBase_::gemm_helper());
    }
  }

};  // class ContractReduce

/// Contract and reduce operation

/// This object uses a tile contraction operation to form a pair reduction
/// operation.
/// \tparam Result The result tile type
/// \tparam Left The left-hand tile type
/// \tparam Right The right-hand tile type
/// \tparam Scalar The scaling factor type
template <typename Result, typename Left, typename Right, typename Scalar>
class ContractReduce<Result, Left, Right,
                     TiledArray::detail::ComplexConjugate<Scalar>>
    : public ContractReduceBase<Result, Left, Right,
                                TiledArray::detail::ComplexConjugate<Scalar>> {
 public:
  typedef ContractReduce<Result, Left, Right,
                         TiledArray::detail::ComplexConjugate<Scalar>>
      ContractReduce_;  ///< This class type
  typedef ContractReduceBase<Result, Left, Right,
                             TiledArray::detail::ComplexConjugate<Scalar>>
      ContractReduceBase_;  ///< This class type
  typedef typename ContractReduceBase_::first_argument_type
      first_argument_type;  ///< The left tile type
  typedef typename ContractReduceBase_::second_argument_type
      second_argument_type;  ///< The right tile type
  typedef decltype(gemm(std::declval<Left>(), std::declval<Right>(), 1,
                        std::declval<math::GemmHelper>()))
      result_type;  ///< The result tile type.
  typedef TiledArray::detail::ComplexConjugate<Scalar> scalar_type;

  using typename ContractReduceBase_::element_mult_op_type;
  using typename ContractReduceBase_::left_value_type;
  using typename ContractReduceBase_::result_value_type;
  using typename ContractReduceBase_::right_value_type;

  /// Compiler generated functions
  ContractReduce() = default;
  ContractReduce(const ContractReduce_&) = default;
  ContractReduce(ContractReduce_&&) = default;
  ~ContractReduce() = default;
  ContractReduce_& operator=(const ContractReduce_&) = default;
  ContractReduce_& operator=(ContractReduce_&&) = default;

  /// Construct contract/reduce functor

  /// \tparam ElementOp a callable with signature element_mult_op_type
  /// \param left_op The left-hand BLAS matrix operation
  /// \param right_op The right-hand BLAS matrix operation
  /// \param alpha The scaling factor applied to the contracted tiles
  /// \param result_rank The rank of the result tensor
  /// \param left_rank The rank of the left-hand tensor
  /// \param right_rank The rank of the right-hand tensor
  /// \param perm The permutation to be applied to the result tensor
  ///        (default = no permute)
  /// \param element_mult_op The element multiply op
  template <
      typename Perm = BipartitePermutation,
      typename ElementMultOp = TiledArray::function_ref<element_mult_op_type>,
      typename = std::enable_if_t<
          TiledArray::detail::is_permutation_v<Perm> &&
          std::is_invocable_r_v<
              result_value_type, std::remove_reference_t<ElementMultOp>,
              const left_value_type&, const right_value_type&>>>
  ContractReduce(const madness::cblas::CBLAS_TRANSPOSE left_op,
                 const madness::cblas::CBLAS_TRANSPOSE right_op,
                 const scalar_type alpha, const unsigned int result_rank,
                 const unsigned int left_rank, const unsigned int right_rank,
                 const Perm& perm = {}, ElementMultOp&& elem_mult_op = {})
      : ContractReduceBase_(left_op, right_op, alpha, result_rank, left_rank,
                            right_rank, perm,
                            std::forward<ElementMultOp>(elem_mult_op)) {}

  /// Create a result type object

  /// Initialize a result object for subsequent reductions
  result_type operator()() const { return result_type(); }

  /// Post processing step
  result_type operator()(result_type& temp) const {
    using TiledArray::empty;
    TA_ASSERT(!empty(temp));

    if (!ContractReduceBase_::perm()) {
      using TiledArray::conj_to;
      return conj_to(temp, ContractReduceBase_::factor().factor());
    }

    using TiledArray::conj;
    return conj(temp, ContractReduceBase_::factor().factor(),
                ContractReduceBase_::perm());
  }

  /// Reduce two result objects

  /// Add \c arg to \c result .
  /// \param[in,out] result The result object that will be the reduction
  /// target
  /// \param[in] arg The argument that will be added to \c result
  void operator()(result_type& result, const result_type& arg) const {
    using TiledArray::add_to;
    add_to(result, arg);
  }

  /// Contract a pair of tiles and add to a target tile

  /// Contract \c left and \c right and add the result to \c result.
  /// \param[in,out] result The result object that will be the reduction
  /// target
  /// \param[in] left The left-hand tile to be contracted
  /// \param[in] right The right-hand tile to be contracted
  void operator()(result_type& result, const first_argument_type& left,
                  const second_argument_type& right) const {
    if constexpr (!ContractReduceBase_::plain_tensors) {
      TA_ASSERT(this->element_mult_op());
      // not yet implemented
      abort();
    } else {
      TA_ASSERT(!this->element_mult_op());
      using TiledArray::empty;
      using TiledArray::gemm;
      if (empty(result))
        result = gemm(left, right, 1, ContractReduceBase_::gemm_helper());
      else
        gemm(result, left, right, 1, ContractReduceBase_::gemm_helper());
    }
  }

};  // class ContractReduce

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_CONTRACT_REDUCE_H__INCLUDED
