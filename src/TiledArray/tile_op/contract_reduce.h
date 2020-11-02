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
#include "../tile_interface/add.h"
#include "../tile_interface/permute.h"

namespace TiledArray {
namespace detail {

/// Contract and (sum) reduce base

/// This implementation class is used to provide shallow copy semantics for
/// ContractReduce. It encodes a binary tensor contraction mapped to a GEMM, as
/// well as the sum reduction and post-processing. \tparam Result The result
/// tile type \tparam Left The left-hand tile type \tparam Right The right-hand
/// tile type \tparam Scalar The scaling factor type
template <typename Result, typename Left, typename Right, typename Scalar>
class ContractReduceBase {
 public:
  typedef ContractReduceBase<Result, Left, Right, Scalar>
      ContractReduceBase_;                    ///< This class type
  typedef const Left& first_argument_type;    ///< The left tile type
  typedef const Right& second_argument_type;  ///< The right tile type
  typedef Result result_type;                 ///< The result type
  typedef Scalar scalar_type;                 ///< The scaling factor type

 private:
  struct Impl {
    template <typename Perm, typename = std::enable_if_t<
                                 TiledArray::detail::is_permutation_v<Perm>>>
    Impl(const madness::cblas::CBLAS_TRANSPOSE left_op,
         const madness::cblas::CBLAS_TRANSPOSE right_op,
         const scalar_type alpha, const unsigned int result_rank,
         const unsigned int left_rank, const unsigned int right_rank,
         const Perm& perm = BipartitePermutation())
        : gemm_helper_(left_op, right_op, result_rank, left_rank, right_rank),
          alpha_(alpha),
          perm_(perm) {}

    math::GemmHelper gemm_helper_;  ///< Gemm helper object
    scalar_type alpha_;  ///< Scaling factor applied to the contraction of
                         ///< the left- and right-hand arguments
    BipartitePermutation perm_;  ///< Permutation that is applied to the final
                                 ///< result tensor
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

  /// \param left_op The left-hand BLAS matrix operation
  /// \param right_op The right-hand BLAS matrix operation
  /// \param alpha The scaling factor applied to the contracted tiles
  /// \param result_rank The rank of the result tensor
  /// \param left_rank The rank of the left-hand tensor
  /// \param right_rank The rank of the right-hand tensor
  /// \param perm The permutation to be applied to the result tensor
  /// (default = no permute)
  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  ContractReduceBase(const madness::cblas::CBLAS_TRANSPOSE left_op,
                     const madness::cblas::CBLAS_TRANSPOSE right_op,
                     const scalar_type alpha, const unsigned int result_rank,
                     const unsigned int left_rank,
                     const unsigned int right_rank,
                     const Perm& perm = BipartitePermutation())
      : pimpl_(std::make_shared<Impl>(left_op, right_op, alpha, result_rank,
                                      left_rank, right_rank, perm)) {}

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
/// sum reduction and post-processing. \tparam Result The result tile type
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

  // Compiler generated defaults are fine. N.B. this is shallow-copy.

  ContractReduce() = default;
  ContractReduce(const ContractReduce_&) = default;
  ContractReduce(ContractReduce_&&) = default;
  ~ContractReduce() = default;
  ContractReduce_& operator=(const ContractReduce_&) = default;
  ContractReduce_& operator=(ContractReduce_&&) = default;

  /// Construct contract/reduce functor

  /// \param left_op The left-hand BLAS matrix operation
  /// \param right_op The right-hand BLAS matrix operation
  /// \param alpha The scaling factor applied to the contracted tiles
  /// \param result_rank The rank of the result tensor
  /// \param left_rank The rank of the left-hand tensor
  /// \param right_rank The rank of the right-hand tensor
  /// \param perm The permutation to be applied to the result tensor
  /// (default = no permute)
  template <
      typename Perm = BipartitePermutation,
      typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
  ContractReduce(const madness::cblas::CBLAS_TRANSPOSE left_op,
                 const madness::cblas::CBLAS_TRANSPOSE right_op,
                 const scalar_type alpha, const unsigned int result_rank,
                 const unsigned int left_rank, const unsigned int right_rank,
                 const Perm& perm = Perm{})
      : ContractReduceBase_(left_op, right_op, alpha, result_rank, left_rank,
                            right_rank, perm) {}

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
  void operator()(result_type& result, first_argument_type left,
                  second_argument_type right) const {
    using TiledArray::empty;
    using TiledArray::gemm;
    if (empty(result))
      result = gemm(left, right, ContractReduceBase_::factor(),
                    ContractReduceBase_::gemm_helper());
    else
      gemm(result, left, right, ContractReduceBase_::factor(),
           ContractReduceBase_::gemm_helper());
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

  // Compiler generated defaults are fine. N.B. This has shallow copy semantics.

  ContractReduce() = default;
  ContractReduce(const ContractReduce_&) = default;
  ContractReduce(ContractReduce_&&) = default;
  ~ContractReduce() = default;
  ContractReduce_& operator=(const ContractReduce_&) = default;
  ContractReduce_& operator=(ContractReduce_&&) = default;

  /// Construct contract/reduce functor

  /// \param left_op The left-hand BLAS matrix operation
  /// \param right_op The right-hand BLAS matrix operation
  /// \param alpha The scaling factor applied to the contracted tiles
  /// \param result_rank The rank of the result tensor
  /// \param left_rank The rank of the left-hand tensor
  /// \param right_rank The rank of the right-hand tensor
  /// \param perm The permutation to be applied to the result tensor
  /// (default = no permute)
  template <
      typename Perm = BipartitePermutation,
      typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
  ContractReduce(const madness::cblas::CBLAS_TRANSPOSE left_op,
                 const madness::cblas::CBLAS_TRANSPOSE right_op,
                 const scalar_type alpha, const unsigned int result_rank,
                 const unsigned int left_rank, const unsigned int right_rank,
                 const Perm& perm = Perm{})
      : ContractReduceBase_(left_op, right_op, alpha, result_rank, left_rank,
                            right_rank, perm) {}

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
  void operator()(result_type& result, first_argument_type left,
                  second_argument_type right) const {
    using TiledArray::empty;
    using TiledArray::gemm;
    if (empty(result))
      result = gemm(left, right, 1, ContractReduceBase_::gemm_helper());
    else
      gemm(result, left, right, 1, ContractReduceBase_::gemm_helper());
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

  /// Compiler generated functions
  ContractReduce() = default;
  ContractReduce(const ContractReduce_&) = default;
  ContractReduce(ContractReduce_&&) = default;
  ~ContractReduce() = default;
  ContractReduce_& operator=(const ContractReduce_&) = default;
  ContractReduce_& operator=(ContractReduce_&&) = default;

  /// Construct contract/reduce functor

  /// \param left_op The left-hand BLAS matrix operation
  /// \param right_op The right-hand BLAS matrix operation
  /// \param alpha The scaling factor applied to the contracted tiles
  /// \param result_rank The rank of the result tensor
  /// \param left_rank The rank of the left-hand tensor
  /// \param right_rank The rank of the right-hand tensor
  /// \param perm The permutation to be applied to the result tensor
  /// (default = no permute)
  template <
      typename Perm = BipartitePermutation,
      typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
  ContractReduce(const madness::cblas::CBLAS_TRANSPOSE left_op,
                 const madness::cblas::CBLAS_TRANSPOSE right_op,
                 const scalar_type alpha, const unsigned int result_rank,
                 const unsigned int left_rank, const unsigned int right_rank,
                 const Perm& perm = {})
      : ContractReduceBase_(left_op, right_op, alpha, result_rank, left_rank,
                            right_rank, perm) {}

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
  void operator()(result_type& result, first_argument_type left,
                  second_argument_type right) const {
    using TiledArray::empty;
    using TiledArray::gemm;
    if (empty(result))
      result = gemm(left, right, 1, ContractReduceBase_::gemm_helper());
    else
      gemm(result, left, right, 1, ContractReduceBase_::gemm_helper());
  }

};  // class ContractReduce

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_CONTRACT_REDUCE_H__INCLUDED
