/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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
 *  gemm_helper.h
 *  Jan 20, 2014
 *
 */

#ifndef TILEDARRAY_MATH_GEMM_HELPER_H__INCLUDED
#define TILEDARRAY_MATH_GEMM_HELPER_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/external/madness.h>
#include <TiledArray/math/blas.h>

namespace TiledArray::math {

/// Contraction to *GEMM helper

/// This object is used to convert tensor contraction to *GEMM operations by
/// providing information on how to fuse dimensions
class GemmHelper {
 private:
  blas::TransposeFlag left_op_;
  ///< Transpose operation that is applied to the left-hand argument
  blas::TransposeFlag right_op_;
  ///< Transpose operation that is applied to the right-hand argument
  unsigned int result_rank_;  ///< The rank of the result tensor

  /// Contraction argument range data

  /// The range data held by this object is the range of the inner and outer
  /// dimensions of the argument tensor. It is assumed that the inner and
  /// outer dimensions are contiguous.
  struct ContractArg {
    unsigned int inner[2];  ///< The inner dimension range
    unsigned int outer[2];  ///< The outer dimension range
    unsigned int rank;      ///< Rank of the argument tensor
  } left_,                  ///< Left-hand argument range data
      right_;               ///< Right-hand argument range data

 public:
  GemmHelper(const blas::TransposeFlag left_op,
             const blas::TransposeFlag right_op,
             const unsigned int result_rank, const unsigned int left_rank,
             const unsigned int right_rank)
      : left_op_(left_op),
        right_op_(right_op),
        result_rank_(result_rank),
        left_(),
        right_() {
    // Compute the number of contracted dimensions in left and right.
    TA_ASSERT(((left_rank + right_rank - result_rank) % 2u) == 0u);

    left_.rank = left_rank;
    right_.rank = right_rank;
    const unsigned int contract_size = num_contract_ranks();

    // Store the inner and outer dimension ranges for the left-hand argument.
    if (left_op == blas::NoTranspose) {
      left_.outer[0] = 0u;
      left_.outer[1] = left_.inner[0] = left_rank - contract_size;
      left_.inner[1] = left_rank;
    } else {
      left_.inner[0] = 0ul;
      left_.inner[1] = left_.outer[0] = contract_size;
      left_.outer[1] = left_rank;
    }

    // Store the inner and outer dimension ranges for the right-hand argument.
    if (right_op == blas::NoTranspose) {
      right_.inner[0] = 0u;
      right_.inner[1] = right_.outer[0] = contract_size;
      right_.outer[1] = right_rank;
    } else {
      right_.outer[0] = 0u;
      right_.outer[1] = right_.inner[0] = right_rank - contract_size;
      right_.inner[1] = right_rank;
    }
  }

  /// Functor copy constructor

  /// Shallow copy of this functor
  /// \param other The functor to be copied
  GemmHelper(const GemmHelper& other)
      : left_op_(other.left_op_),
        right_op_(other.right_op_),
        result_rank_(other.result_rank_),
        left_(other.left_),
        right_(other.right_) {}

  /// Functor assignment operator

  /// \param other The functor to be copied
  GemmHelper& operator=(const GemmHelper& other) {
    left_op_ = other.left_op_;
    right_op_ = other.right_op_;
    result_rank_ = other.result_rank_;
    left_ = other.left_;
    right_ = other.right_;

    return *this;
  }

  /// Compute the number of contracted ranks

  /// \return The number of ranks that are summed by this operation
  unsigned int num_contract_ranks() const {
    return (left_.rank + right_.rank - result_rank_) >> 1;
  }

  /// Result rank accessor

  /// \return The rank of the result tile
  unsigned int result_rank() const { return result_rank_; }

  /// Left-hand argument rank accessor

  /// \return The rank of the left-hand tile
  unsigned int left_rank() const { return left_.rank; }

  /// Right-hand argument rank accessor

  /// \return The rank of the right-hand tile
  unsigned int right_rank() const { return right_.rank; }

  unsigned int left_inner_begin() const { return left_.inner[0]; }
  unsigned int left_inner_end() const { return left_.inner[1]; }
  unsigned int left_outer_begin() const { return left_.outer[0]; }
  unsigned int left_outer_end() const { return left_.outer[1]; }

  unsigned int right_inner_begin() const { return right_.inner[0]; }
  unsigned int right_inner_end() const { return right_.inner[1]; }
  unsigned int right_outer_begin() const { return right_.outer[0]; }
  unsigned int right_outer_end() const { return right_.outer[1]; }

  /// Construct a result range based on \c left and \c right ranges

  /// \tparam R The result range type
  /// \tparam Left The left-hand range type
  /// \tparam Right The right-hand range type
  /// \param left The left-hand range
  /// \param right The right-hand range
  /// \return A range object that can be used in a tensor contraction
  /// defined by this object
  template <typename R, typename Left, typename Right>
  R make_result_range(const Left& left, const Right& right) const {
    // Get pointers to lower and upper bounds of left and right.
    const auto* MADNESS_RESTRICT const left_lower = left.lobound_data();
    const auto* MADNESS_RESTRICT const left_upper = left.upbound_data();
    const auto* MADNESS_RESTRICT const right_lower = right.lobound_data();
    const auto* MADNESS_RESTRICT const right_upper = right.upbound_data();

    // Create the start and finish indices
    std::vector<std::size_t> lower, upper;
    lower.reserve(result_rank_);
    upper.reserve(result_rank_);

    // Copy left-hand argument outer dimensions to start and finish
    for (unsigned int i = left_.outer[0]; i < left_.outer[1]; ++i) {
      lower.push_back(left_lower[i]);
      upper.push_back(left_upper[i]);
    }

    // Copy right-hand argument outer dimensions to start and finish
    for (unsigned int i = right_.outer[0]; i < right_.outer[1]; ++i) {
      lower.push_back(right_lower[i]);
      upper.push_back(right_upper[i]);
    }

    // Construct the result tile range
    return R(lower, upper);
  }

  /// Test that the outer dimensions of left are congruent (have equal extent)
  /// with that of the result tensor

  /// This function can test the start, finish, or size arrays of range
  /// objects.
  /// \tparam Left The left-hand size array type
  /// \tparam Result The result size array type
  /// \param left The left-hand size array to be tested
  /// \param result The result size array to be tested
  /// \return \c true if The outer dimensions of left are congruent with that
  /// of result
  template <typename Left, typename Result>
  bool left_result_congruent(const Left& left, const Result& result) const {
    return std::equal(left + left_.outer[0], left + left_.outer[1], result);
  }

  /// Test that the outer dimensions of right are congruent (have equal extent)
  /// with that of the result tensor

  /// This function can test the start, finish, or size arrays of range
  /// objects.
  /// \tparam Right The right-hand size array type
  /// \tparam Result The result size array type
  /// \param right The right-hand size array to be tested
  /// \param result The result size array to be tested
  /// \return \c true if The outer dimensions of right are congruent with that
  /// of result
  template <typename Right, typename Result>
  bool right_result_congruent(const Right& right, const Result& result) const {
    return std::equal(right + right_.outer[0], right + right_.outer[1],
                      result + (left_.outer[1] - left_.outer[0]));
  }

  /// Test that the inner dimensions of left are congruent (have equal extent)
  /// with that of right

  /// This function can test the start, finish, or size arrays of range
  /// objects.
  /// \tparam Left The left-hand size array type
  /// \tparam Right The right-hand size array type
  /// \param left The left-hand size array to be tested
  /// \param right The right-hand size array to be tested
  /// \return \c true if the outer dimensions of \c left are congruent with
  /// that of \c right, other \c false.
  template <typename Left, typename Right>
  bool left_right_congruent(const Left& left, const Right& right) const {
    return std::equal(left + left_.inner[0], left + left_.inner[1],
                      right + right_.inner[0]);
  }

  /// Compute the matrix dimension that can be used in a *GEMM call

  /// \tparam Left The left-hand range type
  /// \tparam Right The right-hand range type
  /// \param[out] m The number of rows in left-hand and result matrices
  /// \param[out] n The number of columns in the right-hand result matrices
  /// \param[out] k The number of columns in the left-hand matrix and the
  /// number of rows in the right-hand matrix
  /// \param[in] left The left-hand range object
  /// \param[in] right The right-hand range object
  template <typename Left, typename Right>
  void compute_matrix_sizes(integer& m, integer& n, integer& k,
                            const Left& left, const Right& right) const {
    // Check that the arguments are not empty and have the correct ranks
    TA_ASSERT(left.rank() == left_.rank);
    TA_ASSERT(right.rank() == right_.rank);
    const auto* MADNESS_RESTRICT const left_extent = left.extent_data();
    const auto* MADNESS_RESTRICT const right_extent = right.extent_data();

    // Compute fused dimension sizes
    m = 1;
    for (unsigned int i = left_.outer[0]; i < left_.outer[1]; ++i)
      m *= left_extent[i];
    k = 1;
    for (unsigned int i = left_.inner[0]; i < left_.inner[1]; ++i)
      k *= left_extent[i];
    n = 1;
    for (unsigned int i = right_.outer[0]; i < right_.outer[1]; ++i)
      n *= right_extent[i];
  }

  blas::TransposeFlag left_op() const { return left_op_; }
  blas::TransposeFlag right_op() const { return right_op_; }
};  // class GemmHelper

}  // namespace TiledArray::math

#endif  // TILEDARRAY_MATH_GEMM_HELPER_H__INCLUDED
