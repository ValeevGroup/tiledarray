/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2020  Virginia Tech
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
 *  Eduard Valeyev
 *  Department of Chemistry, Virginia Tech
 *
 *  permopt.h
 *  Nov 2, 2020
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_PERMOPT_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_PERMOPT_H__INCLUDED

#include <TiledArray/expressions/index_list.h>
#include <TiledArray/expressions/product.h>
#include <TiledArray/permutation.h>
#include <memory>

namespace TiledArray {
namespace expressions {

// clang-format off
/// Denotes whether permutation op is an identity or a matrix transpose;
/// this is important to be able to fuse permutations into GEMM:
/// - \c identity : an identity permutation
/// - \c matrix_transpose : matrix transpose (e.g. "ijklm" -> "lmijk")
/// - \c general : general permutation
// clang-format on
enum class PermutationType { identity = 1, matrix_transpose = 2, general = 3 };

inline blas::TransposeFlag to_cblas_op(PermutationType permtype) {
  TA_ASSERT(permtype == PermutationType::matrix_transpose ||
            permtype == PermutationType::identity);
  return permtype == PermutationType::matrix_transpose
             ? blas::Transpose
             : blas::NoTranspose;
}

/// Abstract optimizer of permutations for a binary operation
class BinaryOpPermutationOptimizer {
 public:
  /// construct using initial indices for the arguments and preference for which
  /// argument to permute
  /// \param left_indices the initial left argument index list
  /// \param right_indices the initial right argument index list
  /// \param prefer_to_permute_left whether to prefer permuting left argument
  BinaryOpPermutationOptimizer(const IndexList& left_indices,
                               const IndexList& right_indices,
                               const bool prefer_to_permute_left = true)
      : left_indices_(left_indices),
        right_indices_(right_indices),
        prefer_to_permute_left_(prefer_to_permute_left) {}

  /// construct using initial indices for the arguments,
  /// the desired result indices,
  /// and the preference for which
  /// argument to permute
  /// \param result_indices the desired result index list
  /// \param left_indices the initial left argument index list
  /// \param right_indices the initial right argument index list
  /// \param prefer_to_permute_left whether to prefer permuting left argument
  BinaryOpPermutationOptimizer(const IndexList& result_indices,
                               const IndexList& left_indices,
                               const IndexList& right_indices,
                               const bool prefer_to_permute_left = true)
      : result_indices_(result_indices),
        left_indices_(left_indices),
        right_indices_(right_indices),
        prefer_to_permute_left_(prefer_to_permute_left) {}

  BinaryOpPermutationOptimizer() = delete;
  BinaryOpPermutationOptimizer(const BinaryOpPermutationOptimizer&) = default;
  BinaryOpPermutationOptimizer& operator=(const BinaryOpPermutationOptimizer&) =
      default;
  virtual ~BinaryOpPermutationOptimizer() = default;

  /// \return the desired result indices
  const IndexList& result_indices() const {
    TA_ASSERT(result_indices_);
    return result_indices_;
  }
  /// \return initial left indices
  const IndexList& left_indices() const { return left_indices_; }
  /// \return initial right indices
  const IndexList& right_indices() const { return right_indices_; }
  /// \return whether preferred to permute left indices
  bool prefer_to_permute_left() const { return prefer_to_permute_left_; }

  /// \return the proposed left index list
  virtual const IndexList& target_left_indices() const = 0;
  /// \return the proposed right index list
  virtual const IndexList& target_right_indices() const = 0;
  /// \return the proposed result index list (not necessarily same as that
  /// returned by result_indices())
  virtual const IndexList& target_result_indices() const = 0;
  /// \return the type of permutation bringing the initial left index list to
  /// the target left index list
  virtual PermutationType left_permtype() const = 0;
  /// \return the type of permutation bringing the initial right index list to
  /// the target right index list
  virtual PermutationType right_permtype() const = 0;
  /// \return the binary op type
  virtual TensorProduct op_type() const = 0;

 private:
  IndexList result_indices_, left_indices_, right_indices_;
  bool prefer_to_permute_left_;
};

// clang-format off
/// Given left and right index lists computes the suggested indices for the left
/// and right args and the result, as well as (if any) ops to be applied to the
/// args to perform GEMM
// clang-format on
class GEMMPermutationOptimizer : public BinaryOpPermutationOptimizer {
 public:
  GEMMPermutationOptimizer(const GEMMPermutationOptimizer&) = default;
  GEMMPermutationOptimizer& operator=(const GEMMPermutationOptimizer&) =
      default;
  virtual ~GEMMPermutationOptimizer() = default;

  GEMMPermutationOptimizer(const IndexList& left_indices,
                           const IndexList& right_indices,
                           const bool prefer_to_permute_left = true)
      : BinaryOpPermutationOptimizer(left_indices, right_indices,
                                     prefer_to_permute_left) {
    std::tie(target_left_indices_, target_right_indices_,
             target_result_indices_, left_permtype_, right_permtype_) =
        compute_index_list_contraction(left_indices, right_indices,
                                       prefer_to_permute_left);
  }

  GEMMPermutationOptimizer(const IndexList& result_indices,
                           const IndexList& left_indices,
                           const IndexList& right_indices,
                           const bool prefer_to_permute_left = true)
      : BinaryOpPermutationOptimizer(left_indices, right_indices,
                                     prefer_to_permute_left) {
    std::tie(target_left_indices_, target_right_indices_,
             target_result_indices_, left_permtype_, right_permtype_) =
        compute_index_list_contraction(left_indices, right_indices,
                                       prefer_to_permute_left);
  }

  const IndexList& target_left_indices() const override final {
    return target_left_indices_;
  }
  const IndexList& target_right_indices() const override final {
    return target_right_indices_;
  }
  const IndexList& target_result_indices() const override final {
    return target_result_indices_;
  }
  PermutationType left_permtype() const override final {
    return left_permtype_;
  }
  PermutationType right_permtype() const override final {
    return right_permtype_;
  }
  TensorProduct op_type() const override final {
    return TensorProduct::Contraction;
  }

 private:
  IndexList target_left_indices_, target_right_indices_, target_result_indices_;
  PermutationType left_permtype_, right_permtype_;

  static auto find(const IndexList& indices, const std::string& idx,
                   unsigned int i, const unsigned int n) {
    const auto b = indices.begin() + i;
    const auto e = indices.begin() + n;
    const auto it = std::find(b, e, idx);
    return i + std::distance(b, it);
  };

  // clang-format off
  /// Computes index lists for the arguments and result given argument indices only
  /// \return `{left_index_list,right_index_list,result_index_list,left_op,right_op}` where
  ///         - `left_index_list` is the list of target indices for the left argument
  ///         - `right_index_list` is the list of target indices for the right expression
  ///         - `result_index_list` is the list of indices produced in the contraction op (this list may need to be further permuted to produce the target indices; see ContEngine::perm_indices )
  ///         - `left_op` is the permutation to be applied to the left expression result
  ///         - `right_op` is the permutation to be applied to the right expression result
  // clang-format on
  inline std::tuple<IndexList, IndexList, IndexList, PermutationType,
                    PermutationType>
  compute_index_list_contraction(const IndexList& left_indices,
                                 const IndexList& right_indices,
                                 const bool prefer_to_permute_left = true) {
    const auto left_rank = left_indices.size();
    const auto right_rank = right_indices.size();

    container::svector<std::string> result_left_indices;
    result_left_indices.reserve(left_rank);
    container::svector<std::string> result_right_indices;
    result_right_indices.reserve(right_rank);
    container::svector<std::string> result_indices;
    result_indices.reserve(std::max(left_rank, right_rank));

    // Extract left-most result and inner indices from the left-hand argument.
    for (unsigned int i = 0ul; i < left_rank; ++i) {
      const std::string& var = left_indices[i];
      if (find(right_indices, var, 0u, right_rank) == right_rank) {
        // Store outer left variable
        result_left_indices.push_back(var);
        result_indices.push_back(var);
      } else {
        // Store inner left variable
        result_right_indices.push_back(var);
      }
    }

    // Compute the inner and outer dimension ranks.
    const unsigned int inner_rank = result_right_indices.size();
    const unsigned int left_outer_rank = result_left_indices.size();
    const unsigned int right_outer_rank = right_rank - inner_rank;
    const unsigned int result_rank = left_outer_rank + right_outer_rank;

    // Resize result indices if necessary.
    result_indices.reserve(result_rank);

    // If an outer product, result = concat of free indices from left and right
    if (inner_rank == 0u) {
      // Extract the right most outer variables from right hand argument.
      for (unsigned int i = 0ul; i < right_rank; ++i) {
        const std::string& var = right_indices[i];
        result_right_indices.push_back(var);
        result_indices.push_back(var);
      }
      // early return for the inner product since will make the result to be
      // pure concat of the left and right index lists
      return std::make_tuple(
          IndexList(result_left_indices), IndexList(result_right_indices),
          IndexList(result_indices), PermutationType::general,
          PermutationType::general);
    }

    // Initialize flags that will be used to determine the type of permutation
    // that will be applied to the arguments (i.e. no permutation, matrix
    // transpose, or arbitrary permutation).
    bool inner_indices_ordered = true, left_is_no_trans = true,
         left_is_trans = true, right_is_no_trans = true, right_is_trans = true;

    // If the inner index lists of the arguments are not in the same
    // order, one of them will need to be permuted. Here, we determine which
    // argument, left or right, will be permuted if a permutation is
    // required. The argument with the lowest rank is preferred since it is
    // likely to have the smaller memory footprint.
    const bool perm_left =
        (left_rank < right_rank) ||
        ((left_rank == right_rank) && prefer_to_permute_left);

    // Extract variables from the right-hand argument, collect information
    // about the layout of the index lists, and ensure the inner variable
    // lists are in the same order.
    for (unsigned int i = 0ul; i < right_rank; ++i) {
      const std::string& idx = right_indices[i];
      const unsigned int j = find(left_indices, idx, 0u, left_rank);
      if (j == left_rank) {
        // Store outer right index
        result_right_indices.push_back(idx);
        result_indices.push_back(idx);
      } else {
        const unsigned int x = result_left_indices.size() - left_outer_rank;

        // Collect information about the relative position of variables
        inner_indices_ordered =
            inner_indices_ordered && (result_right_indices[x] == idx);
        left_is_no_trans = left_is_no_trans && (j >= left_outer_rank);
        left_is_trans = left_is_trans && (j < inner_rank);
        right_is_no_trans = right_is_no_trans && (i < inner_rank);
        right_is_trans = right_is_trans && (i >= right_outer_rank);

        // Store inner right index
        if (inner_indices_ordered) {
          // Left and right inner index list order is equal.
          result_left_indices.push_back(idx);
        } else if (perm_left) {
          // Permute left so we need to store inner indices according to
          // the order of the right-hand argument.
          result_left_indices.push_back(idx);
          result_right_indices[x] = idx;
          left_is_no_trans = left_is_trans = false;
        } else {
          // Permute right so we need to store inner indices according to
          // the order of the left-hand argument.
          result_left_indices.push_back(result_right_indices[x]);
          right_is_no_trans = right_is_trans = false;
        }
      }
    }

    auto to_tensor_op = [](bool no_trans, bool trans) {
      if (no_trans)
        return PermutationType::identity;
      else if (trans)
        return PermutationType::matrix_transpose;
      else
        return PermutationType::general;
    };

    return std::make_tuple(IndexList(result_left_indices),
                           IndexList(result_right_indices),
                           IndexList(result_indices),
                           to_tensor_op(left_is_no_trans, left_is_trans),
                           to_tensor_op(right_is_no_trans, right_is_trans));
  }

  // clang-format off
  /// Computes index lists for the arguments and result given argument _and_ target result indices
  /// \return `{left_index_list,right_index_list,result_index_list,left_op,right_op}` where
  ///         - `left_index_list` is the list of target indices for the left argument
  ///         - `right_index_list` is the list of target indices for the right expression
  ///         - `result_index_list` is the list of indices produced in the contraction op (this list may need to be further permuted to produce the target indices; see ContEngine::perm_indices )
  ///         - `left_op` is the permutation to be applied to the left expression result
  ///         - `right_op` is the permutation to be applied to the right expression result
  // clang-format on
  inline std::tuple<IndexList, IndexList, IndexList, PermutationType,
                    PermutationType>
  compute_index_list_contraction(const IndexList& target_indices,
                                 const IndexList& left_indices,
                                 const IndexList& right_indices,
                                 const bool prefer_to_permute_left = true) {
    IndexList result_indices_, left_indices_,
        right_indices_;  // intermediate index lists, computed without taking
                         // target_indices into account
    PermutationType left_op_, right_op_;
    std::tie(result_indices_, left_indices_, right_indices_, left_op_,
             right_op_) =
        compute_index_list_contraction(left_indices, right_indices,
                                       prefer_to_permute_left);

    container::svector<std::string> final_left_indices(left_indices_.size()),
        final_right_indices(right_indices_.size()),
        final_result_indices(result_indices_.size());

    // Only permute if the arguments can be permuted
    if ((left_op_ == PermutationType::general) ||
        (right_op_ == PermutationType::general)) {
      // Compute ranks
      const unsigned int result_rank = target_indices.size();
      const unsigned int inner_rank =
          (left_indices_.size() + right_indices_.size() - result_rank) >> 1;
      const unsigned int left_outer_rank = left_indices_.size() - inner_rank;

      // Check that the left- and right-hand outer variables are correctly
      // partitioned in the target index list.
      bool target_partitioned = true;
      for (unsigned int i = 0u; i < left_outer_rank; ++i)
        target_partitioned =
            target_partitioned && (find(target_indices, left_indices_[i], 0u,
                                        left_outer_rank) < left_outer_rank);

      // If target is properly partitioned, then arguments can be permuted
      // to fit the target.
      if (target_partitioned) {
        if (left_op_ == PermutationType::general) {
          // Copy left-hand target variables to left and result index lists.
          for (unsigned int i = 0u; i < left_outer_rank; ++i) {
            const std::string& var = target_indices[i];
            final_left_indices[i] = var;
            final_result_indices[i] = var;
          }
        } else {
          // Copy left-hand outer variables to that of result.
          for (unsigned int i = 0u; i < left_outer_rank; ++i)
            final_result_indices[i] = left_indices_[i];
        }

        if (right_op_ == PermutationType::general) {
          // Copy right-hand target variables to right and result variable
          // lists.
          for (unsigned int i = left_outer_rank, j = inner_rank;
               i < result_rank; ++i, ++j) {
            const std::string& var = target_indices[i];
            final_right_indices[j] = var;
            final_result_indices[i] = var;
          }
        } else {
          // Copy right-hand outer variables to that of result.
          for (unsigned int i = left_outer_rank, j = inner_rank;
               i < result_rank; ++i, ++j)
            final_result_indices[i] = right_indices_[j];
        }
      }
    }

    return std::make_tuple(
        IndexList(final_left_indices), IndexList(final_right_indices),
        IndexList(final_result_indices), left_op_, right_op_);
  }
};

// clang-format off
/// Given left and right index lists computes the suggested indices for the left
/// and right args and the result for computing Hadamard product efficiently
// clang-format on
class HadamardPermutationOptimizer : public BinaryOpPermutationOptimizer {
 public:
  HadamardPermutationOptimizer(const HadamardPermutationOptimizer&) = default;
  HadamardPermutationOptimizer& operator=(const HadamardPermutationOptimizer&) =
      default;
  ~HadamardPermutationOptimizer() = default;

  HadamardPermutationOptimizer(const IndexList& left_indices,
                               const IndexList& right_indices,
                               const bool prefer_to_permute_left = true)
      : BinaryOpPermutationOptimizer(left_indices, right_indices,
                                     prefer_to_permute_left),
        target_result_indices_(prefer_to_permute_left ? right_indices
                                                      : left_indices) {
    TA_ASSERT(left_indices.is_permutation(right_indices));
  }

  HadamardPermutationOptimizer(const IndexList& result_indices,
                               const IndexList& left_indices,
                               const IndexList& right_indices,
                               const bool prefer_to_permute_left = true)
      : BinaryOpPermutationOptimizer(result_indices, left_indices,
                                     right_indices, prefer_to_permute_left) {
    TA_ASSERT(left_indices.is_permutation(result_indices));
    TA_ASSERT(left_indices.is_permutation(right_indices));

    // Determine the equality of the index lists
    bool left_target = true, right_target = true, left_right = true;
    for (unsigned int i = 0u; i < result_indices.size(); ++i) {
      left_target = left_target && left_indices[i] == result_indices[i];
      right_target = right_target && right_indices[i] == result_indices[i];
      left_right = left_right && left_indices[i] == right_indices[i];
    }

    if (left_right) {
      target_result_indices_ = left_indices;
    } else {
      // Determine which argument will be permuted
      const bool perm_left =
          (right_target ||
           ((!(left_target || right_target)) && prefer_to_permute_left));

      target_result_indices_ = perm_left ? right_indices : left_indices;
    }
  }

  const IndexList& target_left_indices() const override final {
    return target_result_indices_;
  }
  const IndexList& target_right_indices() const override final {
    return target_result_indices_;
  }
  const IndexList& target_result_indices() const override final {
    return target_result_indices_;
  }
  PermutationType left_permtype() const override final {
    return PermutationType::general;
  }
  PermutationType right_permtype() const override final {
    return PermutationType::general;
  }
  TensorProduct op_type() const override final {
    return TensorProduct::Hadamard;
  }

 private:
  IndexList target_result_indices_;
};

class NullBinaryOpPermutationOptimizer : public BinaryOpPermutationOptimizer {
 public:
  NullBinaryOpPermutationOptimizer(const NullBinaryOpPermutationOptimizer&) =
      default;
  NullBinaryOpPermutationOptimizer& operator=(
      const NullBinaryOpPermutationOptimizer&) = default;
  ~NullBinaryOpPermutationOptimizer() = default;

  NullBinaryOpPermutationOptimizer(const IndexList& left_indices,
                                   const IndexList& right_indices,
                                   const bool prefer_to_permute_left = true)
      : BinaryOpPermutationOptimizer(left_indices, right_indices,
                                     prefer_to_permute_left) {
    TA_ASSERT(!left_indices);
    TA_ASSERT(!right_indices);
  }

  NullBinaryOpPermutationOptimizer(const IndexList& result_indices,
                                   const IndexList& left_indices,
                                   const IndexList& right_indices,
                                   const bool prefer_to_permute_left = true)
      : BinaryOpPermutationOptimizer(result_indices, left_indices,
                                     right_indices, prefer_to_permute_left) {
    TA_ASSERT(!result_indices);
    TA_ASSERT(!left_indices);
    TA_ASSERT(!right_indices);
  }

  const IndexList& target_left_indices() const override final {
    return left_indices();
  }
  const IndexList& target_right_indices() const override final {
    return right_indices();
  }
  const IndexList& target_result_indices() const override final {
    return left_indices();
  }
  PermutationType left_permtype() const override final {
    return PermutationType::general;
  }
  PermutationType right_permtype() const override final {
    return PermutationType::general;
  }
  TensorProduct op_type() const override final {
    return TensorProduct::Invalid;
  }
};

inline std::shared_ptr<BinaryOpPermutationOptimizer> make_permutation_optimizer(
    TensorProduct product_type, const IndexList& left_indices,
    const IndexList& right_indices, bool prefer_to_permute_left) {
  switch (product_type) {
    case TensorProduct::Hadamard:
      return std::make_shared<HadamardPermutationOptimizer>(
          left_indices, right_indices, prefer_to_permute_left);
    case TensorProduct::Contraction:
      return std::make_shared<GEMMPermutationOptimizer>(
          left_indices, right_indices, prefer_to_permute_left);
    case TensorProduct::Invalid:
      return std::make_shared<NullBinaryOpPermutationOptimizer>(
          left_indices, right_indices, prefer_to_permute_left);
    default:
      abort();
  }
}

inline std::shared_ptr<BinaryOpPermutationOptimizer> make_permutation_optimizer(
    TensorProduct product_type, const IndexList& target_indices,
    const IndexList& left_indices, const IndexList& right_indices,
    bool prefer_to_permute_left) {
  switch (product_type) {
    case TensorProduct::Hadamard:
      return std::make_shared<HadamardPermutationOptimizer>(
          target_indices, left_indices, right_indices, prefer_to_permute_left);
    case TensorProduct::Contraction:
      return std::make_shared<GEMMPermutationOptimizer>(
          target_indices, left_indices, right_indices, prefer_to_permute_left);
    case TensorProduct::Invalid:
      return std::make_shared<NullBinaryOpPermutationOptimizer>(
          target_indices, left_indices, right_indices, prefer_to_permute_left);
    default:
      abort();
  }
}

inline std::shared_ptr<BinaryOpPermutationOptimizer> make_permutation_optimizer(
    const IndexList& left_indices, const IndexList& right_indices,
    bool prefer_to_permute_left) {
  return make_permutation_optimizer(
      compute_product_type(left_indices, right_indices), left_indices,
      right_indices, prefer_to_permute_left);
}

inline std::shared_ptr<BinaryOpPermutationOptimizer> make_permutation_optimizer(
    const IndexList& target_indices, const IndexList& left_indices,
    const IndexList& right_indices, bool prefer_to_permute_left) {
  return make_permutation_optimizer(
      compute_product_type(left_indices, right_indices, target_indices),
      target_indices, left_indices, right_indices, prefer_to_permute_left);
}

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_PERMOPT_H__INCLUDED
