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
 *  cont_engine.h
 *  Mar 31, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_CONT_ENGINE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_CONT_ENGINE_H__INCLUDED

#include <TiledArray/dist_eval/contraction_eval.h>
#include <TiledArray/expressions/binary_engine.h>
#include <TiledArray/proc_grid.h>
#include <TiledArray/tensor/utility.h>
#include <TiledArray/tile_op/contract_reduce.h>

namespace TiledArray {
namespace expressions {

/// types of binary tensor products known to TiledArray
enum class TensorProduct {
  /// fused indices only
  Hadamard,
  /// (at least 1) free and (at least 1) contracted indices
  Contraction,
  /// free, fused, and contracted indices
  General,
  /// invalid
  Invalid = -1
};

/// computes the tensor product type corresponding to the left and right
/// argument indices \param left_indices the left argument index list \param
/// right_indices the right argument index list \return TensorProduct::Invalid
/// is either argument is empty, TensorProduct::Hadamard if the arguments are
/// related by a permutation, else TensorProduct::Contraction
inline TensorProduct compute_product_type(const IndexList& left_indices,
                                          const IndexList& right_indices) {
  TensorProduct result = TensorProduct::Invalid;
  if (left_indices && right_indices) {
    if (left_indices.size() == right_indices.size() &&
        left_indices.is_permutation(right_indices))
      result = TensorProduct::Hadamard;
    else
      result = TensorProduct::Contraction;
  }
  return result;
}

/// computes the tensor product type corresponding to the left and right
/// argument indices, and validates against the target indices
inline TensorProduct compute_product_type(const IndexList& left_indices,
                                          const IndexList& right_indices,
                                          const IndexList& target_indices) {
  auto result = compute_product_type(left_indices, right_indices);
  if (result == TensorProduct::Hadamard)
    TA_ASSERT(left_indices.is_permutation(target_indices));
  return result;
}

// clang-format off
/// Denotes permutation op to be applied to the tensor product arguments in order
/// to express it as a GEMM
/// - \c no_trans : no permutation
/// - \c trans : matrix transpose (i.e. permutation expressible as a matrix
///   transpose, e.g. "ijklm" -> "lmijk")
/// - \c permute_to_no_trans : general permutation
/// \note need to distinguish `trans` from `permute_to_no_trans` since `trans` and `no_trans` can be performed by BLAS' GEMM
// clang-format on
enum class GEMMTensorArgOp { no_trans = 1, trans = 2, permute_to_no_trans = 3 };

// clang-format off
/// Given left and right index lists computes the suggested indices for the left
/// and right args and the result, as well as (if any) ops to be applied to the
/// args to perform GEMM
// clang-format on
struct GEMMPermutationOptimizer {
  unsigned int num_contracted_indices;
  IndexList target_left_indices, target_right_indices, target_result_indices;
  GEMMTensorArgOp left_op, right_op;

  GEMMPermutationOptimizer() = default;
  GEMMPermutationOptimizer(const GEMMPermutationOptimizer&) = default;
  GEMMPermutationOptimizer& operator=(const GEMMPermutationOptimizer&) =
      default;

  GEMMPermutationOptimizer(const IndexList& left_indices,
                           const IndexList& right_indices,
                           const bool prefer_to_permute_left = true) {
    std::tie(num_contracted_indices, target_left_indices, target_right_indices,
             target_result_indices, left_op, right_op) =
        compute_index_list_contraction(left_indices, right_indices,
                                       prefer_to_permute_left);
  }

  // clang-format off
  /// \return `{num_contracted_indices,left_index_list,right_index_list,result_index_list,left_op,right_op}` where
  ///         - `num_contracted_indices` is the number of contracted indices (1 or greater)
  ///         - `left_index_list` is the list of target indices for the left argument
  ///         - `right_index_list` is the list of target indices for the right expression
  ///         - `result_index_list` is the list of indices produced in the contraction op (this list may need to be further permuted to produce the target indices; see ContEngine::perm_indices )
  ///         - `left_op` is the permutation to be applied to the left expression result
  ///         - `right_op` is the permutation to be applied to the right expression result
  // clang-format on
  inline std::tuple<unsigned int, IndexList, IndexList, IndexList,
                    GEMMTensorArgOp, GEMMTensorArgOp>
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

    auto find = [](const IndexList& indices, const std::string& idx,
                   unsigned int i, const unsigned int n) {
      const auto b = indices.begin() + i;
      const auto e = indices.begin() + n;
      const auto it = std::find(b, e, idx);
      return i + std::distance(b, it);
    };

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
      return std::make_tuple(inner_rank, IndexList(result_left_indices),
                             IndexList(result_right_indices),
                             IndexList(result_indices),
                             GEMMTensorArgOp::permute_to_no_trans,
                             GEMMTensorArgOp::permute_to_no_trans);
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
        return GEMMTensorArgOp::no_trans;
      else if (trans)
        return GEMMTensorArgOp::trans;
      else
        return GEMMTensorArgOp::permute_to_no_trans;
    };

    return std::make_tuple(inner_rank, IndexList(result_left_indices),
                           IndexList(result_right_indices),
                           IndexList(result_indices),
                           to_tensor_op(left_is_no_trans, left_is_trans),
                           to_tensor_op(right_is_no_trans, right_is_trans));
  }
};

// clang-format off
/// Given left and right index lists computes the suggested indices for the left
/// and right args and the result for computing Hadamard product efficiently
// clang-format on
struct HadamardPermutationOptimizer {
  IndexList target_left_indices, target_right_indices, target_result_indices;

  HadamardPermutationOptimizer() = default;
  HadamardPermutationOptimizer(const HadamardPermutationOptimizer&) = default;
  HadamardPermutationOptimizer& operator=(const HadamardPermutationOptimizer&) =
      default;

  HadamardPermutationOptimizer(const IndexList& left_indices,
                               const IndexList& right_indices,
                               const bool prefer_to_permute_left = true)
      : target_left_indices(left_indices),
        target_right_indices(right_indices),
        target_result_indices(prefer_to_permute_left ? right_indices
                                                     : left_indices) {
    TA_ASSERT(left_indices.is_permutation(right_indices));
  }
};

// Forward declarations
template <typename, typename>
class MultExpr;
template <typename, typename, typename>
class ScalMultExpr;

/// Multiplication expression engine

/// \tparam Derived The derived engine type
template <typename Derived>
class ContEngine : public BinaryEngine<Derived> {
 public:
  // Class hierarchy typedefs
  typedef ContEngine<Derived> ContEngine_;      ///< This class type
  typedef BinaryEngine<Derived> BinaryEngine_;  ///< Binary base class type
  typedef ExprEngine<Derived>
      ExprEngine_;  ///< Expression engine base class type

  // Argument typedefs
  typedef typename EngineTrait<Derived>::left_type
      left_type;  ///< The left-hand expression type
  typedef typename EngineTrait<Derived>::right_type
      right_type;  ///< The right-hand expression type

  // Operational typedefs
  typedef typename EngineTrait<Derived>::value_type
      value_type;  ///< The result tile type
  typedef typename EngineTrait<Derived>::scalar_type
      scalar_type;  ///< Tile scalar type
  typedef TiledArray::detail::ContractReduce<
      value_type, typename eval_trait<typename left_type::value_type>::type,
      typename eval_trait<typename right_type::value_type>::type,
      scalar_type>
      op_type;  ///< The tile operation type
  typedef
      typename EngineTrait<Derived>::policy policy;  ///< The result policy type
  typedef typename EngineTrait<Derived>::dist_eval_type
      dist_eval_type;  ///< The distributed evaluator type

  // Meta data typedefs
  typedef typename EngineTrait<Derived>::size_type size_type;  ///< Size type
  typedef typename EngineTrait<Derived>::trange_type
      trange_type;  ///< Tiled range type
  typedef typename EngineTrait<Derived>::shape_type shape_type;  ///< Shape type
  typedef typename EngineTrait<Derived>::pmap_interface
      pmap_interface;  ///< Process map interface type

 protected:
  // Import base class variables to this scope
  using BinaryEngine_::left_;
  using BinaryEngine_::right_;
  using ExprEngine_::indices_;
  using ExprEngine_::perm_;
  using ExprEngine_::permute_tiles_;
  using ExprEngine_::pmap_;
  using ExprEngine_::shape_;
  using ExprEngine_::trange_;
  using ExprEngine_::world_;

 protected:
  scalar_type factor_;  ///< Contraction scaling factor

 private:
  BipartiteIndexList left_indices_;   ///< Target left-hand index list
  BipartiteIndexList right_indices_;  ///< Target right-hand index list
  GEMMTensorArgOp left_op_;           ///< Left-hand permute operation
  GEMMTensorArgOp right_op_;          ///< Right-hand permute operation
  op_type op_;                        ///< Tile operation
  TiledArray::detail::ProcGrid
      proc_grid_;  ///< Process grid for the contraction
  size_type K_;    ///< Inner dimension size

  static unsigned int find(const BipartiteIndexList& indices,
                           const std::string& index_label, unsigned int i,
                           const unsigned int n) {
    for (; i < n; ++i) {
      if (indices[i] == index_label) break;
    }

    return i;
  }

 public:
  /// Constructor

  /// \tparam L The left-hand argument expression type
  /// \tparam R The right-hand argument expression type
  /// \param expr The parent expression
  template <typename L, typename R>
  ContEngine(const MultExpr<L, R>& expr)
      : BinaryEngine_(expr),
        factor_(1),
        left_indices_(),
        right_indices_(),
        left_op_(GEMMTensorArgOp::permute_to_no_trans),
        right_op_(GEMMTensorArgOp::permute_to_no_trans),
        op_(),
        proc_grid_(),
        K_(1u) {}

  /// Constructor

  /// \tparam L The left-hand argument expression type
  /// \tparam R The right-hand argument expression type
  /// \tparam S The expression scalar type
  /// \param expr The parent expression
  template <typename L, typename R, typename S>
  ContEngine(const ScalMultExpr<L, R, S>& expr)
      : BinaryEngine_(expr),
        factor_(expr.factor()),
        left_indices_(),
        right_indices_(),
        left_op_(GEMMTensorArgOp::permute_to_no_trans),
        right_op_(GEMMTensorArgOp::permute_to_no_trans),
        op_(),
        proc_grid_(),
        K_(1u) {}

  // Pull base class functions into this class.
  using ExprEngine_::derived;
  using ExprEngine_::indices;

  /// Set the index list for this expression

  /// This function will set the index list for this expression and its
  /// children such that the number of permutations is minimized. The final
  /// index list may not be set to target, which indicates that the
  /// result of this expression will be permuted to match \c target_indices.
  /// \param target_indices The target index list for this expression
  void perm_indices(const BipartiteIndexList& target_indices) {
    // Only permute if the arguments can be permuted
    if ((left_op_ == GEMMTensorArgOp::permute_to_no_trans) ||
        (right_op_ == GEMMTensorArgOp::permute_to_no_trans)) {
      // Compute ranks
      const unsigned int result_rank = target_indices.size();
      const unsigned int inner_rank =
          (left_.indices().size() + right_.indices().size() - result_rank) >> 1;
      const unsigned int left_outer_rank = left_.indices().size() - inner_rank;

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
        if (left_op_ == GEMMTensorArgOp::permute_to_no_trans) {
          // Copy left-hand target variables to left and result index lists.
          for (unsigned int i = 0u; i < left_outer_rank; ++i) {
            const std::string& var = target_indices[i];
            const_cast<std::string&>(left_indices_[i]) = var;
            const_cast<std::string&>(indices_[i]) = var;
          }

          // Permute the left argument with the new index list.
          left_.perm_indices(left_indices_);
        } else {
          // Copy left-hand outer variables to that of result.
          for (unsigned int i = 0u; i < left_outer_rank; ++i)
            const_cast<std::string&>(indices_[i]) = left_indices_[i];
        }

        if (right_op_ == GEMMTensorArgOp::permute_to_no_trans) {
          // Copy right-hand target variables to right and result variable
          // lists.
          for (unsigned int i = left_outer_rank, j = inner_rank;
               i < result_rank; ++i, ++j) {
            const std::string& var = target_indices[i];
            const_cast<std::string&>(right_indices_[j]) = var;
            const_cast<std::string&>(indices_[i]) = var;
          }

          // Permute the left argument with the new index list.
          right_.perm_indices(right_indices_);
        } else {
          // Copy right-hand outer variables to that of result.
          for (unsigned int i = left_outer_rank, j = inner_rank;
               i < result_rank; ++i, ++j)
            const_cast<std::string&>(indices_[i]) = right_indices_[j];
        }
      }
    }
  }

  /// Initialize the index list of this expression

  /// \note This function does not initialize the child data as is done in
  /// \c BinaryEngine. Instead they are initialized in \c MultContEngine and
  /// \c ScalMultContEngine.
  void init_indices() {
    auto inner_left_indices = inner(left_.indices());
    auto inner_right_indices = inner(right_.indices());
    auto inner_product_type =
        compute_product_type(inner_left_indices, inner_right_indices);
    // prefer to permute the arg with fewest leaves to try to minimize the
    // number of possible permutations
    auto outer_opt = GEMMPermutationOptimizer(
        outer(left_.indices()), outer(right_.indices()),
        left_type::leaves <= right_type::leaves);

    IndexList inner_left_target_indices, inner_right_target_indices,
        inner_result_target_indices;
    if (inner_product_type == TensorProduct::Contraction) {
      auto inner_opt =
          GEMMPermutationOptimizer(inner_left_indices, inner_right_indices,
                                   left_type::leaves <= right_type::leaves);
      inner_left_target_indices = inner_opt.target_left_indices;
      inner_right_target_indices = inner_opt.target_right_indices;
      inner_result_target_indices = inner_opt.target_result_indices;
    } else if (inner_product_type == TensorProduct::Hadamard) {
      auto inner_opt =
          HadamardPermutationOptimizer(inner_left_indices, inner_right_indices,
                                       left_type::leaves <= right_type::leaves);
      inner_left_target_indices = inner_opt.target_left_indices;
      inner_right_target_indices = inner_opt.target_right_indices;
      inner_result_target_indices = inner_opt.target_result_indices;
    }

    left_indices_ = BipartiteIndexList(outer_opt.target_left_indices,
                                       inner_left_target_indices);
    right_indices_ = BipartiteIndexList(outer_opt.target_right_indices,
                                        inner_right_target_indices);
    indices_ = BipartiteIndexList(outer_opt.target_result_indices,
                                  inner_result_target_indices);

    // Here we set the type of permutation that will be applied to the
    // argument tensors. If an argument is in matrix form, permutation of
    // the tiles is disabled.
    if (inner_product_type == TensorProduct::Invalid &&
        (outer_opt.left_op == GEMMTensorArgOp::trans ||
         outer_opt.left_op == GEMMTensorArgOp::no_trans)) {
      left_op_ = outer_opt.left_op;
      left_.permute_tiles(false);
    } else {
      left_.perm_indices(left_indices_);
    }
    if (inner_product_type == TensorProduct::Invalid &&
        (outer_opt.right_op == GEMMTensorArgOp::trans ||
         outer_opt.right_op == GEMMTensorArgOp::no_trans)) {
      right_op_ = outer_opt.right_op;
      right_.permute_tiles(false);
    } else {
      right_.perm_indices(right_indices_);
    }
  }

  /// Initialize result tensor structure

  /// This function will initialize the permutation, tiled range, and shape
  /// for the result tensor as well as the tile operation.
  /// \param target_indices The target index list for the result tensor
  void init_struct(const BipartiteIndexList& target_indices) {
    // Initialize children
    left_.init_struct(left_indices_);
    right_.init_struct(right_indices_);

    // Initialize the tile operation in this function because it is used to
    // evaluate the tiled range and shape.

    const madness::cblas::CBLAS_TRANSPOSE left_op =
        (left_op_ == GEMMTensorArgOp::trans ? madness::cblas::Trans
                                            : madness::cblas::NoTrans);
    const madness::cblas::CBLAS_TRANSPOSE right_op =
        (right_op_ == GEMMTensorArgOp::trans ? madness::cblas::Trans
                                             : madness::cblas::NoTrans);

    if (target_indices != indices_) {
      // Initialize permuted structure
      perm_ = ExprEngine_::make_perm(target_indices);
      op_ = op_type(left_op, right_op, factor_, indices_.size(),
                    left_indices_.size(), right_indices_.size(),
                    (permute_tiles_ ? perm_ : BipartitePermutation{}));
      trange_ = ContEngine_::make_trange(outer(perm_));
      shape_ = ContEngine_::make_shape(outer(perm_));
    } else {
      // Initialize non-permuted structure
      op_ = op_type(left_op, right_op, factor_, indices_.size(),
                    left_indices_.size(), right_indices_.size());
      trange_ = ContEngine_::make_trange();
      shape_ = ContEngine_::make_shape();
    }

    if (ExprEngine_::override_ptr_ && ExprEngine_::override_ptr_->shape) {
      shape_ = shape_.mask(*ExprEngine_::override_ptr_->shape);
    }
  }

  /// Initialize result tensor distribution

  /// This function will initialize the world and process map for the result
  /// tensor.
  /// \param world The world were the result will be distributed
  /// \param pmap The process map for the result tensor tiles
  void init_distribution(World* world, std::shared_ptr<pmap_interface> pmap) {
    const unsigned int inner_rank = op_.gemm_helper().num_contract_ranks();
    const unsigned int left_rank = op_.gemm_helper().left_rank();
    const unsigned int right_rank = op_.gemm_helper().right_rank();
    const unsigned int left_outer_rank = left_rank - inner_rank;

    // Get pointers to the argument sizes
    const auto* MADNESS_RESTRICT const left_tiles_size =
        left_.trange().tiles_range().extent_data();
    const auto* MADNESS_RESTRICT const left_element_size =
        left_.trange().elements_range().extent_data();
    const auto* MADNESS_RESTRICT const right_tiles_size =
        right_.trange().tiles_range().extent_data();
    const auto* MADNESS_RESTRICT const right_element_size =
        right_.trange().elements_range().extent_data();

    // Compute the fused sizes of the contraction
    size_type M = 1ul, m = 1ul, N = 1ul, n = 1ul;
    unsigned int i = 0u;
    for (; i < left_outer_rank; ++i) {
      M *= left_tiles_size[i];
      m *= left_element_size[i];
    }
    for (; i < left_rank; ++i) K_ *= left_tiles_size[i];
    for (i = inner_rank; i < right_rank; ++i) {
      N *= right_tiles_size[i];
      n *= right_element_size[i];
    }

    // Construct the process grid.
    proc_grid_ = TiledArray::detail::ProcGrid(*world, M, N, m, n);

    // Initialize children
    left_.init_distribution(world, proc_grid_.make_row_phase_pmap(K_));
    right_.init_distribution(world, proc_grid_.make_col_phase_pmap(K_));

    // Initialize the process map in not already defined
    if (!pmap) pmap = proc_grid_.make_pmap();
    ExprEngine_::init_distribution(world, pmap);
  }

  /// Tiled range factory function

  /// \param perm The permutation to be applied to the array
  /// \return The result tiled range
  trange_type make_trange(const Permutation& perm = {}) const {
    // Compute iteration limits
    const unsigned int left_rank = op_.gemm_helper().left_rank();
    const unsigned int right_rank = op_.gemm_helper().right_rank();
    const unsigned int inner_rank = op_.gemm_helper().num_contract_ranks();
    const unsigned int left_outer_rank = left_rank - inner_rank;

    // Construct the trange input and compute the gemm sizes
    typename trange_type::Ranges ranges(op_.gemm_helper().result_rank());
    unsigned int i = 0ul;
    for (unsigned int x = 0ul; x < left_outer_rank; ++x, ++i) {
      const unsigned int pi = (perm ? perm[i] : i);
      ranges[pi] = left_.trange().data()[x];
    }
    for (unsigned int x = inner_rank; x < right_rank; ++x, ++i) {
      const unsigned int pi = (perm ? perm[i] : i);
      ranges[pi] = right_.trange().data()[x];
    }

#ifndef NDEBUG

    // Check that the contracted dimensions have congruent tilings
    for (unsigned int l = left_outer_rank, r = 0ul; l < left_rank; ++l, ++r) {
      if (!is_congruent(left_.trange().data()[l], right_.trange().data()[r])) {
        if (TiledArray::get_default_world().rank() == 0) {
          TA_USER_ERROR_MESSAGE(
              "The contracted dimensions of the left- "
              "and right-hand arguments are not congruent:"
              << "\n    left  = " << left_.trange()
              << "\n    right = " << right_.trange());

          TA_EXCEPTION(
              "The contracted dimensions of the left- and "
              "right-hand expressions are not congruent.");
        }

        TA_EXCEPTION(
            "The contracted dimensions of the left- and "
            "right-hand expressions are not congruent.");
      }
    }
#endif  // NDEBUG

    return trange_type(ranges.begin(), ranges.end());
  }

  /// Non-permuting shape factory function

  /// \return The result shape
  shape_type make_shape() const {
    const TiledArray::math::GemmHelper shape_gemm_helper(
        madness::cblas::NoTrans, madness::cblas::NoTrans,
        op_.gemm_helper().result_rank(), op_.gemm_helper().left_rank(),
        op_.gemm_helper().right_rank());
    return left_.shape().gemm(right_.shape(), factor_, shape_gemm_helper);
  }

  /// Permuting shape factory function

  /// \param perm The permutation to be applied to the array
  /// \return The result shape
  shape_type make_shape(const Permutation& perm) const {
    const TiledArray::math::GemmHelper shape_gemm_helper(
        madness::cblas::NoTrans, madness::cblas::NoTrans,
        op_.gemm_helper().result_rank(), op_.gemm_helper().left_rank(),
        op_.gemm_helper().right_rank());
    return left_.shape().gemm(right_.shape(), factor_, shape_gemm_helper, perm);
  }

  dist_eval_type make_dist_eval() const {
    // Define the impl type
    typedef TiledArray::detail::Summa<typename left_type::dist_eval_type,
                                      typename right_type::dist_eval_type,
                                      op_type, typename Derived::policy>
        impl_type;

    typename left_type::dist_eval_type left = left_.make_dist_eval();
    typename right_type::dist_eval_type right = right_.make_dist_eval();

    std::shared_ptr<impl_type> pimpl =
        std::make_shared<impl_type>(left, right, *world_, trange_, shape_,
                                    pmap_, perm_, op_, K_, proc_grid_);

    return dist_eval_type(pimpl);
  }

  /// Expression identification tag

  /// \return An expression tag used to identify this expression
  std::string make_tag() const {
    std::stringstream ss;
    ss << "[*]";
    if (factor_ != scalar_type(1)) ss << "[" << factor_ << "]";
    return ss.str();
  }

  /// Expression print

  /// \param os The output stream
  /// \param target_indices The target index list for this expression
  void print(ExprOStream os, const BipartiteIndexList& target_indices) const {
    ExprEngine_::print(os, target_indices);
    os.inc();
    left_.print(os, left_indices_);
    right_.print(os, right_indices_);
    os.dec();
  }

};  // class ContEngine

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_CONT_ENGINE_H__INCLUDED
