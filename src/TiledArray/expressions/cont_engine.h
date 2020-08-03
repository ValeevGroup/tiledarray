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
  using ExprEngine_::perm_;
  using ExprEngine_::permute_tiles_;
  using ExprEngine_::pmap_;
  using ExprEngine_::shape_;
  using ExprEngine_::trange_;
  using ExprEngine_::vars_;
  using ExprEngine_::world_;

 private:
  typedef enum {
    no_trans = 1,
    trans = 2,
    permute_to_no_trans = 3,
  } TensorOp;

 protected:
  scalar_type factor_;  ///< Contraction scaling factor

 private:
  VariableList left_vars_;   ///< Left-hand variable list
  VariableList right_vars_;  ///< Right-hand variable list
  TensorOp left_op_;         ///< Left-hand operation
  TensorOp right_op_;        ///< Right-hand operation
  op_type op_;               ///< Tile operation
  TiledArray::detail::ProcGrid
      proc_grid_;  ///< Process grid for the contraction
  size_type K_;    ///< Inner dimension size

  static unsigned int find(const VariableList& vars, std::string var,
                           unsigned int i, const unsigned int n) {
    for (; i < n; ++i) {
      if (vars[i] == var) break;
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
        left_vars_(),
        right_vars_(),
        left_op_(permute_to_no_trans),
        right_op_(permute_to_no_trans),
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
        left_vars_(),
        right_vars_(),
        left_op_(permute_to_no_trans),
        right_op_(permute_to_no_trans),
        op_(),
        proc_grid_(),
        K_(1u) {}

  // Pull base class functions into this class.
  using ExprEngine_::derived;
  using ExprEngine_::vars;

  /// Set the variable list for this expression

  /// This function will set the variable list for this expression and its
  /// children such that the number of permutations is minimized. The final
  /// variable list may not be set to target, which indicates that the
  /// result of this expression will be permuted to match \c target_vars.
  /// \param target_vars The target variable list for this expression
  void perm_vars(const VariableList& target_vars) {
    // Only permute if the arguments can be permuted
    if ((left_op_ == permute_to_no_trans) ||
        (right_op_ == permute_to_no_trans)) {
      // Compute ranks
      const unsigned int result_rank = target_vars.dim();
      const unsigned int inner_rank =
          (left_.vars().dim() + right_.vars().dim() - result_rank) >> 1;
      const unsigned int left_outer_rank = left_.vars().dim() - inner_rank;

      // Check that the left- and right-hand outer variables are correctly
      // partitioned in the target variable list.
      bool target_partitioned = true;
      for (unsigned int i = 0u; i < left_outer_rank; ++i)
        target_partitioned =
            target_partitioned && (find(target_vars, left_vars_[i], 0u,
                                        left_outer_rank) < left_outer_rank);

      // If target is properly partitioned, then arguments can be permuted
      // to fit the target.
      if (target_partitioned) {
        if (left_op_ == permute_to_no_trans) {
          // Copy left-hand target variables to left and result variable lists.
          for (unsigned int i = 0u; i < left_outer_rank; ++i) {
            const std::string& var = target_vars[i];
            const_cast<std::string&>(left_vars_[i]) = var;
            const_cast<std::string&>(vars_[i]) = var;
          }

          // Permute the left argument with the new variable list.
          left_.perm_vars(left_vars_);
        } else {
          // Copy left-hand outer variables to that of result.
          for (unsigned int i = 0u; i < left_outer_rank; ++i)
            const_cast<std::string&>(vars_[i]) = left_vars_[i];
        }

        if (right_op_ == permute_to_no_trans) {
          // Copy right-hand target variables to right and result variable
          // lists.
          for (unsigned int i = left_outer_rank, j = inner_rank;
               i < result_rank; ++i, ++j) {
            const std::string& var = target_vars[i];
            const_cast<std::string&>(right_vars_[j]) = var;
            const_cast<std::string&>(vars_[i]) = var;
          }

          // Permute the left argument with the new variable list.
          right_.perm_vars(right_vars_);
        } else {
          // Copy right-hand outer variables to that of result.
          for (unsigned int i = left_outer_rank, j = inner_rank;
               i < result_rank; ++i, ++j)
            const_cast<std::string&>(vars_[i]) = right_vars_[j];
        }
      }
    }
  }

  /// Initialize the variable list of this expression

  /// \note This function does not initialize the child data as is done in
  /// \c BinaryEngine. Instead they are initialized in \c MultContEngine and
  /// \c ScalMultContEngine.
  void init_vars() {
    const unsigned int left_rank = left_.vars().dim();
    const unsigned int right_rank = right_.vars().dim();

    // Get non-const references to the argument variable lists.
    auto& left_vars =
        const_cast<container::svector<std::string>&>(left_vars_.data());
    left_vars.reserve(left_rank);
    auto& right_vars =
        const_cast<container::svector<std::string>&>(right_vars_.data());
    right_vars.reserve(right_rank);
    auto& result_vars =
        const_cast<container::svector<std::string>&>(vars_.data());
    result_vars.reserve(std::max(left_rank, right_rank));

    // Extract left-most result and inner variables from the left-hand argument.
    for (unsigned int i = 0ul; i < left_rank; ++i) {
      const std::string& var = left_.vars()[i];
      if (find(right_.vars(), var, 0u, right_rank) == right_rank) {
        // Store outer left variable
        left_vars.push_back(var);
        result_vars.push_back(var);
      } else {
        // Store inner left variable
        right_vars.push_back(var);
      }
    }

    // Compute the inner and outer dimension ranks.
    const unsigned int inner_rank = right_vars.size();
    const unsigned int left_outer_rank = left_vars.size();
    const unsigned int right_outer_rank = right_rank - inner_rank;
    const unsigned int result_rank = left_outer_rank + right_outer_rank;

    // Resize result variables if necessary.
    result_vars.reserve(result_rank);

    // Check for an outer product
    if (inner_rank == 0u) {
      // Extract the right most outer variables from right hand argument.
      for (unsigned int i = 0ul; i < right_rank; ++i) {
        const std::string& var = right_.vars()[i];
        right_vars.push_back(var);
        result_vars.push_back(var);
      }
      return;  // Quick exit
    }

    // Initialize flags that will be used to determine the type of permutation
    // that will be applied to the arguments (i.e. no permutation, transpose,
    // or arbitrary permutation).
    bool inner_vars_ordered = true, left_is_no_trans = true,
         left_is_trans = true, right_is_no_trans = true, right_is_trans = true;

    // If the inner variable lists of the arguments are not in the same
    // order, one of them will need to be permuted. Here, we determine which
    // argument, left or right, will be permuted if a permutation is
    // required. The argument with the lowest rank is preferred since it is
    // likely to have the smaller memory footprint, or the fewest leaves to
    // minimize the number of permutations in the expression.
    const bool perm_left =
        (left_rank < right_rank) || ((left_rank == right_rank) &&
                                     (left_type::leaves <= right_type::leaves));

    // Extract variables from the right-hand argument, collect information
    // about the layout of the variable lists, and ensure the inner variable
    // lists are in the same order.
    for (unsigned int i = 0ul; i < right_rank; ++i) {
      const std::string& var = right_.vars()[i];
      const unsigned int j = find(left_.vars(), var, 0u, left_rank);
      if (j == left_rank) {
        // Store outer right variable
        right_vars.push_back(var);
        result_vars.push_back(var);
      } else {
        const unsigned int x = left_vars.size() - left_outer_rank;

        // Collect information about the relative position of variables
        inner_vars_ordered = inner_vars_ordered && (right_vars[x] == var);
        left_is_no_trans = left_is_no_trans && (j >= left_outer_rank);
        left_is_trans = left_is_trans && (j < inner_rank);
        right_is_no_trans = right_is_no_trans && (i < inner_rank);
        right_is_trans = right_is_trans && (i >= right_outer_rank);

        // Store inner right variable
        if (inner_vars_ordered) {
          // Left and right inner variable list order is equal.
          left_vars.push_back(var);
        } else if (perm_left) {
          // Permute left so we need to store inner variables according to
          // the order of the right-hand argument.
          left_vars.push_back(var);
          right_vars[x] = var;
          left_is_no_trans = left_is_trans = false;
        } else {
          // Permute right so we need to store inner variables according to
          // the order of the left-hand argument.
          left_vars.push_back(right_vars[x]);
          right_is_no_trans = right_is_trans = false;
        }
      }
    }

    // Here we set the type of permutation that will be applied to the
    // argument tensors. If an argument is in matrix form, permutation of
    // the tiles is disabled.
    if (left_is_no_trans) {
      left_op_ = no_trans;
      left_.permute_tiles(false);
    } else if (left_is_trans) {
      left_op_ = trans;
      left_.permute_tiles(false);
    } else {
      left_.perm_vars(left_vars_);
    }
    if (right_is_no_trans) {
      right_op_ = no_trans;
      right_.permute_tiles(false);
    } else if (right_is_trans) {
      right_op_ = trans;
      right_.permute_tiles(false);
    } else {
      right_.perm_vars(right_vars_);
    }
  }

  /// Initialize result tensor structure

  /// This function will initialize the permutation, tiled range, and shape
  /// for the result tensor as well as the tile operation.
  /// \param target_vars The target variable list for the result tensor
  void init_struct(const VariableList& target_vars) {
    // Initialize children
    left_.init_struct(left_vars_);
    right_.init_struct(right_vars_);

    // Initialize the tile operation in this function because it is used to
    // evaluate the tiled range and shape.

    const madness::cblas::CBLAS_TRANSPOSE left_op =
        (left_op_ == trans ? madness::cblas::Trans : madness::cblas::NoTrans);
    const madness::cblas::CBLAS_TRANSPOSE right_op =
        (right_op_ == trans ? madness::cblas::Trans : madness::cblas::NoTrans);

    if (target_vars != vars_) {
      // Initialize permuted structure
      perm_ = ExprEngine_::make_perm(target_vars);
      op_ =
          op_type(left_op, right_op, factor_, vars_.dim(), left_vars_.dim(),
                  right_vars_.dim(), (permute_tiles_ ? perm_ : Permutation()));
      trange_ = ContEngine_::make_trange(perm_);
      shape_ = ContEngine_::make_shape(perm_);
    } else {
      // Initialize non-permuted structure
      op_ = op_type(left_op, right_op, factor_, vars_.dim(), left_vars_.dim(),
                    right_vars_.dim());
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
  trange_type make_trange(const Permutation& perm = Permutation()) const {
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
  /// \param target_vars The target variable list for this expression
  void print(ExprOStream os, const VariableList& target_vars) const {
    ExprEngine_::print(os, target_vars);
    os.inc();
    left_.print(os, left_vars_);
    right_.print(os, right_vars_);
    os.dec();
  }

};  // class ContEngine

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_CONT_ENGINE_H__INCLUDED
