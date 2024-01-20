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
#include <TiledArray/expressions/permopt.h>
#include <TiledArray/proc_grid.h>
#include <TiledArray/tensor/utility.h>
#include <TiledArray/tile_op/contract_reduce.h>
#include <TiledArray/tile_op/mult.h>

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
  using BinaryEngine_::left_indices_;
  using BinaryEngine_::left_inner_permtype_;
  using BinaryEngine_::left_outer_permtype_;
  using BinaryEngine_::right_;
  using BinaryEngine_::right_indices_;
  using BinaryEngine_::right_inner_permtype_;
  using BinaryEngine_::right_outer_permtype_;
  using ExprEngine_::implicit_permute_inner_;
  using ExprEngine_::implicit_permute_outer_;
  using ExprEngine_::indices_;
  using ExprEngine_::perm_;
  using ExprEngine_::pmap_;
  using ExprEngine_::shape_;
  using ExprEngine_::trange_;
  using ExprEngine_::world_;

 protected:
  scalar_type factor_;  ///< Contraction scaling factor

 protected:
  op_type op_;  ///< Tile operation

  // tile types of the result and (after evaluation) left and right arguments
  using result_tile_type = value_type;
  using left_tile_type = typename EngineTrait<left_type>::eval_type;
  using right_tile_type = typename EngineTrait<right_type>::eval_type;

  // tile element types of the result and (after evaluation) left and right
  // arguments
  using result_tile_element_type = typename result_tile_type::value_type;
  using left_tile_element_type = typename left_tile_type::value_type;
  using right_tile_element_type = typename right_tile_type::value_type;

  std::function<void(result_tile_element_type&, const left_tile_element_type&,
                     const right_tile_element_type&)>
      element_nonreturn_op_;  ///< Tile element operation (only non-null for
                              ///< nested tensor expressions)
  std::function<result_tile_element_type(const left_tile_element_type&,
                                         const right_tile_element_type&)>
      element_return_op_;  ///< Same as inner_tile_nonreturn_op_ but returns
                           ///< the result
  TiledArray::detail::ProcGrid
      proc_grid_;    ///< Process grid for the contraction
  size_type K_ = 1;  ///< Inner dimension size

  static unsigned int find(const BipartiteIndexList& indices,
                           const std::string& index_label, unsigned int i,
                           const unsigned int n) {
    for (; i < n; ++i) {
      if (indices[i] == index_label) break;
    }

    return i;
  }

  TensorProduct product_type_ = TensorProduct::Invalid;
  TensorProduct inner_product_type_ = TensorProduct::Invalid;

  /// \return the product type
  TensorProduct product_type() const {
    TA_ASSERT(product_type_ !=
              TensorProduct::Invalid);  // init_indices() must initialize this
    /// only Hadamard and contraction are supported now
    TA_ASSERT(product_type_ == TensorProduct::Hadamard ||
              product_type_ == TensorProduct::Contraction);
    return product_type_;
  }

  /// \return the inner product type
  TensorProduct inner_product_type() const {
    TA_ASSERT(inner_product_type_ !=
              TensorProduct::Invalid);  // init_indices() must initialize this
    /// only Hadamard, contraction, and scale are supported now
    TA_ASSERT(inner_product_type_ == TensorProduct::Hadamard ||
              inner_product_type_ == TensorProduct::Contraction ||
              inner_product_type_ == TensorProduct::Scale);
    return inner_product_type_;
  }

 public:
  /// Constructor

  /// \tparam L The left-hand argument expression type
  /// \tparam R The right-hand argument expression type
  /// \param expr The parent expression
  template <typename L, typename R>
  ContEngine(const MultExpr<L, R>& expr) : BinaryEngine_(expr), factor_(1) {}

  /// Constructor

  /// \tparam L The left-hand argument expression type
  /// \tparam R The right-hand argument expression type
  /// \tparam S The expression scalar type
  /// \param expr The parent expression
  template <typename L, typename R, typename S>
  ContEngine(const ScalMultExpr<L, R, S>& expr)
      : BinaryEngine_(expr), factor_(expr.factor()) {}

  // Pull base class functions into this class.
  using ExprEngine_::derived;
  using ExprEngine_::indices;

  /// Set the index list for this expression

  /// If arguments can be permuted freely and \p target_indices are
  /// well-partitioned into left and right args' indices it is possible
  /// to permute left and right args to order their free indices
  /// the order that \p target_indices requires.
  /// \param target_indices The target index list for this expression
  /// \warning this does not take into account the ranks of the args to decide
  /// whether it makes sense to permute them or the result
  /// \todo take into account the ranks of the args to decide
  /// whether it makes sense to permute them or the result
  void perm_indices(const BipartiteIndexList& target_indices) {
    // assert that init_indices has been called
    TA_ASSERT(left_.indices() && right_.indices());
    if (!this->implicit_permute()) {
      this->template init_indices_<TensorProduct::Contraction>(target_indices);

      // propagate the indices down the tree, if needed
      if (left_indices_ != left_.indices()) {
        left_.perm_indices(left_indices_);
      }
      if (right_indices_ != right_.indices()) {
        right_.perm_indices(right_indices_);
      }
    }
  }

  /// Initialize the index list of this expression

  /// \note This function does not initialize the child data as is done in
  /// \c BinaryEngine. Instead they are initialized in \c MultEngine and
  /// \c ScalMultEngine since they need child indices to determine the type of
  /// product
  void init_indices(bool children_initialized = false) {
    if (!children_initialized) {
      left_.init_indices();
      right_.init_indices();
    }

    this->template init_indices_<TensorProduct::Contraction>();
  }

  /// Initialize the index list of this expression

  /// \param target_indices the target index list
  /// \note This function does not initialize the child data as is done in
  /// \c BinaryEngine. Instead they are initialized in \c MultEngine and
  /// \c ScalMultEngine since they need child indices to determine the type of
  /// product
  void init_indices(const BipartiteIndexList& target_indices) {
    init_indices();
    perm_indices(target_indices);
  }

  /// Initialize result tensor structure

  /// This function will initialize the permutation, tiled range, and shape
  /// for the result tensor as well as the tile operation.
  /// \param target_indices The target index list for the result tensor
  void init_struct(const BipartiteIndexList& target_indices) {
    // precondition checks
    // 1. if ToT inner tile op has been initialized
    if constexpr (TiledArray::detail::is_tensor_of_tensor_v<value_type>) {
      TA_ASSERT(element_nonreturn_op_);
      TA_ASSERT(element_return_op_);
    }

    // Initialize children
    left_.init_struct(left_indices_);
    right_.init_struct(right_indices_);

    // Initialize the tile operation in this function because it is used to
    // evaluate the tiled range and shape.

    const auto left_op = to_cblas_op(left_outer_permtype_);
    const auto right_op = to_cblas_op(right_outer_permtype_);

    // initialize perm_
    this->init_perm(target_indices);

    // initialize op_, trange_, and shape_ which only refer to the outer modes
    if (outer(target_indices) != outer(indices_)) {
      const auto outer_perm = outer(perm_);
      // Initialize permuted structure
      if constexpr (!TiledArray::detail::is_tensor_of_tensor_v<value_type>) {
        op_ = op_type(left_op, right_op, factor_, outer_size(indices_),
                      outer_size(left_indices_), outer_size(right_indices_),
                      (!implicit_permute_outer_ ? outer_perm : Permutation{}));
      } else {
        // factor_ is absorbed into inner_tile_nonreturn_op_
        op_ = op_type(left_op, right_op, scalar_type(1), outer_size(indices_),
                      outer_size(left_indices_), outer_size(right_indices_),
                      (!implicit_permute_outer_ ? outer_perm : Permutation{}),
                      this->element_nonreturn_op_);
      }
      trange_ = ContEngine_::make_trange(outer_perm);
      shape_ = ContEngine_::make_shape(outer_perm);
    } else {
      // Initialize non-permuted structure
      if constexpr (!TiledArray::detail::is_tensor_of_tensor_v<value_type>) {
        op_ = op_type(left_op, right_op, factor_, outer_size(indices_),
                      outer_size(left_indices_), outer_size(right_indices_));
      } else {
        // factor_ is absorbed into inner_tile_nonreturn_op_
        op_ = op_type(left_op, right_op, scalar_type(1), outer_size(indices_),
                      outer_size(left_indices_), outer_size(right_indices_),
                      BipartitePermutation{}, this->element_nonreturn_op_);
      }
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
  void init_distribution(World* world,
                         std::shared_ptr<const pmap_interface> pmap) {
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
        math::blas::NoTranspose, math::blas::NoTranspose,
        op_.gemm_helper().result_rank(), op_.gemm_helper().left_rank(),
        op_.gemm_helper().right_rank());
    return left_.shape().gemm(right_.shape(), factor_, shape_gemm_helper);
  }

  /// Permuting shape factory function

  /// \param perm The permutation to be applied to the array
  /// \return The result shape
  shape_type make_shape(const Permutation& perm) const {
    const TiledArray::math::GemmHelper shape_gemm_helper(
        math::blas::NoTranspose, math::blas::NoTranspose,
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

 protected:
  void init_inner_tile_op(const IndexList& inner_target_indices) {
    if constexpr (TiledArray::detail::is_tensor_of_tensor_v<result_tile_type>) {
      constexpr bool tot_x_tot = TiledArray::detail::is_tensor_of_tensor_v<
          result_tile_type, left_tile_type, right_tile_type>;
      const auto inner_prod = this->inner_product_type();
      TA_ASSERT(inner_prod == TensorProduct::Contraction ||
                inner_prod == TensorProduct::Hadamard ||
                inner_prod == TensorProduct::Scale);
      if (inner_prod == TensorProduct::Contraction) {
        TA_ASSERT(tot_x_tot);
        if constexpr (tot_x_tot) {
          using op_type = TiledArray::detail::ContractReduce<
              result_tile_element_type, left_tile_element_type,
              right_tile_element_type, scalar_type>;
          // factor_ is absorbed into inner_tile_nonreturn_op_
          auto contrreduce_op =
              (inner_target_indices != inner(this->indices_))
                  ? op_type(to_cblas_op(this->left_inner_permtype_),
                            to_cblas_op(this->right_inner_permtype_),
                            this->factor_, inner_size(this->indices_),
                            inner_size(this->left_indices_),
                            inner_size(this->right_indices_),
                            (!this->implicit_permute_inner_ ? inner(this->perm_)
                                                            : Permutation{}))
                  : op_type(to_cblas_op(this->left_inner_permtype_),
                            to_cblas_op(this->right_inner_permtype_),
                            this->factor_, inner_size(this->indices_),
                            inner_size(this->left_indices_),
                            inner_size(this->right_indices_));
          this->element_nonreturn_op_ =
              [contrreduce_op](result_tile_element_type& result,
                               const left_tile_element_type& left,
                               const right_tile_element_type& right) {
                contrreduce_op(result, left, right);
              };
        }  // ToT x ToT
      } else if (inner_prod == TensorProduct::Hadamard) {
        TA_ASSERT(tot_x_tot);
        if constexpr (tot_x_tot) {
          // inner tile op depends on the outer op ... e.g. if outer op
          // is contract then inner must implement (ternary) multiply-add;
          // if the outer is hadamard then the inner is binary multiply
          const auto outer_prod = this->product_type();
          if (this->factor_ == scalar_type{1}) {
            using base_op_type =
                TiledArray::detail::Mult<result_tile_element_type,
                                         left_tile_element_type,
                                         right_tile_element_type, false, false>;
            using op_type = TiledArray::detail::BinaryWrapper<
                base_op_type>;  // can't consume inputs if they are used
                                // multiple times, e.g. when outer op is gemm
            auto mult_op =
                (inner_target_indices != inner(this->indices_))
                    ? op_type(base_op_type(), !this->implicit_permute_inner_
                                                  ? inner(this->perm_)
                                                  : Permutation{})
                    : op_type(base_op_type());
            this->element_nonreturn_op_ =
                [mult_op, outer_prod](result_tile_element_type& result,
                                      const left_tile_element_type& left,
                                      const right_tile_element_type& right) {
                  if (outer_prod == TensorProduct::Hadamard)
                    result = mult_op(left, right);
                  else {
                    TA_ASSERT(outer_prod == TensorProduct::Hadamard ||
                              outer_prod == TensorProduct::Contraction);
                    // there is currently no fused MultAdd ternary Op, only Add
                    // and Mult thus implement this as 2 separate steps
                    // TODO optimize by implementing (ternary) MultAdd
                    if (empty(result))
                      result = mult_op(left, right);
                    else {
                      auto result_increment = mult_op(left, right);
                      add_to(result, result_increment);
                    }
                  }
                };
          } else {
            using base_op_type = TiledArray::detail::ScalMult<
                result_tile_element_type, left_tile_element_type,
                right_tile_element_type, scalar_type, false, false>;
            using op_type = TiledArray::detail::BinaryWrapper<
                base_op_type>;  // can't consume inputs if they are used
                                // multiple times, e.g. when outer op is gemm
            auto mult_op = (inner_target_indices != inner(this->indices_))
                               ? op_type(base_op_type(this->factor_),
                                         !this->implicit_permute_inner_
                                             ? inner(this->perm_)
                                             : Permutation{})
                               : op_type(base_op_type(this->factor_));
            this->element_nonreturn_op_ =
                [mult_op, outer_prod](result_tile_element_type& result,
                                      const left_tile_element_type& left,
                                      const right_tile_element_type& right) {
                  TA_ASSERT(outer_prod == TensorProduct::Hadamard ||
                            outer_prod == TensorProduct::Contraction);
                  if (outer_prod == TensorProduct::Hadamard)
                    result = mult_op(left, right);
                  else {
                    // there is currently no fused MultAdd ternary Op, only Add
                    // and Mult thus implement this as 2 separate steps
                    // TODO optimize by implementing (ternary) MultAdd
                    if (empty(result))
                      result = mult_op(left, right);
                    else {
                      auto result_increment = mult_op(left, right);
                      add_to(result, result_increment);
                    }
                  }
                };
          }
        }  // ToT x T or T x ToT
      } else if (inner_prod == TensorProduct::Scale) {
        TA_ASSERT(!tot_x_tot);
        constexpr bool tot_x_t =
            TiledArray::detail::is_tensor_of_tensor_v<result_tile_type,
                                                      left_tile_type> &&
            TiledArray::detail::is_tensor_v<right_tile_type>;
        constexpr bool t_x_tot =
            TiledArray::detail::is_tensor_of_tensor_v<result_tile_type,
                                                      right_tile_type> &&
            TiledArray::detail::is_tensor_v<left_tile_type>;
        if constexpr (tot_x_t || t_x_tot) {
          using arg_tile_element_type =
              std::conditional_t<tot_x_t, left_tile_element_type,
                                 right_tile_element_type>;
          using scalar_type =
              std::conditional_t<tot_x_t, right_tile_element_type,
                                 left_tile_element_type>;

          auto scal_op = [perm = !this->implicit_permute_inner_
                                     ? inner(this->perm_)
                                     : Permutation{}](
                             const left_tile_element_type& left,
                             const right_tile_element_type& right)
              -> result_tile_element_type {
            using TiledArray::scale;
            if constexpr (tot_x_t) {
              if (perm)
                return scale(left, right, perm);
              else
                return scale(left, right);
            } else if constexpr (t_x_tot) {
              if (perm)
                return scale(right, left, perm);
              else
                return scale(right, left);
            } else
              abort();  // unreachable
          };
          this->element_nonreturn_op_ =
              [scal_op](result_tile_element_type& result,
                        const left_tile_element_type& left,
                        const right_tile_element_type& right) {
                result = scal_op(left, right);
              };
        }
      } else
        abort();  // unsupported TensorProduct type
      TA_ASSERT(element_nonreturn_op_);
      this->element_return_op_ = [inner_tile_nonreturn_op =
                                      this->element_nonreturn_op_](
                                     const left_tile_element_type& left,
                                     const right_tile_element_type& right) {
        result_tile_element_type result;
        inner_tile_nonreturn_op(result, left, right);
        return result;
      };
    }
  }

};  // class ContEngine

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_CONT_ENGINE_H__INCLUDED
