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
#include <TiledArray/dist_eval/unary_eval.h>
#include <TiledArray/expressions/binary_engine.h>
#include <TiledArray/expressions/contraction_retile.h>
#include <TiledArray/expressions/mixed_retile_config.h>
#include <TiledArray/expressions/permopt.h>
#include <TiledArray/pmap/slabbed_pmap.h>
#include <TiledArray/pmap/user_pmap.h>
#include <TiledArray/proc_grid.h>
#include <TiledArray/tensor/arena_einsum.h>
#include <TiledArray/tensor/utility.h>
#include <TiledArray/tile_op/batched_contract_reduce.h>
#include <TiledArray/tile_op/contract_reduce.h>
#include <TiledArray/tile_op/mult.h>
#include <TiledArray/tile_op/noop.h>

namespace TiledArray {
namespace expressions {

// Forward declarations
template <typename, typename>
class MultExpr;
template <typename, typename, typename>
class ScalMultExpr;
template <typename, typename>
class DotInnerExpr;

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

  /// True when both operand tiles are nested (ToT) but the result tile is a
  /// plain tensor of scalars: the inner (nested) modes are fully contracted
  /// (dotted) away to a scalar per outer cell. This is the dot_inner regime.
  /// See also the sibling predicate ContractReduce::denest_to_scalar
  /// (contract_reduce.h); keep the two in sync.
  static constexpr bool denest_to_scalar =
      TiledArray::detail::is_tensor_of_tensor_v<left_tile_type,
                                                right_tile_type> &&
      !TiledArray::detail::is_tensor_of_tensor_v<result_tile_type>;

  // -- Mixed (plain x arena-ToT -> arena-ToT) auto-retile detection ----------
  // Compile-time tile-family predicates mirroring the SUMMA trait at
  // dist_eval/contraction_eval.h (is_arena_tot_v): an arena-backed ToT outer
  // tile is any TA::Tensor<...> whose inner cell type is an ArenaTensor; a plain
  // tensor tile is any TA::Tensor<...> that is NOT a tensor-of-tensor. The
  // env-gated auto-retile (maybe_make_retile_plan_) only compiles in for the
  // mixed shape, so every non-mixed instantiation is a strict no-op.
  template <typename Tile>
  static constexpr bool is_arena_tot_tile_v =
      TiledArray::detail::is_tensor_helper<Tile>::value &&
      TiledArray::is_arena_tensor_v<typename Tile::value_type>;

  template <typename Tile>
  static constexpr bool is_plain_tensor_tile_v =
      TiledArray::detail::is_tensor_helper<Tile>::value &&
      !TiledArray::detail::is_tensor_of_tensor_v<Tile>;

  /// True iff the result is an arena-backed ToT and exactly one operand is plain
  /// dense while the other is arena-ToT -- the only shape the env-gated
  /// auto-retile coarsens. Pure-plain contractions (the MPQC default path) have
  /// a non-arena result and so are never mixed: the retile path stays compiled
  /// out and uninstantiated for them.
  static constexpr bool is_mixed_ =
      is_arena_tot_tile_v<value_type> &&
      ((is_plain_tensor_tile_v<left_tile_type> &&
        is_arena_tot_tile_v<right_tile_type>) ||
       (is_arena_tot_tile_v<left_tile_type> &&
        is_plain_tensor_tile_v<right_tile_type>));
  /// When mixed, whether the PLAIN operand is the LEFT argument (so its external
  /// lives on SUMMA role M); else the plain operand is RIGHT (role N).
  static constexpr bool left_is_plain_ = is_plain_tensor_tile_v<left_tile_type>;

  std::function<void(result_tile_element_type&, const left_tile_element_type&,
                     const right_tile_element_type&)>
      element_nonreturn_op_;  ///< Tile element operation (only non-null for
                              ///< nested tensor expressions)
  std::function<result_tile_element_type(const left_tile_element_type&,
                                         const right_tile_element_type&)>
      element_return_op_;  ///< Same as element_nonreturn_op_ but returns
                           ///< the result
  std::function<result_tile_type(const left_tile_type&,
                                 const right_tile_type&)>
      arena_hadamard_tile_op_;  ///< Whole-tile op for a Hadamard-outer +
                                ///< contraction-inner product on arena
                                ///< (view-inner-cell) ToT tiles, where a
                                ///< value-returning per-cell op cannot be
                                ///< used; null otherwise
  std::function<void(result_tile_type&, const left_tile_type&,
                     const right_tile_type&, const math::GemmHelper&)>
      arena_strided_dgemm_ce_e_tile_op_;  ///< whole-tile ce+e strided DGEMM op
                                          ///< (arena inner OUTER-PRODUCT under
                                          ///< an outer contraction); null
                                          ///< otherwise
  std::function<void(result_tile_type&, const left_tile_type&,
                     const right_tile_type&, const math::GemmHelper&)>
      arena_strided_dgemm_ce_ce_right_tile_op_;  ///< whole-tile ce+ce strided
                                                 ///< DGEMM op (arena inner
                                                 ///< CONTRACTION under an outer
                                                 ///< contraction;
                                                 ///< right-external rides BLAS
                                                 ///< M, left-external rides an
                                                 ///< outer loop); null
                                                 ///< otherwise. Mutually
                                                 ///< exclusive with
                                                 ///< arena_strided_dgemm_ce_e_tile_op_
                                                 ///< (disjoint
                                                 ///< num_contract_ranks()
                                                 ///< gates)
  std::function<void(result_tile_type&, const left_tile_type&,
                     const right_tile_type&, const math::GemmHelper&)>
      arena_strided_dgemm_ce_ce_left_tile_op_;  ///< whole-tile ce+ce strided
                                                ///< DGEMM op, LEFT-clean
                                                ///< mirror: left-external rides
                                                ///< BLAS M, right-external
                                                ///< rides an outer loop.
                                                ///< Mutually exclusive with the
                                                ///< ce_e and ce_ce_right ops.
  using arena_plan_storage_t =
      TiledArray::detail::arena_plan_storage_t<result_tile_type, left_tile_type,
                                               right_tile_type>;
  TA_NO_UNIQUE_ADDRESS arena_plan_storage_t arena_plan_;
  TiledArray::detail::ProcGrid
      proc_grid_;    ///< Process grid for the contraction
  size_type K_ = 1;  ///< Inner dimension size (# of tiles, per slab for
                     ///< general products)
  // General (fused + contracted + free indices) products only:
  unsigned int n_fused_modes_ = 0;  ///< # of leading fused (outer) modes
  size_type n_slabs_ = 1;           ///< # of fused-index tile slabs (COARSE
                                    ///< count == nh_ in the Summa; for an
                                    ///< active COARSEN-H plan < n_slabs_u_)
  size_type n_slabs_u_ = 1;         ///< # of FINE (U) fused-index tile slabs ==
                                    ///< how many H slabs operands/result are
                                    ///< physically tiled into. == n_slabs_
                                    ///< except under an active COARSEN-H plan.
                                    ///< The operand/result SlabbedPmap
                                    ///< replication uses THIS (the arrays carry
                                    ///< U-H tiles); the SUMMA grid/step geometry
                                    ///< uses the coarse n_slabs_.
  bool general_repermute_ = false;  ///< whether the target layout differs
                                    ///< from the canonical result layout, so
                                    ///< the evaluated result is re-permuted
                                    ///< by a streaming unary eval
  size_type proc_h_ = 1;            ///< process-grid extent along the slab (h)
                                    ///< axis of the 3-d grid (# of h-planes)
  size_type proc_h_stride_ = 0;     ///< ranks per h-plane (0 = ungrouped 2-d)

  /// Two-trange retile plan derived from a user .retile() target. Inactive
  /// (active == false) unless the consumer set a target that differs from the
  /// operands' own (U) tilings; when inactive every code path below behaves
  /// exactly as without a retile request. Computed by maybe_make_retile_plan_()
  /// at the end of init_struct[_general]; consumed by later phases (passed to
  /// the Summa ctor, which currently only stores it).
  RetilePlan plan_;

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
    /// only Hadamard, contraction, and general are supported now
    TA_ASSERT(product_type_ == TensorProduct::Hadamard ||
              product_type_ == TensorProduct::Contraction ||
              product_type_ == TensorProduct::General);
    return product_type_;
  }

  /// \return true if the outer product is evaluated by a (batched) SUMMA,
  /// i.e. the tile op is a ContractReduce (a pure contraction or a general
  /// product); false for the elementwise (Hadamard) binary tile op
  bool outer_product_uses_summa() const {
    return product_type_ == TensorProduct::Contraction ||
           product_type_ == TensorProduct::General;
  }

  /// \return for a general product, the number of leading fused (outer)
  /// modes common to the canonical left, right, and result layouts
  /// (GeneralPermutationOptimizer places them first); 0 for the other
  /// product types
  unsigned int n_fused_outer_modes() const {
    if (product_type_ != TensorProduct::General) return 0u;
    auto const& l = outer(left_indices_);
    auto const& r = outer(right_indices_);
    auto const& res = outer(indices_);
    unsigned int nh = 0u;
    while (nh < res.size() && nh < l.size() && nh < r.size() &&
           l[nh] == res[nh] && r[nh] == res[nh])
      ++nh;
    return nh;
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

  /// Constructor

  /// \tparam L The left-hand argument expression type
  /// \tparam R The right-hand argument expression type
  /// \param expr The parent expression
  template <typename L, typename R>
  ContEngine(const DotInnerExpr<L, R>& expr)
      : BinaryEngine_(expr), factor_(1) {}

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
      // a view inner cell (e.g. ArenaTensor) cannot host a value-returning
      // inner op, so element_return_op_ is intentionally left null for it
      if constexpr (!TiledArray::is_tensor_view_v<result_tile_element_type>)
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

    // The ContractReduce tile op needs the per-cell inner element op whenever
    // the operands are nested -- both when the result is also nested
    // (is_tensor_of_tensor_v<value_type>) and in the dot_inner regime, where
    // the nested modes are fully contracted to a plain scalar result
    // (denest_to_scalar). In the latter case the result tile is a plain tensor,
    // so it carries no inner permutation and no arena plan, but it still routes
    // through gemm(result, left, right, helper, elem_muladd_op).
    constexpr bool tot_aware_op =
        TiledArray::detail::is_tensor_of_tensor_v<value_type> ||
        denest_to_scalar;

    // initialize op_, trange_, and shape_ which only refer to the outer modes
    if (outer(target_indices) != outer(indices_)) {
      const auto outer_perm = outer(perm_);
      // Initialize permuted structure
      if constexpr (!tot_aware_op) {
        op_ = op_type(
            left_op, right_op, factor_, outer_size(indices_),
            outer_size(left_indices_), outer_size(right_indices_),
            (!implicit_permute_outer_ ? std::move(outer_perm) : Permutation{}));
      } else {
        auto make_total_perm = [this]() -> BipartitePermutation {
          // dot_inner: the result tile is a plain tensor of scalars with no
          // inner modes, so only the outer permutation applies.
          if constexpr (denest_to_scalar) {
            return this->implicit_permute_outer_
                       ? BipartitePermutation()
                       : BipartitePermutation(outer(this->perm_));
          } else {
            if (this->product_type() != TensorProduct::Contraction ||
                this->implicit_permute_inner_)
              return this->implicit_permute_outer_
                         ? BipartitePermutation()
                         : BipartitePermutation(outer(this->perm_));

            // Here,
            // this->product_type() is Tensor::Contraction, and,
            // this->implicit_permute_inner_ is false

            if (this->inner_product_type() == TensorProduct::Scale) {
              // Owning inner cells apply the inner result permutation in the
              // per-cell scale op, so they carry only the outer perm here. View
              // (arena) cells instead use a perm-free per-cell op + an
              // unpermuted arena plan and rely on op_'s post-processing permute
              // for the inner perm -- so they carry the full perm, like inner
              // Contraction.
              if constexpr (!TiledArray::is_tensor_view_v<
                                result_tile_element_type>)
                return BipartitePermutation(outer(this->perm_));
            }
            return this->perm_;
          }
        };

        auto total_perm = make_total_perm();

        // factor_ is absorbed into inner_tile_nonreturn_op_
        op_ = op_type(left_op, right_op, scalar_type(1), outer_size(indices_),
                      outer_size(left_indices_), outer_size(right_indices_),
                      total_perm, this->element_nonreturn_op_,
                      std::move(this->arena_plan_));
        if constexpr (TiledArray::detail::is_tensor_of_tensor_v<value_type>) {
          // ce+e, ce+ce_right and ce+ce_left are mutually exclusive (ce+e gates
          // on num_contract_ranks()==0; the two ce+ce orientations on disjoint
          // right-/left-clean inner structure), so at most one is non-null and
          // only one install fires.
          if (this->arena_strided_dgemm_ce_e_tile_op_)
            op_.set_strided_oprod_op(this->arena_strided_dgemm_ce_e_tile_op_);
          if (this->arena_strided_dgemm_ce_ce_right_tile_op_)
            op_.set_strided_oprod_op(
                this->arena_strided_dgemm_ce_ce_right_tile_op_);
          if (this->arena_strided_dgemm_ce_ce_left_tile_op_)
            op_.set_strided_oprod_op(
                this->arena_strided_dgemm_ce_ce_left_tile_op_);
        }
        // Plan ownership transferred to op_; mark carrier slot empty so any
        // later use of arena_plan_ reads as "no plan" rather than moved-from.
        if constexpr (!std::is_same_v<arena_plan_storage_t, std::monostate>) {
          this->arena_plan_.reset();
        }
      }
      trange_ = ContEngine_::make_trange(outer_perm);
      shape_ = ContEngine_::make_shape(outer_perm);
    } else {
      // Initialize non-permuted structure

      if constexpr (!tot_aware_op) {
        op_ = op_type(left_op, right_op, factor_, outer_size(indices_),
                      outer_size(left_indices_), outer_size(right_indices_));
      } else {
        auto make_total_perm = [this]() -> BipartitePermutation {
          // dot_inner: the result tile is a plain tensor of scalars with no
          // inner modes; this is the non-permuted branch, so no perm applies.
          if constexpr (denest_to_scalar) {
            return {};
          } else if (this->product_type() != TensorProduct::Contraction ||
                     this->implicit_permute_inner_)
            return {};

          // Here,
          // this->product_type() is Tensor::Contraction, and,
          // this->implicit_permute_inner_ is false

          if (this->inner_product_type() == TensorProduct::Scale) {
            // Owning inner cells apply the inner result permutation in the
            // per-cell scale op, so they carry only the outer perm here. View
            // (arena) cells instead use a perm-free per-cell op + an unpermuted
            // arena plan and rely on op_'s post-processing permute for the
            // inner perm -- so they carry the full perm, like inner
            // Contraction.
            if constexpr (!TiledArray::is_tensor_view_v<
                              result_tile_element_type>)
              return BipartitePermutation(outer(this->perm_));
          }
          return this->perm_;
        };

        auto total_perm = make_total_perm();

        // factor_ is absorbed into inner_tile_nonreturn_op_
        op_ = op_type(left_op, right_op, scalar_type(1), outer_size(indices_),
                      outer_size(left_indices_), outer_size(right_indices_),
                      total_perm, this->element_nonreturn_op_,
                      std::move(this->arena_plan_));
        if constexpr (TiledArray::detail::is_tensor_of_tensor_v<value_type>) {
          // ce+e, ce+ce_right and ce+ce_left are mutually exclusive (ce+e gates
          // on num_contract_ranks()==0; the two ce+ce orientations on disjoint
          // right-/left-clean inner structure), so at most one is non-null and
          // only one install fires.
          if (this->arena_strided_dgemm_ce_e_tile_op_)
            op_.set_strided_oprod_op(this->arena_strided_dgemm_ce_e_tile_op_);
          if (this->arena_strided_dgemm_ce_ce_right_tile_op_)
            op_.set_strided_oprod_op(
                this->arena_strided_dgemm_ce_ce_right_tile_op_);
          if (this->arena_strided_dgemm_ce_ce_left_tile_op_)
            op_.set_strided_oprod_op(
                this->arena_strided_dgemm_ce_ce_left_tile_op_);
        }
        // Plan ownership transferred to op_; mark carrier slot empty so any
        // later use of arena_plan_ reads as "no plan" rather than moved-from.
        if constexpr (!std::is_same_v<arena_plan_storage_t, std::monostate>) {
          this->arena_plan_.reset();
        }
      }
      trange_ = ContEngine_::make_trange();
      shape_ = ContEngine_::make_shape();
    }

    if (ExprEngine_::override_ptr_ && ExprEngine_::override_ptr_->shape) {
      shape_ = shape_.mask(*ExprEngine_::override_ptr_->shape);
    }

    // Two-trange retile plan from a user .retile() target (no-op when absent;
    // inactive when the target coincides with the operands' own tilings).
    maybe_make_retile_plan_();
  }

  /// Build plan_ from a user .retile() target, if one was set. When no target
  /// is present plan_ stays default (inactive) and nothing else changes. When a
  /// target is present the plan is derived from the operands' own (U) tranges,
  /// the OUTER role-partition helper (op_.gemm_helper(), built above), the
  /// INNER contract-reduce helper (recomputed here from the inner index sizes,
  /// exactly as init_inner_tile_op builds contrreduce_op), and the leading
  /// fused-mode count. An identity target (or one coinciding with U) yields an
  /// inactive plan. The plan is consumed by later phases; in this phase it is
  /// only stored and threaded to the Summa ctor.
  /// Synthesize coarse retile targets for the mixed (plain x arena-ToT ->
  /// arena-ToT) shape with the env-gated auto-retile. Per
  /// detail::mixed_retile_config, coarsen THREE roles:
  ///   - the PLAIN operand's external (role M when the plain operand is LEFT,
  ///     role N when it is RIGHT) toward plain_external_target (default 0 =>
  ///     single tile / full collapse);
  ///   - the contracted (K) axis toward contracted_target (default 0 => collapse);
  ///   - the arena-ToT operand's external -- the LEFTOVER SUMMA axis (role N
  ///     when the plain operand is LEFT, role M when it is RIGHT) -- toward
  ///     tot_external_target (default 16).
  /// The fused (H) axes are left EMPTY (kept intact at U). Every target is fed
  /// through coarsen_tr1, which COARSENS ONLY (merges U tiles onto existing U
  /// boundaries; never refines), so a leftover-SUMMA axis already coarser than
  /// the target is kept intact rather than refined (refine is unsupported).
  /// The role axes are extracted by the SAME positional H/M/N/K partition that
  /// make_retile_plan consults (nf leading fused modes; the last nc left axes
  /// are K; the remaining left axes are M; the right axes past nf+nc are N), so
  /// the synthesized targets are in canonical H/M/N/K space -- identical to what
  /// a .retile() caller supplies. This is the default verified by
  /// mixed_T_x_ToT_coarsen_MK_retile_N (plain left -> collapse M+K, coarsen N to
  /// 16) and mixed_ToT_x_T_coarsen_NK_retile_M (plain right -> collapse N+K,
  /// coarsen M to 16).
  void synthesize_mixed_targets_(bool left_is_plain,
                                 const math::GemmHelper& outer_gh,
                                 std::vector<TiledRange1>& tH,
                                 std::vector<TiledRange1>& tM,
                                 std::vector<TiledRange1>& tN,
                                 std::vector<TiledRange1>& tK) {
    (void)tH;  // H stays at U (empty target)
    const TiledRange& left_U = left_.trange();
    const TiledRange& right_U = right_.trange();
    const unsigned int nf = this->n_fused_modes_;
    const unsigned int nc = outer_gh.num_contract_ranks();
    const unsigned int left_rank = static_cast<unsigned int>(left_U.rank());
    const unsigned int right_rank = static_cast<unsigned int>(right_U.rank());
    const std::size_t ext_target =
        detail::mixed_retile_config.plain_external_target;
    const std::size_t k_target = detail::mixed_retile_config.contracted_target;
    const std::size_t tot_ext_target =
        detail::mixed_retile_config.tot_external_target;
    if (left_is_plain) {
      // plain operand is LEFT => its external is role M (left outer axes),
      // collapsed; the arena-ToT external is role N (right outer axes),
      // coarsened to tot_ext_target.
      for (unsigned int i = nf; i + nc < left_rank; ++i)
        tM.push_back(coarsen_tr1(left_U.dim(i), ext_target));
      for (unsigned int i = nf + nc; i < right_rank; ++i)
        tN.push_back(coarsen_tr1(right_U.dim(i), tot_ext_target));
    } else {
      // plain operand is RIGHT => its external is role N (right outer axes),
      // collapsed; the arena-ToT external is role M (left outer axes),
      // coarsened to tot_ext_target.
      for (unsigned int i = nf + nc; i < right_rank; ++i)
        tN.push_back(coarsen_tr1(right_U.dim(i), ext_target));
      for (unsigned int i = nf; i + nc < left_rank; ++i)
        tM.push_back(coarsen_tr1(left_U.dim(i), tot_ext_target));
    }
    // K (contracted) axes are the last nc axes of the left operand.
    for (unsigned int i = left_rank - nc; i < left_rank; ++i)
      tK.push_back(coarsen_tr1(left_U.dim(i), k_target));
  }

  void maybe_make_retile_plan_() {
    // ORDERING (permute-then-retile): when an operand annotation is non-canonical
    // the binary engine physically permutes it into canonical NoTranspose layout
    // BEFORE this plan is built (canonicalization commit e439c747). The plan and
    // the coarse pack/gather therefore operate in canonical contraction-role
    // (H/M/N/K) space; retiling a still-permuted operand would force role<->axis
    // remapping and a permute of fat coarse tiles -- strictly more work. The
    // RESULT permutation is the mirror image: contract in canonical layout, then
    // re-permute the (already carved) result tiles in-rank (make_dist_eval_general
    // / GeneralRepermuteOp + make_repermuted_result_pmap).
    const bool explicit_target =
        ExprEngine_::override_ptr_ &&
        ExprEngine_::override_ptr_->contraction_target.present;
    if constexpr (TiledArray::detail::is_tensor_of_tensor_v<value_type>) {
      // Decide whether we will retile AT ALL before building any helper: an
      // explicit .retile() target, or the mixed auto-retile gate being on. Only
      // then is the INNER contract-reduce helper constructed -- its GemmHelper
      // ctor asserts GEMM rank parity, which a NON-contraction inner product
      // (inner Hadamard, fused broadcast, no-externals general product) does NOT
      // satisfy. Building it unconditionally for every ToT contraction would
      // wrongly throw for those; the stock (no-retile) path must stay inert.
      bool do_retile = explicit_target;
      if constexpr (is_mixed_)
        do_retile = do_retile || detail::mixed_auto_retile_enabled();
      if (!do_retile) return;  // stock SUMMA: plan_ stays inactive

      // OUTER (SUMMA-level) role-partition helper, as already built into op_.
      const math::GemmHelper& outer_gh = op_.gemm_helper();
      // INNER (per-tile contract-reduce) helper: same construction as
      // contrreduce_op in init_inner_tile_op -- index-structure-only, so its
      // contracted-rank count (the only thing make_retile_plan reads) is exact.
      const math::GemmHelper inner_gh(
          to_cblas_op(this->left_inner_permtype_),
          to_cblas_op(this->right_inner_permtype_),
          inner_size(this->indices_), inner_size(this->left_indices_),
          inner_size(this->right_indices_));
      if (explicit_target) {
        // Explicit .retile() target: unchanged behavior.
        const auto& ct = ExprEngine_::override_ptr_->contraction_target;
        plan_ = make_retile_plan(left_.trange(), right_.trange(), ct.targetH,
                                 ct.targetM, ct.targetN, ct.targetK, outer_gh,
                                 inner_gh, this->n_fused_modes_);
        return;
      }
      // No explicit target, gate on, mixed shape: synthesize coarse targets and
      // build the plan. The coarse gather (arena_gather_block) always lays the
      // operand down as one single-page tile, so the strided scale-GEMM is
      // eligible without any extra compaction step (see arena_outer_init's
      // arena_assert_single_page).
      if constexpr (is_mixed_) {
        std::vector<TiledRange1> tH, tM, tN, tK;
        synthesize_mixed_targets_(left_is_plain_, outer_gh, tH, tM, tN, tK);
        plan_ = make_retile_plan(left_.trange(), right_.trange(), tH, tM, tN, tK,
                                 outer_gh, inner_gh, this->n_fused_modes_);
      }
      return;
    } else {
      // a .retile() target only makes sense for nested (ToT) contractions;
      // for plain-tensor contractions there is no two-trange retile to derive.
      if (explicit_target)
        TA_EXCEPTION(
            "MultExpr::retile() is only supported for tensor-of-tensor "
            "contractions");
    }
  }

  /// \return true iff the (active) retile plan has ANY Refine axis (T finer than
  /// U) on any role (Hadamard, SUMMA-M/N/K). The COARSEN/identity active path is
  /// supported at np>1 (operands+result are distributed on the COARSE T-grid so
  /// each U tile is co-located with its covering coarse cell's owner); the
  /// REFINE path at np>1 is deferred and stays rejected. This predicate is
  /// SYMMETRIC across ranks (every rank evaluates the same plan_), so the
  /// np>1-refine guard below throws collectively.
  bool plan_has_refine_() const {
    auto role_refines = [](const std::vector<expressions::AxisNest>& v) {
      for (const auto& ax : v)
        if (ax.dir == expressions::NestDir::Refine) return true;
      return false;
    };
    return role_refines(plan_.hadamard) || role_refines(plan_.summaM) ||
           role_refines(plan_.summaN) || role_refines(plan_.summaK);
  }

  /// Reject an active REFINE plan at MPI world size > 1 (deferred). Shared by
  /// coarse_K_/M_/N_; symmetric across ranks => throws collectively. TA_EXCEPTION
  /// (not TA_ASSERT): .retile() is user-reachable and this must survive Release.
  void reject_refine_at_np_gt_1_() const {
    if (world_->size() > 1 && plan_has_refine_())
      TA_EXCEPTION(
          "in-SUMMA two-trange retile (.retile) with a REFINE axis is currently "
          "supported only at MPI world size 1; np>1 refine support is not yet "
          "implemented (coarsen/identity is supported at np>1)");
  }

  /// \return the COARSE SUMMA-K tile count (number of T contracted tiles)
  /// when the retile plan is active, else the fine (U) count \p k_fine. For an
  /// active plan the coarse count is the product over the SUMMA-K role axes of
  /// the number of T tiles on each (\c AxisNest::groups.size()), since each T
  /// tile spans a contiguous group of fine U tiles. For the K-collapse case
  /// (all U K-tiles gathered into one T tile) this is 1. Inactive plans return
  /// \p k_fine unchanged so the stock SUMMA step count is byte-for-byte intact.
  size_type coarse_K_(size_type k_fine) const {
    if (!plan_.active) return k_fine;
    reject_refine_at_np_gt_1_();
    size_type kc = 1ul;
    for (const auto& ax : plan_.summaK) kc *= ax.groups.size();
    return kc;
  }

  /// \return the COARSE SUMMA-M tile count (number of T result-row tiles) when
  /// the retile plan is active, else the fine (U) count \p m_fine. The coarse
  /// count is the product over the SUMMA-M role axes of the number of T tiles
  /// on each (\c AxisNest::groups.size()). Inactive plans return \p m_fine
  /// unchanged so the stock SUMMA grid is byte-for-byte intact. Like
  /// \c coarse_K_, an active plan is np=1 only (the active result reconciliation
  /// + ride gather are local on a single rank); np>1 active is rejected.
  size_type coarse_M_(size_type m_fine) const {
    if (!plan_.active) return m_fine;
    reject_refine_at_np_gt_1_();
    size_type mc = 1ul;
    for (const auto& ax : plan_.summaM) mc *= ax.groups.size();
    return mc;
  }

  /// \return the COARSE SUMMA-N tile count (number of T result-col tiles) when
  /// the retile plan is active, else the fine (U) count \p n_fine. Mirror of
  /// \c coarse_M_ over the SUMMA-N role axes.
  size_type coarse_N_(size_type n_fine) const {
    if (!plan_.active) return n_fine;
    reject_refine_at_np_gt_1_();
    size_type nc = 1ul;
    for (const auto& ax : plan_.summaN) nc *= ax.groups.size();
    return nc;
  }

  /// \return the COARSE Hadamard (fused) slab count (number of T fused tiles =
  /// SUMMA slabs n_slabs_/nh_) when the retile plan is active, else the fine (U)
  /// count \p h_fine. The coarse count is the product over the Hadamard role
  /// axes of the number of T tiles on each (\c AxisNest::groups.size()), since
  /// each coarse fused tile spans a contiguous group of fine U fused tiles. For
  /// IDENTITY-H this equals the U count (no regression); for COARSEN-H it is
  /// FEWER slabs
  /// because BatchedContractReduce batches over fused_volume(tile.range()) ==
  /// the product of the leading nfused-mode EXTENTS of the operand tile, the
  /// coarse-extent operand tile (gathered by get_col/get_row_coarsen along the
  /// leading fused axis, append_role_t_box(plan_.hadamard,...)) makes each
  /// batched-GEMM call process the WIDER batch automatically. Coarsening H is
  /// therefore both a distribution change (fewer slabs) AND an operand PACK
  /// along the fused outer axis. Inactive plans return \p h_fine unchanged so
  /// the stock SUMMA slab count is byte-for-byte intact. np=1 only for an
  /// active COARSEN-H plan (the coarse-H np>1 distribution is handled separately).
  size_type coarse_H_(size_type h_fine) const {
    if (!plan_.active) return h_fine;
    reject_refine_at_np_gt_1_();
    size_type hc = 1ul;
    for (const auto& ax : plan_.hadamard) hc *= ax.groups.size();
    return hc;
  }

  // == np>1 COARSEN/identity operand+result distribution on the COARSE T-grid ==
  //
  // At np>1 the SUMMA broadcast keys/roots on the COARSE (T) proc_grid_, while
  // operands are stored and the result delivered at the FINE (U) tiling. For the
  // in-step gather to be local and the coarse-keyed broadcast to be consistent,
  // every U tile must live on the rank that owns its covering coarse T-cell in
  // the matching coarse phase pmap. The helpers below compose
  //   owner(U_tile) = coarse_phase_pmap.owner( coarse_cell_ordinal(U_tile))
  // by decomposing the U tile ordinal into per-axis U indices, mapping each U
  // index to its covering T index via the plan's per-role AxisNest groups, and
  // recomposing the coarse-cell ordinal in the layout the coarse phase pmap
  // expects. COARSEN/identity only (refine is rejected at np>1 above); on an
  // identity axis the U->T map is the identity so a pure-identity plan
  // reproduces the stock owner exactly. At np=1 every owner is rank 0, so these
  // are inert there.

  /// Per-axis U->T lookup for axis \p a of \p role (length = # U tiles on a).
  static std::vector<std::size_t> role_u_to_t_axis(
      const std::vector<expressions::AxisNest>& role, std::size_t a) {
    const auto& ax = role[a];
    std::size_t n_u = 0;
    for (const auto& g : ax.groups) n_u = std::max(n_u, g.second);
    std::vector<std::size_t> u2t(n_u, 0);
    for (std::size_t t = 0; t < ax.groups.size(); ++t)
      for (std::size_t u = ax.groups[t].first; u < ax.groups[t].second; ++u)
        u2t[u] = t;
    return u2t;
  }

  /// Number of T tiles on axis \p a of \p role (== groups.size()).
  static std::size_t role_t_extent(
      const std::vector<expressions::AxisNest>& role, std::size_t a) {
    return role[a].groups.size();
  }

  /// Compose the coarse-cell-owner pmap for an operand whose U outer trange is
  /// laid out [(skipped) H axes..., lead-role axes..., tail-role axes...]
  /// (left: M then K; right: K then N). \p coarse_phase is the matching coarse
  /// phase pmap (left: make_row_phase_pmap(coarse_K); right:
  /// make_col_phase_pmap(coarse_N=cols)). The coarse-cell ordinal is
  /// lead_t * tail_coarse + tail_t, matching the CyclicPmap row-major
  /// (rows=lead_coarse, cols=tail_coarse) decode.
  ///
  /// \p n_skip is the number of LEADING Hadamard (fused) U-trange modes to skip
  /// (the general/Hadamard path; 0 on the ordinary 2-d path). When n_skip > 0
  /// the returned pmap is a PER-SLAB base map over only the EXTERNAL (non-H)
  /// tiles -- its size is the per-slab external U-tile count, exactly what
  /// SlabbedPmap expects as its base. The H tiling is identity here
  /// so every slab is distributed identically and the base map is
  /// slab-agnostic; SlabbedPmap then replicates it across n_slabs_.
  std::shared_ptr<const pmap_interface> make_operand_coarse_pmap(
      World& world, const trange_type& u_tr,
      const std::vector<expressions::AxisNest>& lead_role,
      const std::vector<expressions::AxisNest>& tail_role,
      const std::shared_ptr<Pmap>& coarse_phase, std::size_t tail_coarse,
      std::size_t n_skip = 0ul) const {
    const auto& u_tiles = u_tr.tiles_range();
    const std::size_t n_lead = lead_role.size();
    const std::size_t n_tail = tail_role.size();
    const std::size_t rank = n_lead + n_tail;
    // The U trange carries n_skip leading H modes followed by the external
    // (lead-role + tail-role) modes. The base pmap is built over the external
    // tiles only; H is replicated by SlabbedPmap at the call site.
    TA_ASSERT(u_tiles.rank() == n_skip + rank);

    // Precompute per-axis U->T lookups.
    std::vector<std::vector<std::size_t>> lead_u2t(n_lead), tail_u2t(n_tail);
    std::vector<std::size_t> lead_t_ext(n_lead), tail_t_ext(n_tail);
    for (std::size_t a = 0; a < n_lead; ++a) {
      lead_u2t[a] = role_u_to_t_axis(lead_role, a);
      lead_t_ext[a] = role_t_extent(lead_role, a);
    }
    for (std::size_t b = 0; b < n_tail; ++b) {
      tail_u2t[b] = role_u_to_t_axis(tail_role, b);
      tail_t_ext[b] = role_t_extent(tail_role, b);
    }

    // Per-slab external extents are the trailing `rank` axes of the U trange.
    const auto* extent = u_tiles.extent_data();
    std::size_t n_ext_tiles = 1ul;
    for (std::size_t a = 0; a < rank; ++a)
      n_ext_tiles *= static_cast<std::size_t>(extent[n_skip + a]);

    std::vector<ProcessID> owners(n_ext_tiles);
    std::vector<std::size_t> idx(rank);
    for (std::size_t ord = 0; ord < n_ext_tiles; ++ord) {
      // decompose ord row-major into per-external-axis U indices
      std::size_t rem = ord;
      for (std::size_t a = rank; a-- > 0;) {
        const std::size_t e = static_cast<std::size_t>(extent[n_skip + a]);
        idx[a] = e ? (rem % e) : 0;
        rem = e ? (rem / e) : rem;
      }
      // lead_t (row-major over lead T extents), tail_t (row-major over tail)
      std::size_t lead_t = 0;
      for (std::size_t a = 0; a < n_lead; ++a)
        lead_t = lead_t * lead_t_ext[a] + lead_u2t[a][idx[a]];
      std::size_t tail_t = 0;
      for (std::size_t b = 0; b < n_tail; ++b)
        tail_t = tail_t * tail_t_ext[b] + tail_u2t[b][idx[n_lead + b]];
      const std::size_t cell = lead_t * tail_coarse + tail_t;
      owners[ord] = static_cast<ProcessID>(coarse_phase->owner(cell));
    }

    std::size_t local = 0;
    const std::size_t me = world.rank();
    for (auto o : owners)
      if (static_cast<std::size_t>(o) == me) ++local;
    auto owners_ptr = std::make_shared<std::vector<ProcessID>>(std::move(owners));
    return std::make_shared<TiledArray::detail::UserPmap>(
        world, n_ext_tiles, local,
        [owners_ptr](std::size_t t) -> std::size_t {
          return static_cast<std::size_t>((*owners_ptr)[t]);
        });
  }

  /// Compose the coarse-cell-owner pmap for the RESULT whose U trange is laid
  /// out [(skipped) H-axes..., M-axes..., N-axes...]. The coarse result grid
  /// (proc_grid_.make_pmap()) is CyclicPmap(rows=M_grid, cols=N_grid), so the
  /// coarse-cell ordinal is mc * N_grid + nc.
  ///
  /// \p n_skip is the number of LEADING Hadamard (fused) result-trange modes to
  /// skip (the general/Hadamard path; 0 on the ordinary 2-d path). When
  /// n_skip > 0 the returned pmap is a PER-SLAB base map over only the EXTERNAL
  /// (non-H) result tiles -- its size is the per-slab external result-tile
  /// count, exactly what SlabbedPmap expects as its base (H is identity here
  /// so the base is slab-agnostic and SlabbedPmap replicates it).
  std::shared_ptr<const pmap_interface> make_result_coarse_pmap(
      World& world, const trange_type& u_tr,
      const std::shared_ptr<Pmap>& coarse_result, std::size_t N_grid,
      std::size_t n_skip = 0ul) const {
    const auto& u_tiles = u_tr.tiles_range();
    const std::size_t n_m = plan_.summaM.size();
    const std::size_t n_n = plan_.summaN.size();
    const std::size_t rank = n_m + n_n;
    TA_ASSERT(u_tiles.rank() == n_skip + rank);

    std::vector<std::vector<std::size_t>> m_u2t(n_m), n_u2t(n_n);
    std::vector<std::size_t> m_t_ext(n_m), n_t_ext(n_n);
    for (std::size_t a = 0; a < n_m; ++a) {
      m_u2t[a] = role_u_to_t_axis(plan_.summaM, a);
      m_t_ext[a] = role_t_extent(plan_.summaM, a);
    }
    for (std::size_t b = 0; b < n_n; ++b) {
      n_u2t[b] = role_u_to_t_axis(plan_.summaN, b);
      n_t_ext[b] = role_t_extent(plan_.summaN, b);
    }

    // Per-slab external extents are the trailing `rank` axes of the U trange.
    const auto* extent = u_tiles.extent_data();
    std::size_t n_ext_tiles = 1ul;
    for (std::size_t a = 0; a < rank; ++a)
      n_ext_tiles *= static_cast<std::size_t>(extent[n_skip + a]);

    std::vector<ProcessID> owners(n_ext_tiles);
    std::vector<std::size_t> idx(rank);
    for (std::size_t ord = 0; ord < n_ext_tiles; ++ord) {
      std::size_t rem = ord;
      for (std::size_t a = rank; a-- > 0;) {
        const std::size_t e = static_cast<std::size_t>(extent[n_skip + a]);
        idx[a] = e ? (rem % e) : 0;
        rem = e ? (rem / e) : rem;
      }
      std::size_t mc = 0;
      for (std::size_t a = 0; a < n_m; ++a)
        mc = mc * m_t_ext[a] + m_u2t[a][idx[a]];
      std::size_t nc = 0;
      for (std::size_t b = 0; b < n_n; ++b)
        nc = nc * n_t_ext[b] + n_u2t[b][idx[n_m + b]];
      const std::size_t cell = mc * N_grid + nc;
      owners[ord] = static_cast<ProcessID>(coarse_result->owner(cell));
    }

    std::size_t local = 0;
    const std::size_t me = world.rank();
    for (auto o : owners)
      if (static_cast<std::size_t>(o) == me) ++local;
    auto owners_ptr = std::make_shared<std::vector<ProcessID>>(std::move(owners));
    return std::make_shared<TiledArray::detail::UserPmap>(
        world, n_ext_tiles, local,
        [owners_ptr](std::size_t t) -> std::size_t {
          return static_cast<std::size_t>((*owners_ptr)[t]);
        });
  }

  /// Build the (target-layout) result pmap for a general_repermute_ product
  /// that CO-LOCATES each target tile with its canonical source. The general
  /// product is evaluated in CANONICAL layout by the inner Summa, then a
  /// streaming UnaryEvalImpl<GeneralRepermuteOp> permutes each tile and
  /// set_tile()s it to the target ordinal; making the target tile's owner equal
  /// to its canonical source's owner keeps that push in-rank. It also fixes a
  /// correctness bug: make_result_coarse_pmap assumes CANONICAL [H][M][N] axis
  /// order, so it cannot be applied to the PERMUTED target trange directly (the
  /// role axis-maps misalign and index out of bounds). Here \p canonical_pmap
  /// is the (correctly-ordered) coarse result pmap built on \p canonical_tr; for
  /// each target ordinal t we run its coordinate back through \p outer_perm.inv()
  /// to the canonical coordinate and adopt that canonical tile's owner. Since
  /// trange_ == outer_perm * canonical_tr, applying .inv() yields the true
  /// canonical source coordinate, whose per-axis extents match exactly -- so the
  /// canonical ordinal is always in range (the .inv() direction is load-bearing
  /// for that bound; passing the forward perm could index out of range for a
  /// non-involutive permutation). The map is placement-only: set_tile(t) and
  /// find(t) route through the SAME pmap, so the .inv() direction only affects
  /// communication locality, never values.
  std::shared_ptr<const pmap_interface> make_repermuted_result_pmap(
      World& world, const trange_type& target_tr,
      const trange_type& canonical_tr,
      const std::shared_ptr<const pmap_interface>& canonical_pmap,
      const Permutation& outer_perm) const {
    const auto& tgt_tiles = target_tr.tiles_range();
    const auto& can_tiles = canonical_tr.tiles_range();
    const std::size_t vol = tgt_tiles.volume();
    const Permutation inv = outer_perm.inv();
    std::vector<ProcessID> owners(vol);
    for (std::size_t t = 0; t < vol; ++t) {
      const auto tgt_idx = tgt_tiles.idx(t);  // target coordinate
      // NB: Range::idx returns an svector; Permutation::operator* needs a
      // std::vector overload -> copy explicitly.
      std::vector<std::size_t> ci(tgt_idx.begin(), tgt_idx.end());
      std::vector<std::size_t> can_idx = inv * ci;  // back to canonical coord
      const std::size_t s = can_tiles.ordinal(can_idx);
      owners[t] = static_cast<ProcessID>(canonical_pmap->owner(s));
    }
    std::size_t local = 0;
    const std::size_t me = world.rank();
    for (auto o : owners)
      if (static_cast<std::size_t>(o) == me) ++local;
    auto owners_ptr =
        std::make_shared<std::vector<ProcessID>>(std::move(owners));
    return std::make_shared<TiledArray::detail::UserPmap>(
        world, vol, local, [owners_ptr](std::size_t t) -> std::size_t {
          return static_cast<std::size_t>((*owners_ptr)[t]);
        });
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

    // corner case: zero-volume result ... easier to skip proc_grid_
    // construction alltogether
    if (M == 0 || N == 0) {
      left_.init_distribution(world, {});
      right_.init_distribution(world, {});
      ExprEngine_::init_distribution(
          world, (pmap ? pmap : policy::default_pmap(*world, M * N)));
    } else {  // M!=0 && N!=0

      // Construct the process grid. When the retile plan coarsens a result
      // (SUMMA-M/N) axis, the contraction grid's TILE dimensions become the
      // COARSE (T) M/N tile counts while the ELEMENT dimensions (m,n) stay the
      // total U element counts (retiling does not change the element space).
      // This makes proc_grid_.local_rows()/local_cols() iterate COARSE result
      // cells; one coarse cell then covers several U result tiles, reconciled
      // in Summa::initialize/finalize. Inactive plans keep the stock U grid
      // byte-for-byte (coarse_M_/coarse_N_ are the identity then).
      const size_type M_grid = coarse_M_(M);
      const size_type N_grid = coarse_N_(N);
      proc_grid_ = TiledArray::detail::ProcGrid(*world, M_grid, N_grid, m, n);

      if (plan_.active) {
        // np>1 COARSEN/identity: operands are stored at U and the result is
        // delivered at U, but the SUMMA broadcast keys/roots on the COARSE
        // proc_grid_. Co-locate each U tile on the rank that owns its covering
        // coarse T-cell (matching coarse phase pmap), so the in-step gather is
        // local and the coarse-keyed broadcast is consistent. At np=1 every
        // owner is rank 0, identical to the old fine-u_grid behavior. (Refine
        // at np>1 was rejected above; only coarsen/identity reaches here.)
        const size_type K_coarse = coarse_K_(K_);
#ifdef TA_STRIDED_DGEMM_COUNT
        // Witness: expose the coarse SUMMA grid (the retiled operand trange tile
        // counts on each role). A coarsened SUMMA external shows up here even
        // though the per-row strided BLAS GEMM count is invariant.
        TiledArray::detail::g_summa_coarse_m_grid.store(
            static_cast<std::size_t>(M_grid), std::memory_order_relaxed);
        TiledArray::detail::g_summa_coarse_n_grid.store(
            static_cast<std::size_t>(N_grid), std::memory_order_relaxed);
        TiledArray::detail::g_summa_coarse_k_grid.store(
            static_cast<std::size_t>(K_coarse), std::memory_order_relaxed);
#endif
        // left U layout = [M-axes..., K-axes...]; phase rows = M_grid, cols =
        // K_coarse.
        auto left_phase = proc_grid_.make_row_phase_pmap(K_coarse);
        left_.init_distribution(
            world, make_operand_coarse_pmap(*world, left_.trange(),
                                            plan_.summaM, plan_.summaK,
                                            left_phase, K_coarse));
        // right U layout = [K-axes..., N-axes...]; phase rows = K_coarse,
        // cols = N_grid.
        auto right_phase = proc_grid_.make_col_phase_pmap(K_coarse);
        right_.init_distribution(
            world, make_operand_coarse_pmap(*world, right_.trange(),
                                            plan_.summaK, plan_.summaN,
                                            right_phase, N_grid));

        if (!pmap) {
          const auto outer_perm = outer(perm_);
          if (!outer_perm.is_identity()) {
            // trange_ is the PERMUTED target layout; make_result_coarse_pmap
            // assumes canonical [M][N] role order, and -- more importantly --
            // the active retile delivers each U result tile by a LOCAL set_tile
            // from the rank that owns its covering coarse T-cell (arena ToT
            // tiles are not serialized, so a cross-rank set would hang). Build
            // the coarse result pmap on the CANONICAL trange and route each
            // target tile to its canonical source's owner, so a permuted result
            // whose U tiles scatter across the role boundary still lands every
            // carve sub-tile in-rank. Mirrors the general_repermute_ path.
            const auto canonical_tr = ContEngine_::make_trange();
            auto canonical_pmap = make_result_coarse_pmap(
                *world, canonical_tr, proc_grid_.make_pmap(), N_grid);
            pmap = make_repermuted_result_pmap(*world, trange_, canonical_tr,
                                               canonical_pmap, outer_perm);
          } else {
            pmap = make_result_coarse_pmap(*world, trange_,
                                           proc_grid_.make_pmap(), N_grid);
          }
        }
        ExprEngine_::init_distribution(world, pmap);
      } else {
        // Inactive: stock U grid (M_grid==M, N_grid==N so u_grid IS proc_grid_).
        const TiledArray::detail::ProcGrid u_grid = proc_grid_;
        left_.init_distribution(world, u_grid.make_row_phase_pmap(K_));
        right_.init_distribution(world, u_grid.make_col_phase_pmap(K_));
        if (!pmap) pmap = u_grid.make_pmap();
        ExprEngine_::init_distribution(world, pmap);
      }
    }
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

    // When the retile plan is active the SUMMA steps over the COARSE (T) K
    // tile count; the operands remain stored/distributed at the FINE (U) K
    // count (the pmaps above are keyed to K_), so the fine count is threaded
    // separately to drive the in-step U-block gather. Inactive => both == K_.
    std::shared_ptr<impl_type> pimpl = std::make_shared<impl_type>(
        left, right, *world_, trange_, shape_, pmap_, perm_, op_, coarse_K_(K_),
        proc_grid_, /*nh=*/1ul, /*proc_h=*/1ul, /*proc_h_stride=*/0ul, plan_,
        /*k_fine=*/K_);

    return dist_eval_type(pimpl);
  }

  // == General (fused + contracted + free indices) product support ==========
  //
  // The canonical layouts produced by GeneralPermutationOptimizer carry the
  // fused (Hadamard) modes as the leading modes of both arguments and the
  // result: left = (h, e_A, c), right = (h, c, e_B), result = (h, e_A, e_B).
  // The product is evaluated by the batched Summa: every fused-index tile
  // slab is an independent SUMMA distributed over ONE shared 2-d process
  // grid (the owner of a tile is independent of its slab index), and the
  // within-tile fused extents are folded into the tile batch dimension by
  // BatchedContractReduce.

  /// Initialize the result structure of a general product

  /// The general-product analogue of init_struct: builds the *folded*
  /// (fused-mode-free) tile op and the fused-mode-prefixed result trange and
  /// shape.
  /// \param target_indices The target index list for the result tensor
  void init_struct_general(const BipartiteIndexList& target_indices) {
    // precondition checks (mirror init_struct)
    if constexpr (TiledArray::detail::is_tensor_of_tensor_v<value_type>) {
      TA_ASSERT(element_nonreturn_op_);
      // a view inner cell (e.g. ArenaTensor) cannot host a value-returning
      // inner op, so element_return_op_ is intentionally left null for it
      if constexpr (!TiledArray::is_tensor_view_v<result_tile_element_type>)
        TA_ASSERT(element_return_op_);
    }

    // Initialize children
    left_.init_struct(left_indices_);
    right_.init_struct(right_indices_);

    // count the fused modes: the leading indices common to the canonical
    // left, right, and result layouts
    const unsigned int nh = n_fused_outer_modes();
    TA_ASSERT(nh > 0u);  // else this is a pure contraction
    n_fused_modes_ = nh;

    // initialize perm_; a target that differs from the canonical (fused...,
    // left-free..., right-free...) result layout cannot be folded into the
    // batched tile op (BatchedContractReduce must be perm-free), so the
    // product is evaluated in its canonical layout and re-permuted to the
    // target by a streaming unary eval (see make_dist_eval_general)
    this->init_perm(target_indices);
    general_repermute_ = (outer(target_indices) != outer(indices_));

    // Some degenerate folded shapes would carry a rank-0 tensor, which the
    // tile kernels do not support (see synthetic_unit_left_external()):
    //  - a NO-EXTERNAL product (every outer index fused or contracted, e.g.
    //    C("i,j;a,b") = A("x,i,j;a") * B("x,i,j;b")) folds to a rank-0 RESULT;
    //  - a FUSED BROADCAST (the left operand is entirely fused, no contraction,
    //    e.g. C("b,k") = A("b") * B("b,k")) folds to a rank-0 LEFT operand.
    // Evaluate both with a SYNTHETIC UNIT left-external mode: the folded
    // product becomes (1,K) x (K,N) -> (1,N), a supported shape (the no-
    // external case has N == 1, the broadcast has K == 1). The unit mode lives
    // only in the tile op's GemmHelper; tranges, shapes and tiles carry the
    // true ranks, and BatchedContractReduce / SparseShape::gemm_batched detect
    // the synthetic mode from the one-rank mismatch and pad their folded views
    // with a unit extent.
    const unsigned int u = synthetic_unit_left_external();
    //  - a FUSED BROADCAST ON THE RIGHT (the right operand is entirely fused,
    //    no contraction, e.g. C("b,k") = A("b,k") * B("b")) folds to a rank-0
    //    RIGHT operand; a synthetic unit RIGHT-external mode restores a
    //    supported (M,K) x (K,1) -> (M,1) shape (see
    //    synthetic_unit_right_external()).
    const unsigned int u_right = synthetic_unit_right_external();

    // the tile op operates on the folded (fused-mode-free) shapes; the
    // synthetic unit mode leads the folded left operand (trails the folded
    // right operand), so it is NoTrans on that side
    const auto left_op =
        u ? math::blas::NoTranspose : to_cblas_op(left_outer_permtype_);
    const auto right_op =
        u_right ? math::blas::NoTranspose : to_cblas_op(right_outer_permtype_);
    // As in init_struct, the ContractReduce tile op needs the per-cell inner
    // element op when the operands are nested -- including the dot_inner regime
    // (denest_to_scalar), where the result tile is plain but the nested modes
    // are still dotted per outer cell.
    constexpr bool tot_aware_op =
        TiledArray::detail::is_tensor_of_tensor_v<value_type> ||
        denest_to_scalar;
    if constexpr (!tot_aware_op) {
      op_ = op_type(left_op, right_op, factor_,
                    outer_size(indices_) - nh + u + u_right,
                    outer_size(left_indices_) - nh + u,
                    outer_size(right_indices_) - nh + u_right);
    } else {
      // the batched tile op must be perm-free (BatchedContractReduce cannot
      // host the folded-rank result permutation); the outer perm is handled
      // by the streaming re-permute (general_repermute_), so only a genuine
      // (non-identity) explicit inner result permutation requires one. N.B.
      // perm_ may carry a non-null identity inner component when only the
      // outer modes are permuted (the bipartite perm is constructed whole).
      // The dot_inner result has no inner modes, so this check is a no-op
      // there.
      if constexpr (!denest_to_scalar) {
        if (!implicit_permute_inner_ && bool(inner(perm_)) &&
            !inner(perm_).is_identity())
          TA_EXCEPTION(
              "general products of tensors-of-tensors: a non-identity inner "
              "result permutation is not yet supported; reorder the inner "
              "annotation of the result");
      }

      // factor_ is absorbed into element_nonreturn_op_
      op_ = op_type(left_op, right_op, scalar_type(1),
                    outer_size(indices_) - nh + u + u_right,
                    outer_size(left_indices_) - nh + u,
                    outer_size(right_indices_) - nh + u_right,
                    BipartitePermutation{}, this->element_nonreturn_op_,
                    std::move(this->arena_plan_));
      // ce+e, ce+ce_right and ce+ce_left are mutually exclusive; at most one
      // is non-null and only one install fires (see init_struct)
      if constexpr (TiledArray::detail::is_tensor_of_tensor_v<value_type>) {
        if (this->arena_strided_dgemm_ce_e_tile_op_)
          op_.set_strided_oprod_op(this->arena_strided_dgemm_ce_e_tile_op_);
        if (this->arena_strided_dgemm_ce_ce_right_tile_op_)
          op_.set_strided_oprod_op(
              this->arena_strided_dgemm_ce_ce_right_tile_op_);
        if (this->arena_strided_dgemm_ce_ce_left_tile_op_)
          op_.set_strided_oprod_op(
              this->arena_strided_dgemm_ce_ce_left_tile_op_);
      }
      // Plan ownership transferred to op_; mark carrier slot empty so any
      // later use of arena_plan_ reads as "no plan" rather than moved-from.
      if constexpr (!std::is_same_v<arena_plan_storage_t, std::monostate>) {
        this->arena_plan_.reset();
      }
    }

    trange_ = make_trange_general();
    shape_ = make_shape_general();
    if (general_repermute_) {
      // consumers see the target layout; the canonical structures are
      // recomputed in make_dist_eval_general for the inner Summa
      trange_ = outer(perm_) * trange_;
      shape_ = shape_.perm(outer(perm_));
    }

    if (ExprEngine_::override_ptr_ && ExprEngine_::override_ptr_->shape) {
      shape_ = shape_.mask(*ExprEngine_::override_ptr_->shape);
    }

    // Two-trange retile plan from a user .retile() target (no-op when absent;
    // inactive when the target coincides with the operands' own tilings).
    maybe_make_retile_plan_();
  }

  /// \return 1 if the folded general product needs a SYNTHETIC unit
  /// left-external mode, else 0. The folded (fused-mode-free) GEMM cannot host
  /// a rank-0 tensor, which arises in two degenerate cases:
  ///  - rank-0 RESULT: every outer index is fused or contracted (no external),
  ///    i.e. outer_size(indices_) == n_fused_modes_;
  ///  - rank-0 LEFT operand: the left argument is entirely fused with no
  ///    contraction (a fused broadcast / per-fused-block scale), i.e.
  ///    outer_size(left_indices_) == n_fused_modes_.
  /// A unit left-external mode (carried only in the GemmHelper) restores a
  /// supported (1,K) x (K,N) -> (1,N) shape in both cases.
  unsigned int synthetic_unit_left_external() const {
    return (outer_size(indices_) == n_fused_modes_ ||
            outer_size(left_indices_) == n_fused_modes_)
               ? 1u
               : 0u;
  }

  /// \return 1 if the folded general product needs a SYNTHETIC unit
  /// right-external mode, else 0. Mirror of synthetic_unit_left_external() for
  /// the RIGHT operand: a rank-0 RIGHT operand arises when the right argument
  /// is entirely fused with no contraction (a fused broadcast on the right,
  /// e.g. C("b,k") = A("b,k") * B("b")), i.e.
  /// outer_size(right_indices_) == n_fused_modes_. A left-external unit cannot
  /// fix it (the right operand stays rank-0), so a unit right-external mode
  /// (carried only in the GemmHelper) restores a supported (M,K) x (K,1) ->
  /// (M,1) shape. The result gains a trailing unit mode that is absent from the
  /// actual tranges/shapes/tiles, exactly as the left case prepends one.
  unsigned int synthetic_unit_right_external() const {
    return (outer_size(right_indices_) == n_fused_modes_) ? 1u : 0u;
  }

  /// Tiled range factory function for a general product

  /// \return The result tiled range: the fused mode ranges followed by the
  /// left- and right-external mode ranges
  trange_type make_trange_general() const {
    const unsigned int nh = n_fused_modes_;
    const unsigned int nc = op_.gemm_helper().num_contract_ranks();
    // degenerate folds carry a synthetic unit left-external mode in the
    // GemmHelper only (see synthetic_unit_left_external()); the actual tranges
    // do not have it
    const unsigned int u = synthetic_unit_left_external();
    const unsigned int u_right = synthetic_unit_right_external();
    const unsigned int neA = op_.gemm_helper().left_rank() - nc - u;
    const unsigned int neB = op_.gemm_helper().right_rank() - nc - u_right;

    typename trange_type::Ranges ranges(nh + neA + neB);
    unsigned int i = 0ul;
    for (unsigned int x = 0ul; x < nh + neA; ++x, ++i)
      ranges[i] = left_.trange().data()[x];
    for (unsigned int x = nh + nc; x < nh + nc + neB; ++x, ++i)
      ranges[i] = right_.trange().data()[x];

#ifndef NDEBUG
    // the fused and contracted dimensions must have congruent tilings
    for (unsigned int d = 0ul; d < nh; ++d) {
      if (!is_congruent(left_.trange().data()[d], right_.trange().data()[d]))
        TA_EXCEPTION(
            "the fused dimensions of the left- and right-hand expressions "
            "are not congruent");
    }
    for (unsigned int l = nh + neA, r = nh; l < nh + neA + nc; ++l, ++r) {
      if (!is_congruent(left_.trange().data()[l], right_.trange().data()[r]))
        TA_EXCEPTION(
            "the contracted dimensions of the left- and right-hand "
            "expressions are not congruent");
    }
#endif  // NDEBUG

    return trange_type(ranges.begin(), ranges.end());
  }

  /// Shape factory function for a general product

  /// \return The result shape: the fused modes lead; each fused-index slab
  /// is the shape-level contraction of the corresponding argument slabs
  shape_type make_shape_general() const {
    if constexpr (std::is_same_v<shape_type, DenseShape>)
      return shape_type();
    else
      return left_.shape().gemm_batched(right_.shape(), factor_,
                                        op_.gemm_helper(), n_fused_modes_);
  }

  /// Initialize the result distribution of a general product

  /// The 2-d process grid spans the external (free) modes only; the fused
  /// modes are replicated over the grid (the owner of a tile is independent
  /// of its slab index) via SlabbedPmap.
  /// \param world The world where the result will be distributed
  /// \param pmap The process map for the result tensor tiles
  void init_distribution_general(World* world,
                                 std::shared_ptr<const pmap_interface> pmap) {
    const unsigned int nh = n_fused_modes_;
    const unsigned int nc = op_.gemm_helper().num_contract_ranks();
    // degenerate folds carry a synthetic unit left-external mode in the
    // GemmHelper only (see synthetic_unit_left_external()); the actual tranges
    // do not have it
    const unsigned int u = synthetic_unit_left_external();
    const unsigned int u_right = synthetic_unit_right_external();
    const unsigned int neA = op_.gemm_helper().left_rank() - nc - u;
    const unsigned int neB = op_.gemm_helper().right_rank() - nc - u_right;

    // Get pointers to the argument sizes
    const auto* MADNESS_RESTRICT const left_tiles_size =
        left_.trange().tiles_range().extent_data();
    const auto* MADNESS_RESTRICT const left_element_size =
        left_.trange().elements_range().extent_data();
    const auto* MADNESS_RESTRICT const right_tiles_size =
        right_.trange().tiles_range().extent_data();
    const auto* MADNESS_RESTRICT const right_element_size =
        right_.trange().elements_range().extent_data();

    // Compute the slab count and the fused sizes of the per-slab contraction.
    // n_slabs_ is the SUMMA slab count (== nh_ in the Summa). When the retile
    // plan is ACTIVE it is the COARSE Hadamard tile count (coarse_H_): a coarse
    // fused tile spans a contiguous group of fine U fused tiles, so coarsening H
    // yields FEWER, fatter slabs;
    // the per-slab batch is then WIDENED
    // by the operand H-pack (get_col/get_row_coarsen gather the covered U fused
    // tiles along the leading fused axis into one coarse-extent tile, which
    // BatchedContractReduce batches over via fused_volume == the leading
    // nfused-mode EXTENTS), so the coarse slab count and the fat operand tile
    // are two halves of the SAME coarsen. For IDENTITY-H coarse_H_ == the U
    // count (no change); INACTIVE plans keep the U-derived count byte-for-byte.
    size_type M = 1ul, m = 1ul, N = 1ul, n = 1ul;
    // n_slabs_u = the FINE (U) fused-tile count == how many H slabs the operand
    // and result arrays are physically tiled into; n_slabs_ (== nh_ in Summa) =
    // the COARSE SUMMA slab count. They diverge only under an active COARSEN-H
    // plan. The OPERAND/RESULT distribution (SlabbedPmap replication) must use
    // the U count because the arrays carry U-H tiles; the SUMMA grid / step /
    // reduce-task geometry uses the coarse count. Inactive / identity-H =>
    // n_slabs_ == n_slabs_u, byte-for-byte stock.
    size_type n_slabs_u = 1ul;
    K_ = 1ul;
    for (unsigned int i = 0u; i < nh; ++i) n_slabs_u *= left_tiles_size[i];
    n_slabs_ = coarse_H_(n_slabs_u);
    // Carry the FINE (U) fused-tile count to make_dist_eval_general (the
    // general_repermute_ canonical_pmap) and keep it in scope for the grouped
    // active branch below: operand/result distribution replicates over the U
    // count because the arrays carry U-H tiles, while the SUMMA grid/step
    // geometry uses the coarse n_slabs_. Identity-H => the two are equal.
    n_slabs_u_ = n_slabs_u;
    // COARSEN-H at np>1: the COARSE fused count is strictly
    // fewer than the U fused count, so the SUMMA slab axis no longer maps 1:1
    // onto the U storage slabs. This is now supported: the proc_h_ slab
    // grouping rides the COARSE slabs while the operand/result SlabbedPmaps
    // replicate over the U count (n_slabs_u_), and the bcast key-space stays
    // collision-free (tile-bcast keys live in DistCache keyed by the FINE
    // left_.size()/right_.size(); the per-step + static GROUP keys live in the
    // separate group registry keyed by the coarse nsteps_, so the two never
    // alias even though nsteps_ shrinks while left_.size() stays fine -- see
    // dist_eval/contraction_eval.h make_col_group/make_row_group and the
    // static_key_base derivation). Identity-H (n_slabs_ == n_slabs_u) is the
    // prior byte-for-byte path. REFINE-H stays rejected via plan_has_refine_.
    //
    // SCOPE GUARD: COARSEN-H at np>1 is validated only for a
    // SINGLE DENSE Hadamard axis with UNIFORM group widths. Multi-H (the
    // coarse_slab_u_fused_ordinals / step_h row-major flatten over >1 fused
    // axis), per-slab SPARSE-H (the coarse presence probes pin H to slab 0,
    // exact for dense only), and NON-UNIFORM coarse-H (a coarse slab covering
    // a varying number of U slabs -- the grouped SlabbedPmap and the Summa's
    // left.size()/nh_ slab arithmetic both assume a constant u_per_coarse_slab
    // = n_slabs_u / n_slabs_) remain deferred -- reject them loudly at np>1
    // (collective, symmetric across ranks) rather than risk a mis-sized carve
    // / bcast membership skew. np=1 is unaffected (the np=1 coarse-H path
    // covers these by the per-slab carve it has always used). Identity-H
    // (n_slabs_ == n_slabs_u) never trips this.
    if (plan_.active && world->size() > 1 && n_slabs_ < n_slabs_u) {
      bool uniform_single_dense_h =
          plan_.hadamard.size() == 1ul && shape_type::is_dense() &&
          (n_slabs_u % n_slabs_ == 0ul);
      if (uniform_single_dense_h) {
        // each coarse group must cover exactly n_slabs_u / n_slabs_ U tiles
        const std::size_t w = static_cast<std::size_t>(n_slabs_u / n_slabs_);
        for (const auto& g : plan_.hadamard[0].groups)
          if (g.second - g.first != w) {
            uniform_single_dense_h = false;
            break;
          }
      }
      if (!uniform_single_dense_h)
        TA_EXCEPTION(
            "in-SUMMA two-trange retile (.retile) with a COARSEN Hadamard (H) "
            "axis at MPI world size > 1 is currently supported only for a "
            "single DENSE H axis with uniform group widths; multi-H / "
            "sparse-H / non-uniform coarsen-H at np>1 is not yet implemented");
    }
    for (unsigned int i = nh; i < nh + neA; ++i) {
      M *= left_tiles_size[i];
      m *= left_element_size[i];
    }
    for (unsigned int i = nh + neA; i < nh + neA + nc; ++i)
      K_ *= left_tiles_size[i];
    for (unsigned int i = nh + nc; i < nh + nc + neB; ++i) {
      N *= right_tiles_size[i];
      n *= right_element_size[i];
    }

    // corner case: zero-volume result ... easier to skip proc_grid_
    // construction alltogether. The result array is tiled at the U fused count,
    // so its default pmap spans n_slabs_u * M * N (an active COARSEN-H plan does
    // not shrink the result tile count -- the result is delivered at U).
    if (M == 0 || N == 0 || n_slabs_u == 0) {
      left_.init_distribution(world, {});
      right_.init_distribution(world, {});
      ExprEngine_::init_distribution(
          world,
          (pmap ? pmap : policy::default_pmap(*world, n_slabs_u * M * N)));
    } else {
      // Choose the process-grid extent proc_h_ along the slab (h) axis of
      // the 3-d grid: ranks beyond one-per-result-tile are useless to a
      // single slab's 2-d SUMMA, so the surplus is spread over the slab
      // (communication-free) dimension instead -- slab h goes to plane
      // h % proc_h_ of proc_h_stride_ = P / proc_h_ contiguous ranks (the
      // division-remainder ranks idle for this evaluation). For a
      // no-external product (M == N == 1) this degenerates to an effectively
      // 1-d grid over the slabs; for an ordinary contraction (n_slabs_ == 1)
      // it is a pure 2-d grid (proc_h_ == 1).
      // TODO: co-optimize proc_h_ with the 2-d (proc_r, proc_c) aspect ratio
      //       using the h-, left-external-, and right-external-mode element
      //       extents (and a per-rank memory bound), rather than the current
      //       greedy tile-count heuristic.
      //
      // When the retile plan is ACTIVE the SUMMA process grid spans the COARSE
      // (T) external M/N tile counts -- the broadcast keys/roots on the coarse
      // grid even though operands stay tiled at the fine (U) M/N count -- while
      // the element extents (m,n) stay the total U element counts (retiling
      // does not change the element space). So the 2-d cap and the per-slab
      // ProcGrid below must both use coarse M/N. Inactive plans keep the stock
      // U grid byte-for-byte (coarse_M_/coarse_N_ are the identity then).
      // COARSEN-H: n_slabs_ (== nh_) is now the COARSE fused-tile count
      // (coarse_H_), so the proc_h_ slab heuristic spreads ranks over the COARSE
      // slabs. The operand/result SlabbedPmap replication, however, uses
      // n_slabs_u (the U fused-tile count) because the arrays carry U-H tiles --
      // SUMMA iterates coarse slabs while the storage stays at U. Identity-H =>
      // n_slabs_ == n_slabs_u (byte-for-byte the prior behavior).
      // The coarse-H np>1 distribution (proc_h_ over coarse slabs composed with
      // a U-replicated SlabbedPmap) is deferred; COARSEN-H is
      // np=1 scope here (rejected for refine at np>1 above, and the coarse-H
      // np>1 path is not yet exercised).
      const size_type M_grid = coarse_M_(M);
      const size_type N_grid = coarse_N_(N);
      const size_type P = world->size();
      proc_h_ = 1ul;
      if (n_slabs_ > 1ul && P > 1ul) {
        const size_type p2d_cap = std::min<size_type>(P, M_grid * N_grid);
        proc_h_ = std::min<size_type>(n_slabs_,
                                      std::max<size_type>(1ul, P / p2d_cap));
      }
      // keep the invariant proc_h_ == 1 => proc_h_stride_ == 0 (the ungrouped
      // 2-d case) so downstream logic can key off either field; for grouped
      // grids it is the per-plane world-rank count P / proc_h_.
      proc_h_stride_ = (proc_h_ == 1ul) ? 0ul : P / proc_h_;

      if (proc_h_ == 1ul) {
        if (plan_.active) {
          // ACTIVE coarsen/identity (np>=1): COMPOSE the coarse co-location of
          // the ordinary 2-d path (init_distribution :865-892) with the slab
          // replication this general path needs. The per-slab SUMMA grid spans
          // the COARSE external M/N; operands/result stay at the FINE (U)
          // tiling but every U EXTERNAL tile is co-located on the rank owning
          // its covering coarse T-cell (matching coarse phase pmap), so the
          // in-step gather is local and the coarse-keyed broadcast is
          // consistent. SlabbedPmap then replicates that per-slab base map over
          // every (identity-H) slab. At np=1 every owner is rank 0, identical
          // to the stock fine-U grid. (Refine at np>1 was rejected above; only
          // coarsen/identity reaches here.)
          proc_grid_ = TiledArray::detail::ProcGrid(*world, M_grid, N_grid, m, n);
          const size_type K_coarse = coarse_K_(K_);
          // left U layout = [H-axes..., M-axes..., K-axes...]; phase rows =
          // M_grid, cols = K_coarse; skip the nh leading H modes. The
          // SlabbedPmap replicates over n_slabs_u (U fused-tile count) -- the
          // operand carries U-H tiles -- NOT the coarse n_slabs_.
          auto left_phase = proc_grid_.make_row_phase_pmap(K_coarse);
          left_.init_distribution(
              world, std::make_shared<TiledArray::detail::SlabbedPmap>(
                         *world,
                         make_operand_coarse_pmap(*world, left_.trange(),
                                                  plan_.summaM, plan_.summaK,
                                                  left_phase, K_coarse, nh),
                         n_slabs_u));
          // right U layout = [H-axes..., K-axes..., N-axes...]; phase rows =
          // K_coarse, cols = N_grid; skip the nh leading H modes.
          auto right_phase = proc_grid_.make_col_phase_pmap(K_coarse);
          right_.init_distribution(
              world, std::make_shared<TiledArray::detail::SlabbedPmap>(
                         *world,
                         make_operand_coarse_pmap(*world, right_.trange(),
                                                  plan_.summaK, plan_.summaN,
                                                  right_phase, N_grid, nh),
                         n_slabs_u));

          if (!pmap) {
            if (general_repermute_) {
              // trange_ is the PERMUTED target layout; make_result_coarse_pmap
              // assumes canonical [H][M][N] axis order, so build the coarse
              // result pmap on the CANONICAL trange (correct role order, no
              // out-of-bounds role-map index) and route each target tile to its
              // canonical source's owner. This both fixes the permuted-target
              // crash and co-locates the streaming re-permute push in-rank.
              const auto canonical_tr = make_trange_general();
              auto canonical_pmap =
                  std::make_shared<TiledArray::detail::SlabbedPmap>(
                      *world,
                      make_result_coarse_pmap(*world, canonical_tr,
                                              proc_grid_.make_pmap(), N_grid, nh),
                      n_slabs_u);
              pmap = make_repermuted_result_pmap(*world, trange_, canonical_tr,
                                                 canonical_pmap, outer(perm_));
            } else {
              pmap = std::make_shared<TiledArray::detail::SlabbedPmap>(
                  *world,
                  make_result_coarse_pmap(*world, trange_,
                                          proc_grid_.make_pmap(), N_grid, nh),
                  n_slabs_u);
            }
          }
          ExprEngine_::init_distribution(world, pmap);
        } else {
          // Inactive: stock per-slab U grid over the whole world (M_grid==M,
          // N_grid==N, K_coarse==K_ so this is byte-for-byte the prior code).
          proc_grid_ = TiledArray::detail::ProcGrid(*world, M, N, m, n);

          // Initialize children with slab-replicated SUMMA phase maps
          left_.init_distribution(
              world, std::make_shared<TiledArray::detail::SlabbedPmap>(
                         *world, proc_grid_.make_row_phase_pmap(K_), n_slabs_));
          right_.init_distribution(
              world, std::make_shared<TiledArray::detail::SlabbedPmap>(
                         *world, proc_grid_.make_col_phase_pmap(K_), n_slabs_));

          // Initialize the process map if not already defined
          if (!pmap)
            pmap = std::make_shared<TiledArray::detail::SlabbedPmap>(
                *world, proc_grid_.make_pmap(), n_slabs_);
          ExprEngine_::init_distribution(world, pmap);
        }
      } else {
        // grouped (proc_h_ > 1) 3-d grid. Construct this rank's GROUP-LOCAL
        // per-slab process grid (ranks outside the grouped prefix of the world
        // construct a valid not-in-grid instance). The grid shape is a pure
        // function of (proc_h_stride_, <M/N or coarse M/N>, m, n), so it is
        // congruent across all groups, and the CyclicPmap factories below emit
        // GROUP-LOCAL owners in [0, proc_h_stride_) which the h-grouped
        // SlabbedPmap offsets by each slab's group.
        const size_type rank = world->rank();
        const bool in_groups = rank < proc_h_ * proc_h_stride_;
        const ProcessID grid_offset =
            in_groups ? ProcessID((rank / proc_h_stride_) * proc_h_stride_)
                      : ProcessID(0);

        if (plan_.active) {
          // ACTIVE coarsen/identity, h-grouped (the np>=2 "ride single-tile"
          // optimum: M_grid * N_grid collapses to 1 so the 2-d cap is 1 and the
          // surplus ranks ride the slab axis, proc_h_ = min(n_slabs_, P)). This
          // is the SAME coarse co-location as the ungrouped active branch above
          // (init_distribution :865-892 mirrored, n_skip = nh), composed with
          // the GROUP-LOCAL coarse process grid and the h-grouped SlabbedPmap.
          // The group-local proc_grid_ phase pmaps emit GROUP-LOCAL coarse-cell
          // owners in [0, proc_h_stride_); make_operand_coarse_pmap /
          // make_result_coarse_pmap co-locate every U external tile on the
          // group-local rank owning its covering coarse T-cell; the h-grouped
          // SlabbedPmap then offsets each slab's base owner by the group of the
          // COARSE slab covering it (coarse slab c -> group c % proc_h_ at
          // world-rank offset (c % proc_h_) * proc_h_stride_). The SUMMA grid /
          // step geometry rides the COARSE slabs (proc_h_ = min(n_slabs_, P)),
          // so the SlabbedPmap replicates over the U slab count n_slabs_u_
          // (operands/result carry U-H tiles) but groups by the covering coarse
          // slab via u_per_coarse_slab = n_slabs_u_ / n_slabs_. The result base
          // is the GLOBAL coarse slab index in initialize_active /
          // finalize_active (a coarse slab carves into ALL its covered U fused
          // tiles, each at base uh * per_u_mn), and result_tile_owner delegates
          // to THIS pmap, so the carve's set_tile lands on the rank that ran
          // that coarse slab. Identity-H => n_slabs_u_ == n_slabs_ and
          // u_per_coarse_slab == 1, byte-for-byte the prior behavior.
          // Refine at np>1 was rejected above; COARSEN-H is now supported here
          // single dense H axis (guarded above).
          const size_type u_per_coarse_slab = n_slabs_u / n_slabs_;
          proc_grid_ = TiledArray::detail::ProcGrid(
              *world, TiledArray::detail::rank_subset, grid_offset,
              proc_h_stride_, M_grid, N_grid, m, n);
          const size_type K_coarse = coarse_K_(K_);
          // left U layout = [H-axes..., M-axes..., K-axes...]; phase rows =
          // M_grid, cols = K_coarse; skip the nh leading H modes.
          auto left_phase = proc_grid_.make_row_phase_pmap(K_coarse);
          left_.init_distribution(
              world, std::make_shared<TiledArray::detail::SlabbedPmap>(
                         *world,
                         make_operand_coarse_pmap(*world, left_.trange(),
                                                  plan_.summaM, plan_.summaK,
                                                  left_phase, K_coarse, nh),
                         n_slabs_u, proc_h_, proc_h_stride_,
                         u_per_coarse_slab));
          // right U layout = [H-axes..., K-axes..., N-axes...]; phase rows =
          // K_coarse, cols = N_grid; skip the nh leading H modes.
          auto right_phase = proc_grid_.make_col_phase_pmap(K_coarse);
          right_.init_distribution(
              world, std::make_shared<TiledArray::detail::SlabbedPmap>(
                         *world,
                         make_operand_coarse_pmap(*world, right_.trange(),
                                                  plan_.summaK, plan_.summaN,
                                                  right_phase, N_grid, nh),
                         n_slabs_u, proc_h_, proc_h_stride_,
                         u_per_coarse_slab));

          if (!pmap) {
            if (general_repermute_) {
              // See the proc_h_==1 branch: the PERMUTED target trange cannot go
              // through make_result_coarse_pmap (canonical-order assumption);
              // build the grouped coarse result pmap on the CANONICAL trange and
              // co-locate each target tile with its canonical source.
              const auto canonical_tr = make_trange_general();
              auto canonical_pmap =
                  std::make_shared<TiledArray::detail::SlabbedPmap>(
                      *world,
                      make_result_coarse_pmap(*world, canonical_tr,
                                              proc_grid_.make_pmap(), N_grid, nh),
                      n_slabs_u, proc_h_, proc_h_stride_, u_per_coarse_slab);
              pmap = make_repermuted_result_pmap(*world, trange_, canonical_tr,
                                                 canonical_pmap, outer(perm_));
            } else {
              pmap = std::make_shared<TiledArray::detail::SlabbedPmap>(
                  *world,
                  make_result_coarse_pmap(*world, trange_,
                                          proc_grid_.make_pmap(), N_grid, nh),
                  n_slabs_u, proc_h_, proc_h_stride_, u_per_coarse_slab);
            }
          }
          ExprEngine_::init_distribution(world, pmap);
        } else {
          // Inactive: stock group-local per-slab U grid (M_grid==M, N_grid==N,
          // K_coarse==K_ so this is byte-for-byte the prior code).
          proc_grid_ = TiledArray::detail::ProcGrid(
              *world, TiledArray::detail::rank_subset, grid_offset,
              proc_h_stride_, M, N, m, n);

          left_.init_distribution(
              world, std::make_shared<TiledArray::detail::SlabbedPmap>(
                         *world, proc_grid_.make_row_phase_pmap(K_), n_slabs_,
                         proc_h_, proc_h_stride_));
          right_.init_distribution(
              world, std::make_shared<TiledArray::detail::SlabbedPmap>(
                         *world, proc_grid_.make_col_phase_pmap(K_), n_slabs_,
                         proc_h_, proc_h_stride_));

          // Initialize the process map if not already defined
          if (!pmap)
            pmap = std::make_shared<TiledArray::detail::SlabbedPmap>(
                *world, proc_grid_.make_pmap(), n_slabs_, proc_h_,
                proc_h_stride_);
          ExprEngine_::init_distribution(world, pmap);
        }
      }
    }
  }

  /// Streaming tile re-permute op for general products whose target layout
  /// differs from the canonical (fused..., free...) result layout: the
  /// batched tile op must stay perm-free, so the consumer-side unary eval
  /// applies the result permutation per tile instead
  struct GeneralRepermuteOp {
    typedef value_type result_type;
    typedef value_type argument_type;
    static constexpr bool is_consumable = false;
    /// Only the *outer* (result-layout) permutation is applied here; inner
    /// (within-cell) permutation of tensor-of-tensor results is handled
    /// separately (see init_struct_general / implicit_permute_inner_), so this
    /// op stores a plain outer Permutation to avoid accidentally permuting
    /// inner contents.
    Permutation perm;
    /// false when the consumer fuses the permutation into its own operation
    /// (implicit permute, e.g. a transposed GEMM): then only the tile
    /// ordinals/trange are remapped (by the host UnaryEvalImpl) and the tile
    /// contents are delivered in the canonical layout
    bool permute_contents = true;
    result_type operator()(const argument_type& tile) const {
      if (!permute_contents) return tile;
      TiledArray::detail::Noop<value_type, value_type, false> noop;
      return noop(tile, perm);
    }
  };

  /// Construct the distributed evaluator of a general product

  /// \return The batched-Summa distributed evaluator for this expression,
  /// wrapped in a streaming re-permute when the target layout differs from
  /// the canonical result layout
  dist_eval_type make_dist_eval_general() const {
    typedef TiledArray::detail::BatchedContractReduce<op_type> batched_op_type;
    typedef TiledArray::detail::Summa<typename left_type::dist_eval_type,
                                      typename right_type::dist_eval_type,
                                      batched_op_type, typename Derived::policy>
        impl_type;

    typename left_type::dist_eval_type left = left_.make_dist_eval();
    typename right_type::dist_eval_type right = right_.make_dist_eval();

    if (!general_repermute_) {
      // Active plan => SUMMA steps over the COARSE (T) K count while operands
      // stay at the FINE (U) K count (pmaps keyed to K_); thread the fine
      // count separately for the in-step U-block gather. Inactive => K_ == K_.
      std::shared_ptr<impl_type> pimpl = std::make_shared<impl_type>(
          left, right, *world_, trange_, shape_, pmap_, perm_,
          batched_op_type(op_, n_fused_modes_), coarse_K_(K_), proc_grid_,
          n_slabs_, proc_h_, proc_h_stride_, plan_, /*k_fine=*/K_);
      return dist_eval_type(pimpl);
    }

    // evaluate in the canonical layout (Summa with perm-free op), then
    // re-permute tiles to the target layout with a streaming unary eval;
    // trange_/shape_ hold the target-layout structures (see
    // init_struct_general), the canonical ones are recomputed here
    auto const canonical_trange = make_trange_general();
    auto const canonical_shape = [this]() {
      auto s = make_shape_general();
      if (ExprEngine_::override_ptr_ && ExprEngine_::override_ptr_->shape) {
        // the consumer-supplied mask is expressed in the target layout
        auto const inv_perm = outer(perm_).inv();
        s = s.mask(ExprEngine_::override_ptr_->shape->perm(inv_perm));
      }
      return s;
    }();
    // the inner Summa's result placement must be slab-replicated (the owner
    // of a tile independent of its slab index), regardless of the
    // (target-layout) pmap the consumer supplied for this node. The result is
    // delivered at the FINE (U) tiling, so the SlabbedPmap replicates over the
    // U slab count n_slabs_u_ (== n_slabs_ for identity-H); under a grouped
    // COARSEN-H plan it groups U slabs by their covering coarse slab via
    // u_per_coarse_slab = n_slabs_u_ / n_slabs_ (mirrors the non-repermute
    // result pmap above so the carve's owner is consistent).
    // The inner Summa delivers its result at the FINE (U) canonical_trange, so
    // its per-slab base pmap must map every FINE U external tile to the owner of
    // its covering COARSE cell -- exactly what the non-repermute active branches
    // do via make_result_coarse_pmap (init_distribution_general, the active
    // branches). A bare proc_grid_.make_pmap() base is COARSE-sized
    // (coarse_M*coarse_N), so SlabbedPmap.size() = coarse * n_slabs_u !=
    // canonical_trange.volume() = fine * n_slabs_u, which trips the TensorImpl
    // ctor size assert and, in Release, deadlocks at np>1.
    // For an INACTIVE plan coarse == fine so proc_grid_.make_pmap() is already
    // U-sized; keep it (make_result_coarse_pmap requires an active plan's
    // summaM/summaN roles). proc_grid_.cols() == N_grid (the coarse N tile
    // count); n_fused_modes_ == nh (the leading fused modes to skip).
    auto canonical_base =
        plan_.active
            ? make_result_coarse_pmap(*world_, canonical_trange,
                                      proc_grid_.make_pmap(), proc_grid_.cols(),
                                      n_fused_modes_)
            : proc_grid_.make_pmap();
    auto canonical_pmap =
        proc_h_ == 1ul
            ? std::make_shared<TiledArray::detail::SlabbedPmap>(
                  *world_, canonical_base, n_slabs_u_)
            : std::make_shared<TiledArray::detail::SlabbedPmap>(
                  *world_, canonical_base, n_slabs_u_, proc_h_, proc_h_stride_,
                  n_slabs_u_ / n_slabs_);
    std::shared_ptr<impl_type> pimpl = std::make_shared<impl_type>(
        left, right, *world_, canonical_trange, canonical_shape, canonical_pmap,
        BipartitePermutation{}, batched_op_type(op_, n_fused_modes_),
        coarse_K_(K_), proc_grid_, n_slabs_, proc_h_, proc_h_stride_, plan_,
        /*k_fine=*/K_);
    dist_eval_type canonical(pimpl);

    typedef TiledArray::detail::UnaryEvalImpl<
        dist_eval_type, GeneralRepermuteOp, typename Derived::policy>
        repermute_impl_type;
    std::shared_ptr<repermute_impl_type> wrapper =
        std::make_shared<repermute_impl_type>(
            canonical, *world_, trange_, shape_, pmap_, perm_,
            GeneralRepermuteOp{outer(perm_), !this->implicit_permute_outer_});
    return dist_eval_type(wrapper);
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
    if constexpr (denest_to_scalar) {
      // dot_inner: both operand cells are inner tensors, the result cell is a
      // scalar. The inner modes are fully contracted (a flat, non-conjugating
      // dot of the operand cells) and accumulated into the scalar result cell.
      // This mirrors the phantom-unit denest in init_inner_tile_op_owning_ but
      // writes a bare scalar instead of a unit-extent [1] cell. The outer
      // ContractReduce (built in init_struct) routes the !plain_tensors case to
      // gemm(result, left, right, helper, elem_muladd_op), invoking this op per
      // outer cell.
      const scalar_type factor = this->factor_;
      // shared flat (non-conjugating) scalar dot of two inner cells, scaled by
      // factor; returns the contribution for one outer cell (0 if either
      // operand cell is empty). The numerically-sensitive accumulation lives
      // here only -- see also denest_to_scalar at contract_reduce.h.
      auto flat_dot = [factor](const left_tile_element_type& left,
                               const right_tile_element_type& right)
          -> result_tile_element_type {
        if (left.empty() || right.empty()) return result_tile_element_type{0};
        const std::size_t n = left.range().volume();
        TA_ASSERT(n == right.range().volume());
        const auto* lp = left.data();
        const auto* rp = right.data();
        result_tile_element_type acc{0};
        for (std::size_t j = 0; j < n; ++j) acc += lp[j] * rp[j];
        return static_cast<result_tile_element_type>(factor) * acc;
      };
      this->element_nonreturn_op_ = [flat_dot](
                                        result_tile_element_type& result,
                                        const left_tile_element_type& left,
                                        const right_tile_element_type& right) {
        result += flat_dot(left, right);
      };
      // value-returning form for the outer-Hadamard regime (the Mult binary
      // tile op maps it over the outer cells via TiledArray::binary)
      this->element_return_op_ = [flat_dot](
                                     const left_tile_element_type& left,
                                     const right_tile_element_type& right)
          -> result_tile_element_type { return flat_dot(left, right); };
      return;
    }
    if constexpr (TiledArray::detail::is_tensor_of_tensor_v<result_tile_type>) {
      constexpr bool tot_x_tot = TiledArray::detail::is_tensor_of_tensor_v<
          result_tile_type, left_tile_type, right_tile_type>;
      if constexpr (tot_x_tot &&
                    TiledArray::is_tensor_view_v<result_tile_element_type>) {
        // ToT x ToT with non-owning view inner cells (e.g. ArenaTensor). A
        // view cell cannot host a value-returning inner op, so the
        // owning-cell inner-op builder cannot be used. The supported nested
        // products are:
        //  - the elementwise pure Hadamard (outer Hadamard, inner Hadamard),
        //    where the inner element op is unused anyway -- MultEngine::
        //    make_tile_op passes none and the outer Mult tile op recurses
        //    through Tensor<view>::mult -- so element_*_op_ is left null;
        //  - inner Hadamard under outer Contraction, routed through the
        //    arena fast path with a left_range plan and a per-cell
        //    `r += l * rr` (optionally scaled) op: result cells are
        //    pre-shaped from non-empty left cells, then accumulated in
        //    place over the K-panel;
        //  - inner Contraction (incl. inner outer-product) under either
        //    outer regime, routed through the arena fast path: it writes
        //    results in place into pre-shaped view cells, so only
        //    element_nonreturn_op_ is needed.
        // Every other nested product is deferred.
        const auto inner_prod = this->inner_product_type();
        if (inner_prod == TensorProduct::Hadamard &&
            this->product_type() == TensorProduct::Hadamard) {
          // pure Hadamard: element_*_op_ left null
        } else if (inner_prod == TensorProduct::Hadamard &&
                   this->outer_product_uses_summa()) {
          // outer Contraction + inner Hadamard on view inner tiles.
          // Mirror the owning-tile path (init_inner_tile_op_owning_): the
          // SUMMA shapes each result cell from a non-empty left inner cell
          // (left_range plan), and the per-cell op accumulates `r += l * rr`
          // -- or `r += (l * rr) * factor_` when scaled -- via
          // fused_hadamard_inplace into the pre-shaped view cell. No
          // value-returning per-cell op is needed, so this works for view
          // cells; non-identity inner result permutation is rejected here
          // (the owning fallback that materializes a permuted return cell
          // cannot run for views).
          constexpr bool arena_eligible_h_view =
              TiledArray::detail::is_contraction_arena_tot_v<
                  result_tile_type, left_tile_type, right_tile_type>;
          if constexpr (!arena_eligible_h_view) {
            TA_EXCEPTION(
                "nested Hadamard on view inner tiles is supported only for "
                "arena-backed tensors-of-tensors");
          } else {
            this->arena_plan_ = TiledArray::detail::make_contraction_arena_plan<
                result_tile_type, left_tile_type, right_tile_type>(
                TiledArray::detail::ArenaInnerShapeKind::left_range,
                std::nullopt, inner(this->perm_));
            if (!bool(this->arena_plan_))
              TA_EXCEPTION(
                  "nested Hadamard on view inner tiles: the arena fast path "
                  "was inactive (arena disabled, or a non-identity inner "
                  "result permutation -- not yet supported on view cells)");
            if (this->factor_ == scalar_type{1}) {
              this->element_nonreturn_op_ =
                  TiledArray::detail::make_fused_hadamard_lambda<
                      result_tile_element_type, left_tile_element_type,
                      right_tile_element_type>();
            } else {
              this->element_nonreturn_op_ =
                  TiledArray::detail::make_fused_hadamard_scaled_lambda<
                      result_tile_element_type, left_tile_element_type,
                      right_tile_element_type>(this->factor_);
            }
          }
          // element_return_op_ left null: a view cell cannot be
          // value-returned (see the init_struct precondition check).
        } else if (inner_prod == TensorProduct::Contraction) {
          constexpr bool arena_eligible =
              TiledArray::detail::is_contraction_arena_tot_v<
                  result_tile_type, left_tile_type, right_tile_type>;
          if constexpr (!arena_eligible) {
            TA_EXCEPTION(
                "nested contraction on view inner tiles is supported only "
                "for arena-backed tensors-of-tensors");
          } else {
            // Phantom-unit denest: result inner indices all phantom (⊗ₙ) -- the
            // real inner modes are fully contracted, so the inner product is a
            // flat dot into a unit-extent [1]^phantom_rank cell. The arena plan
            // shapes the [1] result cells (unit_range); the per-cell op fills
            // the lone element via the dot. Operands are read flat, so no view
            // cell carries the phantom mode and no GEMM rank match is required.
            const auto result_inner = inner(this->indices_);
            bool result_inner_all_phantom = result_inner.size() > 0;
            for (std::size_t m = 0; m < result_inner.size(); ++m)
              if (!TiledArray::detail::is_phantom_unit_label(result_inner[m])) {
                result_inner_all_phantom = false;
                break;
              }
            if (result_inner_all_phantom) {
              const scalar_type factor = this->factor_;
              this->element_nonreturn_op_ =
                  [factor](result_tile_element_type& result,
                           const left_tile_element_type& left,
                           const right_tile_element_type& right) {
                    if (left.empty() || right.empty()) return;
                    using Numeric =
                        typename result_tile_element_type::numeric_type;
                    const std::size_t n = left.range().volume();
                    TA_ASSERT(n == right.range().volume());
                    const auto* lp = left.data();
                    const auto* rp = right.data();
                    Numeric acc{0};
                    for (std::size_t j = 0; j < n; ++j) acc += lp[j] * rp[j];
                    // result cell is pre-shaped [1] by the unit_range plan.
                    result.data()[0] += static_cast<Numeric>(factor) * acc;
                  };
              if (this->outer_product_uses_summa()) {
                this->arena_plan_ =
                    TiledArray::detail::make_contraction_arena_plan<
                        result_tile_type, left_tile_type, right_tile_type>(
                        TiledArray::detail::ArenaInnerShapeKind::unit_range,
                        std::nullopt, Permutation{}, result_inner.size());
                if (!bool(this->arena_plan_))
                  TA_EXCEPTION(
                      "phantom-unit denest on view inner tiles: the arena fast "
                      "path was inactive (arena disabled)");
              } else {
                // outer Hadamard: a whole-tile arena op that shapes each
                // result outer cell as a unit-extent [1] cell and fills it via
                // the phantom-dot per-cell op.
                this->arena_hadamard_tile_op_ =
                    [cell_op = this->element_nonreturn_op_,
                     phantom_rank = result_inner.size()](
                        const left_tile_type& l,
                        const right_tile_type& r) -> result_tile_type {
                  return TiledArray::detail::arena_hadamard_phantom_dot<
                      result_tile_type>(l, r, phantom_rank, cell_op);
                };
              }
            } else {
              using op_type = TiledArray::detail::ContractReduce<
                  result_tile_element_type, left_tile_element_type,
                  right_tile_element_type, scalar_type>;
              // The inner op is built *perm-free* on purpose. factor_ is
              // absorbed into element_nonreturn_op_; operand inner transposes
              // are folded into the inner GEMM via left_/right_inner_permtype_.
              // A non-identity inner *result* permutation is NOT placed on this
              // op (make_fused_contraction_lambda asserts a perm-free op); it
              // is applied downstream instead -- by op_'s post-processing
              // permute for a contraction outer product, or by
              // arena_hadamard_inner_contract's slab-level post-pass for a
              // Hadamard outer product.
              auto contrreduce_op = op_type(
                  to_cblas_op(this->left_inner_permtype_),
                  to_cblas_op(this->right_inner_permtype_), this->factor_,
                  inner_size(this->indices_), inner_size(this->left_indices_),
                  inner_size(this->right_indices_));
              // perm-free per-cell in-place contraction; used by both outer
              // regimes below
              this->element_nonreturn_op_ =
                  TiledArray::detail::make_fused_contraction_lambda<
                      result_tile_element_type, left_tile_element_type,
                      right_tile_element_type>(contrreduce_op);
              if (this->outer_product_uses_summa()) {
                // outer contraction: the SUMMA result is shaped from operand
                // inner cells by arena_plan_; op_'s post-processing permute
                // applies the (outer + inner) result permutation.
                this->arena_plan_ =
                    TiledArray::detail::make_contraction_arena_plan<
                        result_tile_type, left_tile_type, right_tile_type>(
                        TiledArray::detail::ArenaInnerShapeKind::
                            gemm_result_range,
                        std::make_optional(contrreduce_op.gemm_helper()),
                        Permutation{});
                if (!bool(this->arena_plan_))
                  TA_EXCEPTION(
                      "nested contraction on view inner tiles: the arena fast "
                      "path was inactive (arena disabled)");
                // ce+e (hce+e): inner OUTER product (no inner contraction)
                // under outer contraction on arena view cells -> one strided
                // DGEMM per result cell (ride the contracted index into BLAS
                // K). Only the canonical perm-free layout is fused; a
                // non-identity inner result perm is applied downstream and left
                // to the per-cell path here.
                // The strided kernel is specialized to view (arena) inner cells
                // with double storage, and its static_assert requires that of
                // ALL THREE operands (result, left, right). Gate on the same
                // 3-operand predicate so a mixed-operand contraction (e.g. a
                // view/double result with a non-view or non-double operand, or
                // float/complex inner) stays on the generic per-cell path and
                // never instantiates the double-view-only kernel (which would
                // be a hard compile error rather than a graceful fallback).
                if constexpr (
                    TiledArray::is_tensor_view_v<result_tile_element_type> &&
                    TiledArray::is_tensor_view_v<left_tile_element_type> &&
                    TiledArray::is_tensor_view_v<right_tile_element_type> &&
                    std::is_same_v<
                        typename result_tile_element_type::numeric_type,
                        double> &&
                    std::is_same_v<
                        typename left_tile_element_type::numeric_type,
                        double> &&
                    std::is_same_v<
                        typename right_tile_element_type::numeric_type,
                        double>) {
                  if (contrreduce_op.gemm_helper().num_contract_ranks() == 0 &&
                      (!bool(inner(this->perm_)) ||
                       inner(this->perm_).is_identity())) {
                    const scalar_type factor = this->factor_;
                    this->arena_strided_dgemm_ce_e_tile_op_ =
                        [factor](result_tile_type& Cc, const left_tile_type& Lt,
                                 const right_tile_type& Rt,
                                 const math::GemmHelper& gh) {
                          using integer = TiledArray::math::blas::integer;
                          integer M, N, K;
                          gh.compute_matrix_sizes(M, N, K, Lt.range(),
                                                  Rt.range());
                          TiledArray::detail::arena_strided_dgemm_ce_e(
                              Cc, Lt, Rt, static_cast<std::size_t>(M),
                              static_cast<std::size_t>(N),
                              static_cast<std::size_t>(K), gh.left_op(),
                              gh.right_op(), double(factor));
                        };
                  }
                  // ce+ce (hce+ce): inner CONTRACTION (num_contract_ranks() >=
                  // 1) under outer contraction. One operand inner must be a
                  // pure contraction vector; that side's outer-external rides
                  // BLAS M with one strided DGEMM per (batch, other-external,
                  // outer-contraction) cell. Two orientations (right-clean ->
                  // ce_ce_right, left-clean -> ce_ce_left); see the either-side
                  // rule below. Sibling of the ce+e arm above (disjoint
                  // num_contract_ranks gate) so at most one strided op
                  // installs.
                  const auto& inner_gh = contrreduce_op.gemm_helper();
                  const bool inner_contraction =
                      inner_gh.num_contract_ranks() >= 1;
                  // STRIDED-APPLICABILITY RULE (matrix x matrix exclusion).
                  // The ce+ce core assumes the RIGHT inner cell is a pure
                  // contraction vector R[k,μ̃](a4) -- i.e. the right operand
                  // carries NO inner external. When BOTH operand inners carry
                  // an external (a genuine inner matrix x matrix, e.g.
                  // C(m,n;μ,ν) = A(m,k;μ,κ) * B(k,n;κ,ν)), riding μ̃ into BLAS M
                  // would need a two-level stride the kernel cannot represent:
                  // the per-cell `clean` probe fails and the GEMV fallback then
                  // silently contributes nothing (the result cell volume P*Q no
                  // longer matches the left cell). Refuse the install so such
                  // shapes take the generic per-cell contraction path. The
                  // right inner-external rank is right_rank -
                  // num_contract_ranks; the supported (right-clean) shape has
                  // it == 0. EITHER-SIDE rule: an inner contraction is
                  // strided-castable iff at least ONE operand inner is a pure
                  // contraction vector (no inner external). right-clean ->
                  // ce_ce_right (ride the right-external into BLAS M);
                  // left-clean -> ce_ce_left (ride the left-external into BLAS
                  // M). When BOTH inners carry an external (a genuine inner
                  // matrix x matrix, e.g. C(m,n;μ,ν) = A(m,k;μ,κ) * B(k,n;κ,ν))
                  // neither fires and the generic per-cell path runs. An
                  // operand's inner-external rank is its rank -
                  // num_contract_ranks; clean == 0.
                  const bool right_inner_clean =
                      inner_gh.right_rank() == inner_gh.num_contract_ranks();
                  const bool left_inner_clean =
                      inner_gh.left_rank() == inner_gh.num_contract_ranks();
                  // Derive the outer-contracted rank `oc` from the outer index
                  // sizes (same helper used by the outer op when building op_).
                  // for a general product the leading fused modes do not
                  // participate in the outer GEMM (they are folded into the
                  // tile batch dimension), so exclude them from the rank
                  // accounting
                  const auto nh = this->n_fused_outer_modes();
                  const auto oc = (outer_size(this->left_indices_) - nh +
                                   outer_size(this->right_indices_) - nh -
                                   (outer_size(this->indices_) - nh)) /
                                  2;
                  // the ridden operand must carry an outer external to ride.
                  const bool right_has_ext =
                      outer_size(this->right_indices_) - nh > oc;
                  const bool left_has_ext =
                      outer_size(this->left_indices_) - nh > oc;
                  // canonical inner orientation: identity == "no inner
                  // transpose". right core assumes L=(a1,a4), R=(a4); left core
                  // assumes L=(a4), R=(a4,b1). Either way BOTH inner permtypes
                  // must be identity and there must be no inner result perm.
                  // This gate is LOAD-BEARING for correctness.
                  const bool inner_canonical =
                      this->left_inner_permtype_ ==
                          TiledArray::expressions::PermutationType::identity &&
                      this->right_inner_permtype_ ==
                          TiledArray::expressions::PermutationType::identity &&
                      (!bool(inner(this->perm_)) ||
                       inner(this->perm_).is_identity());
                  // RELAXED gate. The strided kernel can fold a
                  // matrix_transpose of the EXTERNAL-carrying operand into the
                  // inner GEMM op flag (zero-copy), because matrix_transpose is
                  // a contiguous two-block swap (permopt) so the cell still
                  // flattens cleanly. The CLEAN (pure contraction vector) side
                  // must stay identity, the result inner must not be permuted,
                  // and a `general` inner perm still falls back. right arm:
                  // left carries the external (may be T), right is the vector
                  // (id). left arm: mirror.
                  auto inner_pt_ok =
                      [](TiledArray::expressions::PermutationType p) {
                        return p == TiledArray::expressions::PermutationType::
                                        identity ||
                               p == TiledArray::expressions::PermutationType::
                                        matrix_transpose;
                      };
                  const bool no_result_inner_perm =
                      !bool(inner(this->perm_)) ||
                      inner(this->perm_).is_identity();
                  const bool right_arm_ok =
                      inner_contraction && no_result_inner_perm &&
                      right_inner_clean && right_has_ext &&
                      this->right_inner_permtype_ ==
                          TiledArray::expressions::PermutationType::identity &&
                      inner_pt_ok(this->left_inner_permtype_);
                  const bool left_arm_ok =
                      inner_contraction && no_result_inner_perm &&
                      left_inner_clean && left_has_ext &&
                      this->left_inner_permtype_ ==
                          TiledArray::expressions::PermutationType::identity &&
                      inner_pt_ok(this->right_inner_permtype_);
                  if (right_arm_ok) {
                    const scalar_type factor = this->factor_;
                    const bool left_inner_T =
                        this->left_inner_permtype_ ==
                        TiledArray::expressions::PermutationType::
                            matrix_transpose;
                    this->arena_strided_dgemm_ce_ce_right_tile_op_ =
                        [factor, left_inner_T](result_tile_type& Cc,
                                               const left_tile_type& Lt,
                                               const right_tile_type& Rt,
                                               const math::GemmHelper& gh) {
                          math::blas::integer Mo = 0, No = 0, Ko = 0;
                          gh.compute_matrix_sizes(Mo, No, Ko, Lt.range(),
                                                  Rt.range());
                          TiledArray::detail::arena_strided_dgemm_ce_ce_right(
                              Cc, Lt, Rt, static_cast<std::size_t>(Mo),
                              static_cast<std::size_t>(No),
                              static_cast<std::size_t>(Ko), gh.left_op(),
                              gh.right_op(), double(factor), left_inner_T);
                        };
                  } else if (left_arm_ok) {
                    const scalar_type factor = this->factor_;
                    const bool right_inner_T =
                        this->right_inner_permtype_ ==
                        TiledArray::expressions::PermutationType::
                            matrix_transpose;
                    this->arena_strided_dgemm_ce_ce_left_tile_op_ =
                        [factor, right_inner_T](result_tile_type& Cc,
                                                const left_tile_type& Lt,
                                                const right_tile_type& Rt,
                                                const math::GemmHelper& gh) {
                          math::blas::integer Mo = 0, No = 0, Ko = 0;
                          gh.compute_matrix_sizes(Mo, No, Ko, Lt.range(),
                                                  Rt.range());
                          TiledArray::detail::arena_strided_dgemm_ce_ce_left(
                              Cc, Lt, Rt, static_cast<std::size_t>(Mo),
                              static_cast<std::size_t>(No),
                              static_cast<std::size_t>(Ko), gh.left_op(),
                              gh.right_op(), double(factor), right_inner_T);
                        };
                  }
                  // [strided-dgemm] install-decision instrumentation. For each
                  // ToT contraction reaching this double-view path, report
                  // whether a strided-DGEMM regime (hce+e / hc+e / hce+ce)
                  // FIRED or the contraction REVERTED to the generic by-cell
                  // evaluation path (with the blocking guard). Gated by
                  // TA_STRIDED_DGEMM_VERBOSE; a no-op otherwise.
                  if (TiledArray::detail::strided_dgemm_verbose()) {
                    if (this->arena_strided_dgemm_ce_e_tile_op_) {
                      TiledArray::detail::strided_dgemm_log(
                          left_has_ext ? "hce+e  FIRES (ce+e)"
                                       : "hc+e   FIRES (ce+e)");
                    } else if (this->arena_strided_dgemm_ce_ce_right_tile_op_) {
                      TiledArray::detail::strided_dgemm_log(
                          "hce+ce FIRES (ce+ce right)");
                    } else if (this->arena_strided_dgemm_ce_ce_left_tile_op_) {
                      TiledArray::detail::strided_dgemm_log(
                          "hce+ce FIRES (ce+ce left)");
                    } else if (!inner_contraction) {
                      // ce+e candidate (no inner contraction); the only guard
                      // that can block its install is a non-identity inner
                      // result perm.
                      TiledArray::detail::strided_dgemm_log(
                          left_has_ext
                              ? "hce+e  REVERTED -> by-cell (inner result perm)"
                              : "hc+e   REVERTED -> by-cell (inner result "
                                "perm)");
                    } else if (!inner_canonical) {
                      // ce+ce candidate blocked by a non-canonical inner perm.
                      // Break down WHICH operand/result perm is non-identity
                      // (matrix_transpose 'T' vs general 'gen') and the inner
                      // ranks, so free transposes can be told apart from
                      // interleaving general perms.
                      auto pt = [](TiledArray::expressions::PermutationType p)
                          -> const char* {
                        switch (p) {
                          case TiledArray::expressions::PermutationType::
                              identity:
                            return "id";
                          case TiledArray::expressions::PermutationType::
                              matrix_transpose:
                            return "T";
                          case TiledArray::expressions::PermutationType::
                              general:
                            return "gen";
                        }
                        return "?";
                      };
                      std::string msg =
                          "hce+ce REVERTED -> by-cell (non-canonical inner "
                          "perm: L=";
                      msg += pt(this->left_inner_permtype_);
                      msg += " R=";
                      msg += pt(this->right_inner_permtype_);
                      msg += " resInner=";
                      msg += (bool(inner(this->perm_)) ? "perm" : "id");
                      msg += "; innerRank L/R/contract=";
                      msg += std::to_string(inner_gh.left_rank());
                      msg += "/";
                      msg += std::to_string(inner_gh.right_rank());
                      msg += "/";
                      msg += std::to_string(inner_gh.num_contract_ranks());
                      msg += ")";
                      TiledArray::detail::strided_dgemm_log(msg.c_str());
                    } else {
                      // ce+ce candidate, canonical inner, but no clean side /
                      // no outer external to ride.
                      TiledArray::detail::strided_dgemm_log(
                          !(right_inner_clean || left_inner_clean)
                              ? "hce+ce REVERTED -> by-cell (matrix x matrix, "
                                "no clean inner side)"
                              : "hce+ce REVERTED -> by-cell (no outer external "
                                "to ride)");
                    }
                  }
                }
              } else {
                // outer Hadamard: MultEngine builds a binary tile op, which
                // cannot use a value-returning per-cell op. Supply a whole-tile
                // arena op that shapes the result from per-cell inner GEMMs and
                // fills it in place; the inner result permutation is a
                // slab-level post-pass inside the kernel.
                this->arena_hadamard_tile_op_ =
                    [cell_op = this->element_nonreturn_op_,
                     inner_gh = contrreduce_op.gemm_helper(),
                     inner_perm = inner(this->perm_)](
                        const left_tile_type& l,
                        const right_tile_type& r) -> result_tile_type {
                  return TiledArray::detail::arena_hadamard_inner_contract<
                      result_tile_type>(l, r, inner_gh, cell_op, inner_perm);
                };
              }
            }
          }
          // element_return_op_ left null: a view cell cannot be
          // value-returned (see the init_struct precondition check).
        } else {
          TA_EXCEPTION(
              "nested non-contraction product on view inner tiles (e.g. "
              "ArenaTensor) is not yet supported; only the elementwise "
              "Hadamard product and the inner contraction are");
        }
      } else {
        init_inner_tile_op_owning_(inner_target_indices);
      }
    }
  }

  /// Builds the inner-cell element op (element_nonreturn_op_ /
  /// element_return_op_) for a nested-tensor expression. init_inner_tile_op
  /// dispatches every case here except ToT x ToT with non-owning view inner
  /// cells -- a view cell cannot host the value-returning inner ops this
  /// builder constructs.
  void init_inner_tile_op_owning_(const IndexList& inner_target_indices) {
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
          // Phantom-unit denest: every result inner index is a phantom unit
          // (⊗ₙ), i.e. the real inner modes are fully contracted. The inner
          // product is a flat (non-conjugating) dot of the operand cells
          // accumulated into the lone element of a unit-extent [1]^phantom_rank
          // result cell. Operands are read flat, so neither carries the phantom
          // mode -- no GEMM, no ContractReduce rank match. element_return_op_
          // (built below) wraps this for the outer-Hadamard regime.
          const auto result_inner = inner(this->indices_);
          bool result_inner_all_phantom = result_inner.size() > 0;
          for (std::size_t m = 0; m < result_inner.size(); ++m)
            if (!TiledArray::detail::is_phantom_unit_label(result_inner[m])) {
              result_inner_all_phantom = false;
              break;
            }
          if (result_inner_all_phantom) {
            const std::size_t phantom_rank = result_inner.size();
            const scalar_type factor = this->factor_;
            this->element_nonreturn_op_ =
                [phantom_rank, factor](result_tile_element_type& result,
                                       const left_tile_element_type& left,
                                       const right_tile_element_type& right) {
                  if (left.empty() || right.empty()) return;
                  using Numeric =
                      typename result_tile_element_type::numeric_type;
                  const std::size_t n = left.range().volume();
                  TA_ASSERT(n == right.range().volume());
                  const auto* lp = left.data();
                  const auto* rp = right.data();
                  Numeric acc{0};
                  for (std::size_t j = 0; j < n; ++j) acc += lp[j] * rp[j];
                  acc *= static_cast<Numeric>(factor);
                  if (TA::empty(result)) {
                    using R = typename result_tile_element_type::range_type;
                    TiledArray::container::svector<std::size_t> ext(
                        phantom_rank, 1);
                    result = result_tile_element_type(R(ext), Numeric{0});
                  }
                  result.data()[0] += acc;
                };
          } else {
            using op_type = TiledArray::detail::ContractReduce<
                result_tile_element_type, left_tile_element_type,
                right_tile_element_type, scalar_type>;
            // factor_ is absorbed into inner_tile_nonreturn_op_
            auto contrreduce_op =
                (inner_target_indices != inner(this->indices_))
                    ? op_type(
                          to_cblas_op(this->left_inner_permtype_),
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
            constexpr bool arena_eligible =
                TiledArray::detail::is_contraction_arena_tot_v<
                    result_tile_type, left_tile_type, right_tile_type>;
            if constexpr (arena_eligible) {
              if (this->outer_product_uses_summa()) {
                this->arena_plan_ =
                    TiledArray::detail::make_contraction_arena_plan<
                        result_tile_type, left_tile_type, right_tile_type>(
                        TiledArray::detail::ArenaInnerShapeKind::
                            gemm_result_range,
                        std::make_optional(contrreduce_op.gemm_helper()),
                        inner(this->perm_));
              }
            }
            if constexpr (arena_eligible) {
              if (this->arena_plan_) {
                this->element_nonreturn_op_ =
                    TiledArray::detail::make_fused_contraction_lambda<
                        result_tile_element_type, left_tile_element_type,
                        right_tile_element_type>(contrreduce_op);
              } else {
                this->element_nonreturn_op_ =
                    [contrreduce_op,
                     permute_inner = !this->outer_product_uses_summa()](
                        result_tile_element_type& result,
                        const left_tile_element_type& left,
                        const right_tile_element_type& right) {
                      contrreduce_op(result, left, right);
                      // permutations of result are applied as "postprocessing"
                      if (permute_inner && !TA::empty(result))
                        result = contrreduce_op(result);
                    };
              }
            } else {
              this->element_nonreturn_op_ =
                  [contrreduce_op,
                   permute_inner = !this->outer_product_uses_summa()](
                      result_tile_element_type& result,
                      const left_tile_element_type& left,
                      const right_tile_element_type& right) {
                    contrreduce_op(result, left, right);
                    // permutations of result are applied as "postprocessing"
                    if (permute_inner && !TA::empty(result))
                      result = contrreduce_op(result);
                  };
            }
          }
        }  // ToT x ToT
      } else if (inner_prod == TensorProduct::Hadamard) {
        TA_ASSERT(tot_x_tot);
        if constexpr (tot_x_tot) {
          // inner tile op depends on the outer op ... e.g. if outer op
          // is contract then inner must implement (ternary) multiply-add;
          // if the outer is hadamard then the inner is binary multiply
          const bool outer_uses_summa = this->outer_product_uses_summa();
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
            constexpr bool arena_eligible_h_unit =
                TiledArray::detail::is_contraction_arena_tot_v<
                    result_tile_type, left_tile_type, right_tile_type>;
            if constexpr (arena_eligible_h_unit) {
              if (this->outer_product_uses_summa()) {
                this->arena_plan_ =
                    TiledArray::detail::make_contraction_arena_plan<
                        result_tile_type, left_tile_type, right_tile_type>(
                        TiledArray::detail::ArenaInnerShapeKind::left_range,
                        std::nullopt, inner(this->perm_));
              }
            }
            if constexpr (arena_eligible_h_unit) {
              if (this->arena_plan_) {
                this->element_nonreturn_op_ =
                    TiledArray::detail::make_fused_hadamard_lambda<
                        result_tile_element_type, left_tile_element_type,
                        right_tile_element_type>();
              } else {
                this->element_nonreturn_op_ =
                    [mult_op, outer_uses_summa](
                        result_tile_element_type& result,
                        const left_tile_element_type& left,
                        const right_tile_element_type& right) {
                      if (!outer_uses_summa)
                        result = mult_op(left, right);
                      else {  // outer product evaluated by (batched) SUMMA
                        // there is currently no fused MultAdd ternary Op, only
                        // Add and Mult thus implement this as 2 separate steps
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
            } else {
              this->element_nonreturn_op_ =
                  [mult_op, outer_uses_summa](
                      result_tile_element_type& result,
                      const left_tile_element_type& left,
                      const right_tile_element_type& right) {
                    if (!outer_uses_summa)
                      result = mult_op(left, right);
                    else {  // outer product evaluated by (batched) SUMMA
                      // there is currently no fused MultAdd ternary Op, only
                      // Add and Mult thus implement this as 2 separate steps
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
            constexpr bool arena_eligible_h_scaled =
                TiledArray::detail::is_contraction_arena_tot_v<
                    result_tile_type, left_tile_type, right_tile_type>;
            if constexpr (arena_eligible_h_scaled) {
              if (this->outer_product_uses_summa()) {
                this->arena_plan_ =
                    TiledArray::detail::make_contraction_arena_plan<
                        result_tile_type, left_tile_type, right_tile_type>(
                        TiledArray::detail::ArenaInnerShapeKind::left_range,
                        std::nullopt, inner(this->perm_));
              }
            }
            if constexpr (arena_eligible_h_scaled) {
              if (this->arena_plan_) {
                this->element_nonreturn_op_ =
                    TiledArray::detail::make_fused_hadamard_scaled_lambda<
                        result_tile_element_type, left_tile_element_type,
                        right_tile_element_type>(this->factor_);
              } else {
                this->element_nonreturn_op_ =
                    [mult_op, outer_uses_summa](
                        result_tile_element_type& result,
                        const left_tile_element_type& left,
                        const right_tile_element_type& right) {
                      if (!outer_uses_summa)
                        result = mult_op(left, right);
                      else {
                        // there is currently no fused MultAdd ternary Op, only
                        // Add and Mult thus implement this as 2 separate steps
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
            } else {
              this->element_nonreturn_op_ =
                  [mult_op, outer_uses_summa](
                      result_tile_element_type& result,
                      const left_tile_element_type& left,
                      const right_tile_element_type& right) {
                    if (!outer_uses_summa)
                      result = mult_op(left, right);
                    else {
                      // there is currently no fused MultAdd ternary Op, only
                      // Add and Mult thus implement this as 2 separate steps
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
          constexpr auto kind =
              tot_x_t ? TiledArray::detail::ArenaInnerShapeKind::left_range
                      : TiledArray::detail::ArenaInnerShapeKind::right_range;
          constexpr bool arena_eligible_scale =
              TiledArray::detail::is_contraction_arena_tot_v<
                  result_tile_type, left_tile_type, right_tile_type>;
          if constexpr (arena_eligible_scale) {
            // the fused arena scale ops are factor-free; a non-unit
            // expression-level prefactor (ScalMult) takes the fallback op,
            // which absorbs it
            if (this->outer_product_uses_summa() &&
                this->factor_ == scalar_type(1)) {
              // The inner perm handed to the plan must match how the inner
              // *result* permutation is applied for this result cell type --
              // and the two cell types apply it in different places:
              //
              //   * View (arena) cells: pass an identity inner perm so the
              //     plan is always built (pre-shaping result cells in the
              //     unpermuted operand inner layout) and the perm-free fused
              //     scale op is selected; the inner result perm is applied
              //     downstream by op_'s post-processing permute (carried in
              //     make_total_perm for view cells).
              //
              //   * Owning cells: pass the inner result perm so the plan bails
              //     (nullopt) on a non-identity inner perm, falling back to the
              //     per-cell op that applies the inner perm itself -- matching
              //     the outer-only total_perm make_total_perm carries here.
              //     (A trivial inner perm still lets the plan + fused op run.)
              Permutation plan_inner_perm;
              if constexpr (!TiledArray::is_tensor_view_v<
                                result_tile_element_type>)
                plan_inner_perm = inner(this->perm_);
              this->arena_plan_ =
                  TiledArray::detail::make_contraction_arena_plan<
                      result_tile_type, left_tile_type, right_tile_type>(
                      kind, std::nullopt, plan_inner_perm);
            }
          }
          // Fallback per-element op for the scale inner-product when no
          // arena plan is in play. The Contraction outer product is the
          // fused AXPY `result += (perm ^ tot) * scalar` -- no scaled
          // temporary, so it works uniformly for owning and view inner
          // cells. The Hadamard outer product is an assignment
          // `result = (perm ^ tot) * scalar`, which needs value-returning
          // `scale`; only owning inner cells support it.
          // N.B. the expression-level scalar prefactor (factor_, != 1 for
          // ScalMult expressions) multiplies the plain operand's element
          auto fallback_op =
              [perm = !this->implicit_permute_inner_ ? inner(this->perm_)
                                                     : Permutation{},
               outer_uses_summa = this->outer_product_uses_summa(),
               factor = this->factor_](result_tile_element_type& result,
                                       const left_tile_element_type& left,
                                       const right_tile_element_type& right) {
                if (outer_uses_summa) {
                  using TiledArray::axpy_to;
                  if constexpr (tot_x_t) {
                    if (left.empty()) return;  // absent cell: no contribution
                    if (perm)
                      axpy_to(result, left, right * factor, perm);
                    else
                      axpy_to(result, left, right * factor);
                  } else {
                    if (right.empty()) return;  // absent cell: no contribution
                    if (perm)
                      axpy_to(result, right, left * factor, perm);
                    else
                      axpy_to(result, right, left * factor);
                  }
                } else {
                  if constexpr (!TiledArray::is_tensor_view_v<
                                    result_tile_element_type>) {
                    using TiledArray::scale;
                    if constexpr (tot_x_t)
                      result = perm ? scale(left, right * factor, perm)
                                    : scale(left, right * factor);
                    else
                      result = perm ? scale(right, left * factor, perm)
                                    : scale(right, left * factor);
                  } else {
                    TA_EXCEPTION(
                        "Tensor<View> scale-inner Hadamard-outer product: a "
                        "view result cell cannot be value-assigned a fresh "
                        "scaled tensor");
                  }
                }
              };
          if constexpr (arena_eligible_scale) {
            if (this->arena_plan_) {
              if constexpr (tot_x_t)
                this->element_nonreturn_op_ =
                    TiledArray::detail::make_fused_scale_tot_x_t_lambda<
                        result_tile_element_type, left_tile_element_type,
                        right_tile_element_type>();
              else
                this->element_nonreturn_op_ =
                    TiledArray::detail::make_fused_scale_t_x_tot_lambda<
                        result_tile_element_type, left_tile_element_type,
                        right_tile_element_type>();
            } else {
              this->element_nonreturn_op_ = fallback_op;
            }
          } else {
            this->element_nonreturn_op_ = fallback_op;
          }
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
