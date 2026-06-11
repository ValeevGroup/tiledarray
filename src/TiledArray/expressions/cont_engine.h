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
#include <TiledArray/pmap/slabbed_pmap.h>
#include <TiledArray/proc_grid.h>
#include <TiledArray/tensor/arena_einsum.h>
#include <TiledArray/tensor/utility.h>
#include <TiledArray/tile_op/batched_contract_reduce.h>
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
  size_type n_slabs_ = 1;           ///< # of fused-index tile slabs

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

    // initialize op_, trange_, and shape_ which only refer to the outer modes
    if (outer(target_indices) != outer(indices_)) {
      const auto outer_perm = outer(perm_);
      // Initialize permuted structure
      if constexpr (!TiledArray::detail::is_tensor_of_tensor_v<value_type>) {
        op_ = op_type(
            left_op, right_op, factor_, outer_size(indices_),
            outer_size(left_indices_), outer_size(right_indices_),
            (!implicit_permute_outer_ ? std::move(outer_perm) : Permutation{}));
      } else {
        auto make_total_perm = [this]() -> BipartitePermutation {
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
      trange_ = ContEngine_::make_trange(outer_perm);
      shape_ = ContEngine_::make_shape(outer_perm);
    } else {
      // Initialize non-permuted structure

      if constexpr (!TiledArray::detail::is_tensor_of_tensor_v<value_type>) {
        op_ = op_type(left_op, right_op, factor_, outer_size(indices_),
                      outer_size(left_indices_), outer_size(right_indices_));
      } else {
        auto make_total_perm = [this]() -> BipartitePermutation {
          if (this->product_type() != TensorProduct::Contraction ||
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

      // Construct the process grid.
      proc_grid_ = TiledArray::detail::ProcGrid(*world, M, N, m, n);

      // Initialize children
      left_.init_distribution(world, proc_grid_.make_row_phase_pmap(K_));
      right_.init_distribution(world, proc_grid_.make_col_phase_pmap(K_));

      // Initialize the process map if not already defined
      if (!pmap) pmap = proc_grid_.make_pmap();
      ExprEngine_::init_distribution(world, pmap);
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

    std::shared_ptr<impl_type> pimpl =
        std::make_shared<impl_type>(left, right, *world_, trange_, shape_,
                                    pmap_, perm_, op_, K_, proc_grid_);

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

    // initialize perm_; an interleaved target (a result permutation that
    // mixes fused and free modes) is not yet supported -- the canonical
    // result layout must equal the target
    this->init_perm(target_indices);
    if (outer(target_indices) != outer(indices_))
      TA_EXCEPTION(
          "general products (fused + contracted + free indices): targets "
          "that interleave fused and free indices are not yet supported; "
          "reorder the result annotation to (fused..., left-free..., "
          "right-free...)");

    // the tile op operates on the folded (fused-mode-free) shapes
    const auto left_op = to_cblas_op(left_outer_permtype_);
    const auto right_op = to_cblas_op(right_outer_permtype_);
    if constexpr (!TiledArray::detail::is_tensor_of_tensor_v<value_type>) {
      op_ = op_type(left_op, right_op, factor_, outer_size(indices_) - nh,
                    outer_size(left_indices_) - nh,
                    outer_size(right_indices_) - nh);
    } else {
      // the batched tile op must be perm-free (BatchedContractReduce cannot
      // host the folded-rank result permutation); the outer perm is empty by
      // the interleaved-target gate above, so only an explicit inner result
      // permutation can require one
      if (!implicit_permute_inner_ && bool(inner(perm_)))
        TA_EXCEPTION(
            "general products of tensors-of-tensors: a non-identity inner "
            "result permutation is not yet supported; reorder the inner "
            "annotation of the result");

      // factor_ is absorbed into element_nonreturn_op_
      op_ = op_type(left_op, right_op, scalar_type(1),
                    outer_size(indices_) - nh, outer_size(left_indices_) - nh,
                    outer_size(right_indices_) - nh, BipartitePermutation{},
                    this->element_nonreturn_op_, std::move(this->arena_plan_));
      // ce+e, ce+ce_right and ce+ce_left are mutually exclusive; at most one
      // is non-null and only one install fires (see init_struct)
      if (this->arena_strided_dgemm_ce_e_tile_op_)
        op_.set_strided_oprod_op(this->arena_strided_dgemm_ce_e_tile_op_);
      if (this->arena_strided_dgemm_ce_ce_right_tile_op_)
        op_.set_strided_oprod_op(
            this->arena_strided_dgemm_ce_ce_right_tile_op_);
      if (this->arena_strided_dgemm_ce_ce_left_tile_op_)
        op_.set_strided_oprod_op(this->arena_strided_dgemm_ce_ce_left_tile_op_);
      // Plan ownership transferred to op_; mark carrier slot empty so any
      // later use of arena_plan_ reads as "no plan" rather than moved-from.
      if constexpr (!std::is_same_v<arena_plan_storage_t, std::monostate>) {
        this->arena_plan_.reset();
      }
    }

    trange_ = make_trange_general();
    shape_ = make_shape_general();

    if (ExprEngine_::override_ptr_ && ExprEngine_::override_ptr_->shape) {
      shape_ = shape_.mask(*ExprEngine_::override_ptr_->shape);
    }
  }

  /// Tiled range factory function for a general product

  /// \return The result tiled range: the fused mode ranges followed by the
  /// left- and right-external mode ranges
  trange_type make_trange_general() const {
    const unsigned int nh = n_fused_modes_;
    const unsigned int nc = op_.gemm_helper().num_contract_ranks();
    const unsigned int neA = op_.gemm_helper().left_rank() - nc;
    const unsigned int neB = op_.gemm_helper().right_rank() - nc;

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
    const unsigned int neA = op_.gemm_helper().left_rank() - nc;
    const unsigned int neB = op_.gemm_helper().right_rank() - nc;

    // Get pointers to the argument sizes
    const auto* MADNESS_RESTRICT const left_tiles_size =
        left_.trange().tiles_range().extent_data();
    const auto* MADNESS_RESTRICT const left_element_size =
        left_.trange().elements_range().extent_data();
    const auto* MADNESS_RESTRICT const right_tiles_size =
        right_.trange().tiles_range().extent_data();
    const auto* MADNESS_RESTRICT const right_element_size =
        right_.trange().elements_range().extent_data();

    // Compute the slab count and the fused sizes of the per-slab contraction
    size_type M = 1ul, m = 1ul, N = 1ul, n = 1ul;
    n_slabs_ = 1ul;
    for (unsigned int i = 0u; i < nh; ++i) n_slabs_ *= left_tiles_size[i];
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
    // construction alltogether
    if (M == 0 || N == 0 || n_slabs_ == 0) {
      left_.init_distribution(world, {});
      right_.init_distribution(world, {});
      ExprEngine_::init_distribution(
          world,
          (pmap ? pmap : policy::default_pmap(*world, n_slabs_ * M * N)));
    } else {
      // Construct the per-slab process grid.
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
  }

  /// Construct the distributed evaluator of a general product

  /// \return The batched-Summa distributed evaluator for this expression
  dist_eval_type make_dist_eval_general() const {
    typedef TiledArray::detail::BatchedContractReduce<op_type> batched_op_type;
    typedef TiledArray::detail::Summa<typename left_type::dist_eval_type,
                                      typename right_type::dist_eval_type,
                                      batched_op_type, typename Derived::policy>
        impl_type;

    typename left_type::dist_eval_type left = left_.make_dist_eval();
    typename right_type::dist_eval_type right = right_.make_dist_eval();

    std::shared_ptr<impl_type> pimpl = std::make_shared<impl_type>(
        left, right, *world_, trange_, shape_, pmap_, perm_,
        batched_op_type(op_, n_fused_modes_), K_, proc_grid_, n_slabs_);

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
                      !bool(inner(this->perm_))) {
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
                      !bool(inner(this->perm_));
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
                  const bool no_result_inner_perm = !bool(inner(this->perm_));
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
            if (this->outer_product_uses_summa()) {
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
          auto fallback_op = [perm = !this->implicit_permute_inner_
                                         ? inner(this->perm_)
                                         : Permutation{},
                              outer_uses_summa =
                                  this->outer_product_uses_summa()](
                                 result_tile_element_type& result,
                                 const left_tile_element_type& left,
                                 const right_tile_element_type& right) {
            if (outer_uses_summa) {
              using TiledArray::axpy_to;
              if constexpr (tot_x_t) {
                if (left.empty()) return;  // absent cell: no contribution
                if (perm)
                  axpy_to(result, left, right, perm);
                else
                  axpy_to(result, left, right);
              } else {
                if (right.empty()) return;  // absent cell: no contribution
                if (perm)
                  axpy_to(result, right, left, perm);
                else
                  axpy_to(result, right, left);
              }
            } else {
              if constexpr (!TiledArray::is_tensor_view_v<
                                result_tile_element_type>) {
                using TiledArray::scale;
                if constexpr (tot_x_t)
                  result = perm ? scale(left, right, perm) : scale(left, right);
                else
                  result = perm ? scale(right, left, perm) : scale(right, left);
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
