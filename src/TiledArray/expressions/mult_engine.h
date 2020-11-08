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
 *  mult_engine.h
 *  Mar 31, 2014
 *
 */

#ifndef TILEDARRAY_EXPRESSIONS_MULT_ENGINE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_MULT_ENGINE_H__INCLUDED

#include <TiledArray/expressions/cont_engine.h>
#include <TiledArray/external/btas.h>
#include <TiledArray/tile_op/binary_wrapper.h>
#include <TiledArray/tile_op/mult.h>
#include <btas/optimize/contract.h>

namespace TiledArray {
namespace expressions {

// Forward declarations
template <typename, typename>
class MultExpr;
template <typename, typename, typename>
class ScalMultExpr;
template <typename, typename, typename>
class MultEngine;
template <typename, typename, typename, typename>
class ScalMultEngine;

template <typename Tile>
inline auto make_tile_contract_op(const IndexList& left_indices,
                                  const IndexList& right_indices,
                                  const IndexList& result_indices) {
  static_assert(TiledArray::detail::is_ta_tensor_v<Tile> ||
                    TiledArray::detail::is_btas_tensor_v<Tile>,
                "only support BTAS and TA tensors as inner tensors of ToT");
  // btas::contract only works for col-major storage
  if constexpr (TiledArray::detail::is_btas_tensor_v<
                    Tile>) {  // can do arbitrary contractions here
                              //    auto indices_to_integers = [&]() {
                              //      // collect all unique indices
                              //      IndexList::container_type all_indices;
    //      all_indices.reserve(left_indices.size() + right_indices.size() +
    //                          result_indices.size());
    //      all_indices.insert(all_indices.end(), left_indices.data().begin(),
    //                         left_indices.data().end());
    //      all_indices.insert(all_indices.end(), right_indices.data().begin(),
    //                         right_indices.data().end());
    //      all_indices.insert(all_indices.end(), result_indices.data().begin(),
    //                         result_indices.data().end());
    //      auto end_of_unique = std::unique(all_indices.begin(),
    //      all_indices.end()); all_indices.erase(end_of_unique,
    //      all_indices.end());
    //
    //      // map each index list to numbers
    //      auto to_integers = [](const container::svector<std::string> indices,
    //                            const container::svector<std::string>
    //                            all_indices) {
    //        TA_ASSERT(!all_indices.empty());
    //        btas::DEFAULT::index<std::int64_t> result;
    //        result.reserve(indices.size());
    //        for (const auto& idx : indices) {
    //          auto pos = std::find(all_indices.begin(), all_indices.end(),
    //          idx) -
    //                     all_indices.begin();
    //          TA_ASSERT(pos < all_indices.size());
    //          result.push_back(pos);
    //        }
    //        return result;
    //      };
    //      auto left_annotation = to_integers(left_indices.data(),
    //      all_indices); auto right_annotation =
    //      to_integers(right_indices.data(), all_indices); auto
    //      result_annotation = to_integers(result_indices.data(), all_indices);
    //      return std::make_tuple(left_annotation, right_annotation,
    //                             result_annotation);
    //    };
    //    auto [left_annotation, right_annotation, result_annotation] =
    //        indices_to_integers();
    //    return [left_annotation = std::move(left_annotation),
    //            right_annotation = std::move(right_annotation),
    //            result_annotation = std::move(result_annotation)](
    //               const Tile& left_tile, const Tile& right_tile) {
    //      Tile result;
    //      btas::contract(1, left_tile, left_annotation, right_tile,
    //                     right_annotation, 0, result, result_annotation);
    //      return result;
    //    };
  }
}

template <typename Left, typename Right, typename Result>
struct EngineTrait<MultEngine<Left, Right, Result>> {
  static_assert(
      std::is_same<typename EngineTrait<Left>::policy,
                   typename EngineTrait<Right>::policy>::value,
      "The left- and right-hand expressions must use the same policy class");

  // Argument typedefs
  typedef Left left_type;    ///< The left-hand expression type
  typedef Right right_type;  ///< The right-hand expression type

  // Operational typedefs
  typedef TiledArray::detail::Mult<
      Result, typename EngineTrait<Left>::eval_type,
      typename EngineTrait<Right>::eval_type, EngineTrait<Left>::consumable,
      EngineTrait<Right>::consumable>
      op_base_type;  ///< The base tile operation type
  typedef TiledArray::detail::BinaryWrapper<op_base_type>
      op_type;  ///< The tile operation type
  typedef typename op_type::result_type value_type;  ///< The result tile type
  typedef typename eval_trait<value_type>::type
      eval_type;  ///< Evaluation tile type
  typedef typename TiledArray::detail::numeric_type<value_type>::type
      scalar_type;                       ///< Tile scalar type
  typedef typename Left::policy policy;  ///< The result policy type
  typedef TiledArray::detail::DistEval<value_type, policy>
      dist_eval_type;  ///< The distributed evaluator type

  // Meta data typedefs
  typedef typename policy::ordinal_type size_type;   ///< Size type
  typedef typename policy::trange_type trange_type;  ///< Tiled range type
  typedef typename policy::shape_type shape_type;    ///< Shape type
  typedef typename policy::pmap_interface
      pmap_interface;  ///< Process map interface type

  static constexpr bool consumable = is_consumable_tile<eval_type>::value;
  static constexpr unsigned int leaves =
      EngineTrait<Left>::leaves + EngineTrait<Right>::leaves;
};

template <typename Left, typename Right, typename Scalar, typename Result>
struct EngineTrait<ScalMultEngine<Left, Right, Scalar, Result>> {
  static_assert(
      std::is_same<typename EngineTrait<Left>::policy,
                   typename EngineTrait<Right>::policy>::value,
      "The left- and right-hand expressions must use the same policy class");

  // Argument typedefs
  typedef Left left_type;    ///< The left-hand expression type
  typedef Right right_type;  ///< The right-hand expression type

  // Operational typedefs
  typedef Scalar scalar_type;  ///< Tile scalar type
  typedef TiledArray::detail::ScalMult<
      Result, typename EngineTrait<Left>::eval_type,
      typename EngineTrait<Right>::eval_type, scalar_type,
      EngineTrait<Left>::consumable, EngineTrait<Right>::consumable>
      op_base_type;  ///< The base tile operation type
  typedef TiledArray::detail::BinaryWrapper<op_base_type>
      op_type;  ///< The tile operation type
  typedef typename op_type::result_type value_type;  ///< The result tile type
  typedef typename eval_trait<value_type>::type
      eval_type;                         ///< Evaluation tile type
  typedef typename Left::policy policy;  ///< The result policy type
  typedef TiledArray::detail::DistEval<value_type, policy>
      dist_eval_type;  ///< The distributed evaluator type

  // Meta data typedefs
  typedef typename policy::ordinal_type size_type;   ///< Size type
  typedef typename policy::trange_type trange_type;  ///< Tiled range type
  typedef typename policy::shape_type shape_type;    ///< Shape type
  typedef typename policy::pmap_interface
      pmap_interface;  ///< Process map interface type

  static constexpr bool consumable = is_consumable_tile<eval_type>::value;
  static constexpr unsigned int leaves =
      EngineTrait<Left>::leaves + EngineTrait<Right>::leaves;
};

/// Multiplication expression engine

/// This implements any expression encoded with the multiplication operator.
/// This includes Hadamard product, e.g. \code (c("i,j")=)a("i,j")*b("i,j")
/// \endcode , and pure contractions, e.g. \code (c("i,j")=)a("i,k")*b("k,j")
/// \endcode . \internal mixed Hadamard-contraction case, e.g. \code
/// c("i,j,l")=a("i,l,k")*b("j,l,k") \endcode , is not supported since
///   this requires that the result labels are assigned by user (currently they
///   are computed by this engine)
/// \tparam Left The left-hand engine type
/// \tparam Right The right-hand engine type
/// \tparam Result The result tile type
template <typename Left, typename Right, typename Result>
class MultEngine : public ContEngine<MultEngine<Left, Right, Result>> {
 public:
  // Class hierarchy typedefs
  typedef MultEngine<Left, Right, Result> MultEngine_;  ///< This class type
  typedef ContEngine<MultEngine_>
      ContEngine_;  ///< Contraction engine base class
  typedef BinaryEngine<MultEngine_> BinaryEngine_;  ///< Binary base class type
  typedef BinaryEngine<MultEngine_>
      ExprEngine_;  ///< Expression engine base class type

  // Argument typedefs
  typedef typename EngineTrait<MultEngine_>::left_type
      left_type;  ///< The left-hand expression type
  typedef typename EngineTrait<MultEngine_>::right_type
      right_type;  ///< The right-hand expression type

  // Operational typedefs
  typedef typename EngineTrait<MultEngine_>::value_type
      value_type;  ///< The result tile type
  typedef typename EngineTrait<MultEngine_>::op_base_type
      op_base_type;  ///< The tile operation type
  typedef typename EngineTrait<MultEngine_>::op_type
      op_type;  ///< The tile operation type
  typedef typename EngineTrait<MultEngine_>::policy
      policy;  ///< The result policy type
  typedef typename EngineTrait<MultEngine_>::dist_eval_type
      dist_eval_type;  ///< The distributed evaluator type
  typedef typename EngineTrait<MultEngine_>::scalar_type
      scalar_type;  ///< Tile scalar type

  // Meta data typedefs
  typedef
      typename EngineTrait<MultEngine_>::size_type size_type;  ///< Size type
  typedef typename EngineTrait<MultEngine_>::trange_type
      trange_type;  ///< Tiled range type
  typedef
      typename EngineTrait<MultEngine_>::shape_type shape_type;  ///< Shape type
  typedef typename EngineTrait<MultEngine_>::pmap_interface
      pmap_interface;  ///< Process map interface type

 private:
  TensorProduct product_type_ = TensorProduct::Invalid;
  TensorProduct inner_product_type_ = TensorProduct::Invalid;

 public:
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
    /// only Hadamard and contraction are supported now
    TA_ASSERT(inner_product_type_ == TensorProduct::Hadamard ||
              inner_product_type_ == TensorProduct::Contraction);
    return inner_product_type_;
  }

 public:
  /// Constructor

  /// \tparam L The left-hand argument expression type
  /// \tparam R The right-hand argument expression type
  /// \param expr The parent expression
  template <typename L, typename R>
  MultEngine(const MultExpr<L, R>& expr) : ContEngine_(expr) {}

  /// Set the index list for this expression

  /// This function will set the index list for this expression and its
  /// children such that the number of permutations is minimized. The final
  /// index list may not be set to target, which indicates that the
  /// result of this expression will be permuted to match \c target_indices.
  /// \param target_indices The target index list for this expression
  void perm_indices(const BipartiteIndexList& target_indices) {
    if (product_type() == TensorProduct::Contraction)
      ContEngine_::perm_indices(target_indices);
    else {
      BinaryEngine_::perm_indices(target_indices);
    }
  }

  /// Initialize the index list of this expression

  /// \param target_indices The target index list for this expression
  void init_indices(const BipartiteIndexList& target_indices) {
    // to decide what type of product this is must initialize indices down
    // the tree.
    // N.B. since this may be a contraction we do not know the target indices
    // for the left and right, hence do target-neutral initialization
    BinaryEngine_::left_.init_indices();
    BinaryEngine_::right_.init_indices();
    product_type_ = compute_product_type(outer(BinaryEngine_::left_.indices()),
                                         outer(BinaryEngine_::right_.indices()),
                                         outer(target_indices));
    inner_product_type_ = compute_product_type(
        inner(BinaryEngine_::left_.indices()),
        inner(BinaryEngine_::right_.indices()), inner(target_indices));

    // TODO support general products that involve fused, contracted, and free
    // indices Example: in ijk * jkl -> ijl indices i and l are free, index k is
    // contracted, and index j is fused
    // N.B. Currently only 2 types of products are supported:
    // - Hadamard product (in which all indices are fused), and,
    // - pure contraction (>=1 contracted, 0 fused, >=1 free indices)
    // For the ToT arguments only the Hadamard product is supported

    // Check the *outer* indices to determine whether the arguments are
    // - contracted, or
    // - Hadamard-multiplied
    // The latter is indicated by the equality (modulo permutation) of
    // the outer left and right arg indices to the target indices.
    // Only the outer indices matter here since the inner indices only encode
    // the tile op; the type of the tile op does not need to match the type of
    // the operation on the outer indices
    if (product_type() == TensorProduct::Hadamard) {
      // assumes inner op is also Hadamard
      BinaryEngine_::perm_indices(target_indices);
    } else {
      auto children_initialized = true;
      ContEngine_::init_indices(children_initialized);
      ContEngine_::perm_indices(target_indices);
    }
  }

  /// Initialize the index list of this expression
  void init_indices() {
    // to decide what type of product this is must initialize indices down the
    // tree
    BinaryEngine_::left_.init_indices();
    BinaryEngine_::right_.init_indices();
    auto children_initialized = true;
    product_type_ =
        compute_product_type(outer(BinaryEngine_::left_.indices()),
                             outer(BinaryEngine_::right_.indices()));
    inner_product_type_ =
        compute_product_type(inner(BinaryEngine_::left_.indices()),
                             inner(BinaryEngine_::right_.indices()));

    if (product_type() == TensorProduct::Hadamard) {
      BinaryEngine_::init_indices(children_initialized);
    } else {
      ContEngine_::init_indices(children_initialized);
    }
  }

  /// Initialize result tensor structure

  /// This function will initialize the permutation, tiled range, and shape
  /// for the result tensor.
  /// \param target_indices The target index list for the result tensor
  void init_struct(const BipartiteIndexList& target_indices) {
    if (product_type() == TensorProduct::Contraction)
      ContEngine_::init_struct(target_indices);
    else
      BinaryEngine_::init_struct(target_indices);

    if constexpr (TiledArray::detail::is_tensor_of_tensor_v<value_type>) {
      const auto inner_prod = inner_product_type();
      if (inner_prod == TensorProduct::Contraction) {
        using inner_tile_type = typename value_type::value_type;
        using contract_inner_tile_type =
            TiledArray::detail::ContractReduce<inner_tile_type, inner_tile_type,
                                               inner_tile_type, scalar_type>;
        auto contrreduce_op =
            (inner(target_indices) != inner(this->indices_))
                ? contract_inner_tile_type(
                      to_cblas_op(this->left_inner_permtype_),
                      to_cblas_op(this->right_inner_permtype_), this->factor_,
                      inner_size(this->indices_),
                      inner_size(this->left_indices_),
                      inner_size(this->right_indices_),
                      (this->permute_tiles_ ? inner(this->perm_)
                                            : Permutation{}))
                : contract_inner_tile_type(
                      to_cblas_op(this->left_inner_permtype_),
                      to_cblas_op(this->right_inner_permtype_), this->factor_,
                      inner_size(this->indices_),
                      inner_size(this->left_indices_),
                      inner_size(this->right_indices_));
        this->inner_tile_op_ = [contrreduce_op](const inner_tile_type& left,
                                                const inner_tile_type& right) {
          inner_tile_type result;
          contrreduce_op(result, left, right);
          return result;
        };
      }
    }
  }

  /// Initialize result tensor distribution

  /// This function will initialize the world and process map for the result
  /// tensor.
  /// \param world The world were the result will be distributed
  /// \param pmap The process map for the result tensor tiles
  void init_distribution(World* world, std::shared_ptr<pmap_interface> pmap) {
    if (product_type() == TensorProduct::Contraction)
      ContEngine_::init_distribution(world, pmap);
    else
      BinaryEngine_::init_distribution(world, pmap);
  }

  /// Non-permuting tiled range factory function

  /// \return The result tiled range object
  trange_type make_trange() const {
    if (product_type() == TensorProduct::Contraction)
      return ContEngine_::make_trange();
    else
      return BinaryEngine_::make_trange();
  }

  /// Permuting tiled range factory function

  /// \param perm The permutation to be applied to the array
  /// \return The result tiled range object
  trange_type make_trange(const Permutation& perm) const {
    if (product_type() == TensorProduct::Contraction)
      return ContEngine_::make_trange(perm);
    else
      return BinaryEngine_::make_trange(perm);
  }

  /// Non-permuting shape factory function

  /// \return The result shape
  shape_type make_shape() const {
    return BinaryEngine_::left_.shape().mult(BinaryEngine_::right_.shape());
  }

  /// Permuting shape factory function

  /// \param perm The permutation to be applied to the array
  /// \return The result shape
  shape_type make_shape(const Permutation& perm) const {
    return BinaryEngine_::left_.shape().mult(BinaryEngine_::right_.shape(),
                                             outer(perm));
  }

  /// Non-permuting tile operation factory function

  /// \return The tile operation
  op_type make_tile_op() const {
    if constexpr (TiledArray::detail::is_tensor_of_tensor_v<
                      value_type>) {  // nested tensors
      const auto inner_prod = inner_product_type();
      if (inner_prod == TensorProduct::Invalid ||
          inner_prod == TensorProduct::Hadamard) {
        return op_type(op_base_type());
      } else if (inner_prod == TensorProduct::Contraction) {
        return op_type(op_base_type(this->inner_tile_op_));
      } else
        abort();
    } else {  // plain tensors
      return op_type(op_base_type());
    }
  }

  /// Permuting tile operation factory function

  /// \param perm The permutation to be applied to the result
  /// \return The tile operation
  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  op_type make_tile_op(const Perm& perm) const {
    if constexpr (TiledArray::detail::is_tensor_of_tensor_v<
                      value_type>) {  // nested tensors
      const auto inner_prod = inner_product_type();
      if (inner_prod == TensorProduct::Invalid ||
          inner_prod == TensorProduct::Hadamard) {
        return op_type(op_base_type(), perm);
      } else if (inner_prod == TensorProduct::Contraction) {
        return op_type(op_base_type(this->inner_tile_op_), perm);
      } else
        abort();
    } else {  // plain tensor
      return op_type(op_base_type(), perm);
    }
  }

  /// Construct the distributed evaluator for this expression

  /// \return The distributed evaluator that will evaluate this expression
  dist_eval_type make_dist_eval() const {
    if (product_type() == TensorProduct::Contraction)
      return ContEngine_::make_dist_eval();
    else
      return BinaryEngine_::make_dist_eval();
  }

  /// Expression identification tag

  /// \return An expression tag used to identify this expression
  const char* make_tag() const { return "[*] "; }

  /// Expression print

  /// \param os The output stream
  /// \param target_indices The target index list for this expression
  void print(ExprOStream os, const BipartiteIndexList& target_indices) const {
    if (product_type() == TensorProduct::Contraction)
      return ContEngine_::print(os, target_indices);
    else
      return BinaryEngine_::print(os, target_indices);
  }
};  // class MultEngine

/// Scaled multiplication expression engine

/// Similar to MultEngine but implements the result of MultEngine scaled by a
/// constant. \tparam Left The left-hand engine type \tparam Right The
/// Right-hand engine type \tparam Scalar The scaling factor type \tparam Result
/// The result tile type
template <typename Left, typename Right, typename Scalar, typename Result>
class ScalMultEngine
    : public ContEngine<ScalMultEngine<Left, Right, Scalar, Result>> {
 public:
  // Class hierarchy typedefs
  typedef ScalMultEngine<Left, Right, Scalar, Result>
      ScalMultEngine_;  ///< This class type
  typedef ContEngine<ScalMultEngine_>
      ContEngine_;  ///< Contraction engine base class
  typedef BinaryEngine<ScalMultEngine_>
      BinaryEngine_;  ///< Binary base class type
  typedef BinaryEngine<ScalMultEngine_>
      ExprEngine_;  ///< Expression engine base class type

  // Argument typedefs
  typedef typename EngineTrait<ScalMultEngine_>::left_type
      left_type;  ///< The left-hand expression type
  typedef typename EngineTrait<ScalMultEngine_>::right_type
      right_type;  ///< The right-hand expression type

  // Operational typedefs
  typedef typename EngineTrait<ScalMultEngine_>::value_type
      value_type;  ///< The result tile type
  typedef typename EngineTrait<ScalMultEngine_>::scalar_type
      scalar_type;  ///< Tile scalar type
  typedef typename EngineTrait<ScalMultEngine_>::op_base_type
      op_base_type;  ///< The tile operation type
  typedef typename EngineTrait<ScalMultEngine_>::op_type
      op_type;  ///< The tile operation type
  typedef typename EngineTrait<ScalMultEngine_>::policy
      policy;  ///< The result policy type
  typedef typename EngineTrait<ScalMultEngine_>::dist_eval_type
      dist_eval_type;  ///< The distributed evaluator type

  // Meta data typedefs
  typedef typename EngineTrait<ScalMultEngine_>::size_type
      size_type;  ///< Size type
  typedef typename EngineTrait<ScalMultEngine_>::trange_type
      trange_type;  ///< Tiled range type
  typedef typename EngineTrait<ScalMultEngine_>::shape_type
      shape_type;  ///< Shape type
  typedef typename EngineTrait<ScalMultEngine_>::pmap_interface
      pmap_interface;  ///< Process map interface type

 private:
  TensorProduct product_type_ = TensorProduct::Invalid;
  TensorProduct inner_product_type_ = TensorProduct::Invalid;

 public:
  /// \return the product type
  TensorProduct product_type() const {
    TA_ASSERT(product_type_ != TensorProduct::Invalid);
    /// only Hadamard and contraction are supported now
    TA_ASSERT(product_type_ == TensorProduct::Hadamard ||
              product_type_ == TensorProduct::Contraction);
    return product_type_;
  }

  /// \return the inner product type
  TensorProduct inner_product_type() const {
    TA_ASSERT(inner_product_type_ != TensorProduct::Invalid);
    /// only Hadamard and contraction are supported now
    TA_ASSERT(inner_product_type_ == TensorProduct::Hadamard ||
              inner_product_type_ == TensorProduct::Contraction);
    return inner_product_type_;
  }

 public:
  /// Constructor

  /// \tparam L The left-hand argument expression type
  /// \tparam R The right-hand argument expression type
  /// \tparam S The expression scalar type
  /// \param expr The parent expression
  template <typename L, typename R, typename S>
  ScalMultEngine(const ScalMultExpr<L, R, S>& expr) : ContEngine_(expr) {}

  /// Set the index list for this expression

  /// This function will set the index list for this expression and its
  /// children such that the number of permutations is minimized. The final
  /// index list may not be set to target, which indicates that the
  /// result of this expression will be permuted to match \c target_indices.
  /// \param target_indices The target index list for this expression
  void perm_indices(const BipartiteIndexList& target_indices) {
    if (product_type() == TensorProduct::Contraction)
      ContEngine_::perm_indices(target_indices);
    else {
      BinaryEngine_::perm_indices(target_indices);
    }
  }

  /// Initialize the index list of this expression

  /// \param target_indices The target index list for this expression
  void init_indices(const BipartiteIndexList& target_indices) {
    BinaryEngine_::left_.init_indices();
    BinaryEngine_::right_.init_indices();
    product_type_ = compute_product_type(outer(BinaryEngine_::left_.indices()),
                                         outer(BinaryEngine_::right_.indices()),
                                         outer(target_indices));

    if (product_type() == TensorProduct::Hadamard) {
      // since already initialized left and right arg indices assign the target
      // indices
      BinaryEngine_::perm_indices(target_indices);
    } else {
      ContEngine_::init_indices(target_indices);
    }
  }

  /// Initialize the index list of this expression
  void init_indices() {
    BinaryEngine_::left_.init_indices();
    BinaryEngine_::right_.init_indices();
    product_type_ =
        compute_product_type(outer(BinaryEngine_::left_.indices()),
                             outer(BinaryEngine_::right_.indices()));
    if (product_type() == TensorProduct::Hadamard) {
      auto outer_indices = outer((left_type::leaves <= right_type::leaves)
                                     ? BinaryEngine_::left_.indices()
                                     : BinaryEngine_::right_.indices());
      // assume inner op is also Hadamard
      // TODO compute inner indices using inner_product_type_
      auto inner_indices = inner((left_type::leaves <= right_type::leaves)
                                     ? BinaryEngine_::left_.indices()
                                     : BinaryEngine_::right_.indices());
      ExprEngine_::indices_ = BipartiteIndexList(outer_indices, inner_indices);
    } else {
      ContEngine_::init_indices();
    }
  }

  /// Initialize result tensor structure

  /// This function will initialize the permutation, tiled range, and shape
  /// for the result tensor.
  /// \param target_indices The target index list for the result tensor
  void init_struct(const BipartiteIndexList& target_indices) {
    if (product_type() == TensorProduct::Contraction)
      ContEngine_::init_struct(target_indices);
    else
      BinaryEngine_::init_struct(target_indices);
  }

  /// Initialize result tensor distribution

  /// This function will initialize the world and process map for the result
  /// tensor.
  /// \param world The world were the result will be distributed
  /// \param pmap The process map for the result tensor tiles
  void init_distribution(World* world, std::shared_ptr<pmap_interface> pmap) {
    if (product_type() == TensorProduct::Contraction)
      ContEngine_::init_distribution(world, pmap);
    else
      BinaryEngine_::init_distribution(world, pmap);
  }

  /// Construct the distributed evaluator for this expression

  /// \return The distributed evaluator that will evaluate this expression
  dist_eval_type make_dist_eval() const {
    if (product_type() == TensorProduct::Contraction)
      return ContEngine_::make_dist_eval();
    else
      return BinaryEngine_::make_dist_eval();
  }

  /// Non-permuting tiled range factory function

  /// \return The result tiled range object
  trange_type make_trange() const {
    if (product_type() == TensorProduct::Contraction)
      return ContEngine_::make_trange();
    else
      return BinaryEngine_::make_trange();
  }

  /// Permuting tiled range factory function

  /// \param perm The permutation to be applied to the array
  /// \return The result tiled range object
  trange_type make_trange(const Permutation& perm) const {
    if (product_type() == TensorProduct::Contraction)
      return ContEngine_::make_trange(perm);
    else
      return BinaryEngine_::make_trange(perm);
  }

  /// Non-permuting shape factory function

  /// \return The result shape
  shape_type make_shape() const {
    return BinaryEngine_::left_.shape().mult(BinaryEngine_::right_.shape(),
                                             ContEngine_::factor_);
  }

  /// Permuting shape factory function

  /// \param perm The permutation to be applied to the array
  /// \return The result shape
  shape_type make_shape(const Permutation& perm) const {
    return BinaryEngine_::left_.shape().mult(BinaryEngine_::right_.shape(),
                                             ContEngine_::factor_, perm);
  }

  /// Non-permuting tile operation factory function

  /// \return The tile operation
  op_type make_tile_op() const {
    return op_type(op_base_type(ContEngine_::factor_));
  }

  /// Permuting tile operation factory function

  /// \param perm The permutation to be applied to tiles
  /// \return The tile operation
  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  op_type make_tile_op(const Perm& perm) const {
    return op_type(op_base_type(ContEngine_::factor_), perm);
  }

  /// Expression identification tag

  /// \return An expression tag used to identify this expression
  std::string make_tag() const {
    std::stringstream ss;
    ss << "[*] [" << ContEngine_::factor_ << "] ";
    return ss.str();
  }

  /// Expression print

  /// \param os The output stream
  /// \param target_indices The target index list for this expression
  void print(ExprOStream os, const BipartiteIndexList& target_indices) const {
    if (product_type() == TensorProduct::Contraction)
      return ContEngine_::print(os, target_indices);
    else
      return BinaryEngine_::print(os, target_indices);
  }

};  // class ScalMultEngine

}  // namespace expressions
}  // namespace TiledArray

#endif  // TILEDARRAY_EXPRESSIONS_MULT_ENGINE_H__INCLUDED
