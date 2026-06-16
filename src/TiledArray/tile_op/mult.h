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
 *  mult.h
 *  May 8, 2013
 *
 */

#ifndef TILEDARRAY_TILE_OP_MULT_H__INCLUDED
#define TILEDARRAY_TILE_OP_MULT_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/tensor/type_traits.h>
#include <TiledArray/tile_op/tile_interface.h>
#include <TiledArray/util/function.h>
#include <TiledArray/zero_tensor.h>

namespace TiledArray {
namespace detail {

/// Tile multiplication operation

/// This class implements element-wise multiplication of two tiles,
/// optionally followed by a permutation of the result, that can
/// be customized to arbitrary binary operation types. Thus this is
/// essentially the binary `std::transform` with `std::multiplies` as the
/// default binary op; the binary op is lowered automatically when applied to
/// nested tensors.
///
/// \tparam Result The result tile type
/// \tparam Left The left-hand argument type
/// \tparam Right The right-hand argument type
/// \tparam LeftConsumable If `true`, the left-hand tile is a temporary and
/// may be consumed
/// \tparam RightConsumable If `true`, the right-hand tile is a temporary
/// and may be consumed
/// \note Input tiles can be consumed only if their type matches the result
/// type.
template <typename Result, typename Left, typename Right, bool LeftConsumable,
          bool RightConsumable>
class Mult {
 public:
  typedef Mult<Result, Left, Right, LeftConsumable, RightConsumable> Mult_;
  typedef Left left_type;      ///< Left-hand argument base type
  typedef Right right_type;    ///< Right-hand argument base type
  typedef Result result_type;  ///< The result tile type

  using left_value_type = typename left_type::value_type;
  using right_value_type = typename right_type::value_type;
  using result_value_type = typename result_type::value_type;
  using element_op_type = result_value_type(const left_value_type&,
                                            const right_value_type&);

  /// Indicates whether it is *possible* to consume the left tile
  static constexpr bool left_is_consumable =
      LeftConsumable && std::is_same<result_type, left_type>::value;
  /// Indicates whether it is *possible* to consume the right tile
  static constexpr bool right_is_consumable =
      RightConsumable && std::is_same<result_type, right_type>::value;

 private:
  /// type-erased reference to custom element_op
  /// \note the lifetime is managed by the callee!
  TiledArray::function_ref<element_op_type> element_op_;

  /// True when this Mult's result has view inner cells (e.g. ArenaTensor),
  /// the only case in which tile_op_ is ever populated. Gates instantiation
  /// of eval_tile_op so non-view result types (which need not provide a
  /// `permute` member) are unaffected.
  static constexpr bool uses_tile_op_ =
      TiledArray::is_tensor_view_v<result_value_type>;

  /// True when the plain element-wise `mult` fallback (used when no custom
  /// element_op_ is supplied) is type-compatible with the result tile type.
  /// This is false only for the dot_inner denest: both operands are nested
  /// (ToT) but the result is a plain (non-nested) tensor of scalars, so
  /// `mult(left, right)` would yield a ToT, not the scalar-element result. In
  /// that case a custom element_op_ is always supplied, so the fallback is
  /// unreachable; the `if constexpr` guards below keep it from being
  /// instantiated. Note this stays true for the mixed Hadamard product
  /// ToT * T -> ToT (nested * plain -> nested), where the plain fallback is
  /// the correct path.
  static constexpr bool plain_mult_ok_ =
      !(TiledArray::detail::is_nested_tensor_v<left_value_type> &&
        TiledArray::detail::is_nested_tensor_v<right_value_type> &&
        !TiledArray::detail::is_nested_tensor_v<result_value_type>);

  /// type-erased reference to a whole-tile op. When set, eval() delegates the
  /// entire tile product to it. Used for arena tensor-of-tensors products
  /// whose per-cell op cannot value-return (e.g. ArenaTensor view inner
  /// cells), so the result tile must be shaped and filled as a unit.
  /// \note the lifetime is managed by the callee!
  TiledArray::function_ref<result_type(const left_type&, const right_type&)>
      tile_op_;

  /// Delegates the whole tile product to tile_op_.
  result_type eval_tile_op(const left_type& first,
                           const right_type& second) const {
    return tile_op_(first, second);
  }

  /// Delegates the whole tile product to tile_op_, then permutes the result.
  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type eval_tile_op(const left_type& first, const right_type& second,
                           const Perm& perm) const {
    result_type result = tile_op_(first, second);
    if (perm) result = result.permute(perm);
    return result;
  }

  // Permuting tile evaluation function
  // These operations cannot consume the argument tile since this operation
  // requires temporary storage space.
  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type eval(const left_type& first, const right_type& second,
                   const Perm& perm) const {
    if constexpr (uses_tile_op_) {
      if (tile_op_) return eval_tile_op(first, second, perm);
    }
    if constexpr (plain_mult_ok_) {
      if (!element_op_) {
        using TiledArray::mult;
        return mult(first, second, perm);
      }
    } else {
      TA_ASSERT(element_op_);
    }
    using TiledArray::binary;
    return binary(first, second, element_op_, perm);
  }

  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type eval(ZeroTensor, const right_type& second,
                   const Perm& perm) const {
    TA_ASSERT(false);  // Invalid arguments for this operation
    return result_type();
  }

  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type eval(const left_type& first, ZeroTensor, const Perm& perm) const {
    TA_ASSERT(false);  // Invalid arguments for this operation
    return result_type();
  }

  // Non-permuting tile evaluation functions
  // The compiler will select the correct functions based on the
  // consumability of the arguments.

  template <bool LC, bool RC,
            typename std::enable_if<!(LC || RC)>::type* = nullptr>
  result_type eval(const left_type& first, const right_type& second) const {
    if constexpr (uses_tile_op_) {
      if (tile_op_) return eval_tile_op(first, second);
    }
    if constexpr (plain_mult_ok_) {
      if (!element_op_) {
        using TiledArray::mult;
        return mult(first, second);
      }
    } else {
      TA_ASSERT(element_op_);
    }
    using TiledArray::binary;
    return binary(first, second, element_op_);
  }

  template <bool LC, bool RC, typename std::enable_if<LC>::type* = nullptr>
  result_type eval(left_type& first, const right_type& second) const {
    if constexpr (uses_tile_op_) {
      if (tile_op_) return eval_tile_op(first, second);
    }
    if constexpr (plain_mult_ok_) {
      if (!element_op_) {
        if constexpr (uses_tile_op_) {
          // View inner cells (e.g. ArenaTensor): a "consumable" tile is a
          // shallow handle whose arena slab may be aliased by a persistent
          // array, so an in-place mult_to would corrupt that operand. Always
          // produce a fresh result for view-cell tiles.
          using TiledArray::mult;
          return mult(first, second);
        } else {
          using TiledArray::mult_to;
          return mult_to(std::move(first), second);
        }
      }
    } else {
      TA_ASSERT(element_op_);
    }
    // TODO figure out why this does not compiles!!!
    //            using TiledArray::inplace_binary;
    //            return inplace_binary(std::move(first), second,
    //            element_op_);
    using TiledArray::binary;
    return binary(first, second, element_op_);
  }

  template <bool LC, bool RC,
            typename std::enable_if<!LC && RC>::type* = nullptr>
  result_type eval(const left_type& first, right_type& second) const {
    if constexpr (uses_tile_op_) {
      if (tile_op_) return eval_tile_op(first, second);
    }
    if constexpr (plain_mult_ok_) {
      if (!element_op_) {
        if constexpr (uses_tile_op_) {
          // View inner cells: never consume a shallow handle in place (see the
          // consume-left overload above).
          using TiledArray::mult;
          return mult(first, second);
        } else {
          using TiledArray::mult_to;
          return mult_to(std::move(second), first);
        }
      }
    } else {
      TA_ASSERT(element_op_);
    }
    {  // WARNING: element_op_ might be noncommuting, so can't swap first
       // and second! for GEMM could optimize, but can't introspect
       // element_op_
      using TiledArray::binary;
      return binary(first, second, element_op_);
    }
  }

  template <bool LC, bool RC, typename std::enable_if<!RC>::type* = nullptr>
  result_type eval(ZeroTensor, const right_type& second) const {
    TA_ASSERT(false);  // Invalid arguments for this operation
    return result_type();
  }

  template <bool LC, bool RC, typename std::enable_if<RC>::type* = nullptr>
  result_type eval(ZeroTensor, right_type& second) const {
    TA_ASSERT(false);  // Invalid arguments for this operation
    return result_type();
  }

  template <bool LC, bool RC, typename std::enable_if<!LC>::type* = nullptr>
  result_type eval(const left_type& first, ZeroTensor) const {
    TA_ASSERT(false);  // Invalid arguments for this operation
    return result_type();
  }

  template <bool LC, bool RC, typename std::enable_if<LC>::type* = nullptr>
  result_type eval(left_type& first, ZeroTensor) const {
    TA_ASSERT(false);  // Invalid arguments for this operation
    return result_type();
  }

 public:
  /// The default constructor uses default op (`std::multiplies`) for the
  /// element-wise operation. This is valid for both plain and nested tensors.
  /// (times op is lowered naturally)
  Mult() = default;
  /// Construct using custom element-wise op
  /// \tparam ElementOp a callable with signature element_op_type
  /// \param op the element-wise operation
  template <typename ElementOp,
            typename = std::enable_if_t<
                !std::is_same_v<std::remove_reference_t<ElementOp>, Mult_> &&
                std::is_invocable_r_v<
                    result_value_type, std::remove_reference_t<ElementOp>,
                    const left_value_type&, const right_value_type&>>>
  explicit Mult(ElementOp&& op) : element_op_(std::forward<ElementOp>(op)) {}

  /// Tag selecting the whole-tile-op constructor.
  struct tile_op_tag {};

  /// Construct using a whole-tile op. When set, eval() delegates the entire
  /// tile product to \p op instead of multiplying element-wise. Used for
  /// arena tensor-of-tensors products whose per-cell op cannot value-return.
  /// \tparam TileOp a callable with signature
  ///         `result_type(const left_type&, const right_type&)`
  /// \param op the whole-tile operation
  template <typename TileOp, typename = std::enable_if_t<std::is_invocable_r_v<
                                 result_type, std::remove_reference_t<TileOp>,
                                 const left_type&, const right_type&>>>
  Mult(tile_op_tag, TileOp&& op) : tile_op_(std::forward<TileOp>(op)) {}

  /// Multiply-and-permute operator

  /// Compute the product of two tiles and permute the result.
  /// \tparam L The left-hand tile argument type
  /// \tparam R The right-hand tile argument type
  /// \param left The left-hand tile argument
  /// \param right The right-hand tile argument
  /// \param perm The permutation applied to the result tile
  /// \return The permuted and scaled product of `left` and `right`.
  template <
      typename L, typename R, typename Perm,
      typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
  result_type operator()(L&& left, R&& right, const Perm& perm) const {
    return eval(std::forward<L>(left), std::forward<R>(right), perm);
  }

  /// Multiply operator

  /// Compute the product of two tiles.
  /// \tparam L The left-hand tile argument type
  /// \tparam R The right-hand tile argument type
  /// \param left The left-hand tile argument
  /// \param right The right-hand tile argument
  /// \return The scaled product of `left` and `right`.
  template <typename L, typename R>
  result_type operator()(L&& left, R&& right) const {
    return Mult_::template eval<left_is_consumable, right_is_consumable>(
        std::forward<L>(left), std::forward<R>(right));
  }

  /// Multiply right to left

  /// Multiply the right tile to the left.
  /// \tparam R The right-hand tile argument type
  /// \param left The left-hand tile argument
  /// \param right The right-hand tile argument
  /// \return The product of `left` and `right`.
  template <typename R>
  result_type consume_left(left_type& left, R&& right) const {
    constexpr bool can_consume_left =
        is_consumable_tile<left_type>::value &&
        std::is_same<result_type, left_type>::value;
    constexpr bool can_consume_right =
        right_is_consumable && !(std::is_const<R>::value || can_consume_left);
    return Mult_::template eval<can_consume_left, can_consume_right>(
        left, std::forward<R>(right));
  }

  /// Multiply left to right

  /// Multiply the left tile to the right.
  /// \tparam L The left-hand tile argument type
  /// \param left The left-hand tile argument
  /// \param right The right-hand tile argument
  /// \return The product of `left` and `right`.
  template <typename L>
  result_type consume_right(L&& left, right_type& right) const {
    constexpr bool can_consume_right =
        is_consumable_tile<right_type>::value &&
        std::is_same<result_type, right_type>::value;
    constexpr bool can_consume_left =
        left_is_consumable && !(std::is_const<L>::value || can_consume_right);
    return Mult_::template eval<can_consume_left, can_consume_right>(
        std::forward<L>(left), right);
  }

};  // class Mult

/// Tile scale-multiplication operation

/// This multiplication operation will multiply the content two tiles and
/// apply a permutation to the result tensor. If no permutation is given or
/// the permutation is null, then the result is not permuted.
/// \tparam Result The result tile type
/// \tparam Left The left-hand argument type
/// \tparam Right The right-hand argument type
/// \tparam Scalar The scaling factor type
/// \tparam LeftConsumable If `true`, the left-hand tile is a temporary and
/// may be consumed
/// \tparam RightConsumable If `true`, the right-hand tile is a temporary
/// and may be consumed
/// \note Input tiles can be consumed only if their type matches the result
/// type.
template <typename Result, typename Left, typename Right, typename Scalar,
          bool LeftConsumable, bool RightConsumable>
class ScalMult {
 public:
  typedef ScalMult<Result, Left, Right, Scalar, LeftConsumable,
                   RightConsumable>
      ScalMult_;               ///< This class type
  typedef Left left_type;      ///< Left-hand argument base type
  typedef Right right_type;    ///< Right-hand argument base type
  typedef Scalar scalar_type;  ///< Scaling factor type
  typedef Result result_type;  ///< Result tile type

  /// Indicates whether it is *possible* to consume the left tile
  static constexpr bool left_is_consumable =
      LeftConsumable && std::is_same<result_type, left_type>::value;
  /// Indicates whether it is *possible* to consume the right tile
  static constexpr bool right_is_consumable =
      RightConsumable && std::is_same<result_type, right_type>::value;

 private:
  scalar_type factor_;  ///< The scaling factor

  // Permuting tile evaluation function
  // These operations cannot consume the argument tile since this operation
  // requires temporary storage space.

  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type eval(const left_type& first, const right_type& second,
                   const Perm& perm) const {
    using TiledArray::mult;
    return mult(first, second, factor_, perm);
  }

  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type eval(ZeroTensor, const right_type& second,
                   const Perm& perm) const {
    TA_ASSERT(false);  // Invalid arguments for this operation
    return result_type();
  }

  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type eval(const left_type& first, ZeroTensor, const Perm& perm) const {
    TA_ASSERT(false);  // Invalid arguments for this operation
    return result_type();
  }

  // Non-permuting tile evaluation functions
  // The compiler will select the correct functions based on the
  // consumability of the arguments.

  template <bool LC, bool RC,
            typename std::enable_if<!(LC || RC)>::type* = nullptr>
  result_type eval(const left_type& first, const right_type& second) const {
    using TiledArray::mult;
    return mult(first, second, factor_);
  }

  template <bool LC, bool RC, typename std::enable_if<LC>::type* = nullptr>
  result_type eval(left_type& first, const right_type& second) const {
    using TiledArray::mult_to;
    return mult_to(std::move(first), second, factor_);
  }

  template <bool LC, bool RC,
            typename std::enable_if<!LC && RC>::type* = nullptr>
  result_type eval(const left_type& first, right_type& second) const {
    using TiledArray::mult_to;
    return mult_to(std::move(second), first, factor_);
  }

  template <bool LC, bool RC, typename std::enable_if<!RC>::type* = nullptr>
  result_type eval(ZeroTensor, const right_type& second) const {
    TA_ASSERT(false);  // Invalid arguments for this operation
    return result_type();
  }

  template <bool LC, bool RC, typename std::enable_if<RC>::type* = nullptr>
  result_type eval(ZeroTensor, right_type& second) const {
    TA_ASSERT(false);  // Invalid arguments for this operation
    return result_type();
  }

  template <bool LC, bool RC, typename std::enable_if<!LC>::type* = nullptr>
  result_type eval(const left_type& first, ZeroTensor) const {
    TA_ASSERT(false);  // Invalid arguments for this operation
    return result_type();
  }

  template <bool LC, bool RC, typename std::enable_if<LC>::type* = nullptr>
  result_type eval(left_type& first, ZeroTensor) const {
    TA_ASSERT(false);  // Invalid arguments for this operation
    return result_type();
  }

 public:
  // Compiler generated functions
  ScalMult(const ScalMult_&) = default;
  ScalMult(ScalMult_&&) = default;
  ~ScalMult() = default;
  ScalMult_& operator=(const ScalMult_&) = default;
  ScalMult_& operator=(ScalMult_&&) = default;

  /// Constructor

  /// \param factor The scaling factor applied to result tiles
  explicit ScalMult(const Scalar factor) : factor_(factor) {}

  /// Scale-multiply-and-permute operator

  /// Compute the scaled product of two tiles and permute the result.
  /// \tparam L The left-hand tile argument type
  /// \tparam R The right-hand tile argument type
  /// \param left The left-hand tile argument
  /// \param right The right-hand tile argument
  /// \param perm The permutation applied to the result tile
  /// \return The permuted and scaled product of `left` and `right`.
  template <
      typename L, typename R, typename Perm,
      typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
  result_type operator()(L&& left, R&& right, const Perm& perm) const {
    return eval(std::forward<L>(left), std::forward<R>(right), perm);
  }

  /// Scale-and-multiply operator

  /// Compute the scaled product of two tiles.
  /// \tparam L The left-hand tile argument type
  /// \tparam R The right-hand tile argument type
  /// \param left The left-hand tile argument
  /// \param right The right-hand tile argument
  /// \return The scaled product of `left` and `right`.
  template <typename L, typename R>
  result_type operator()(L&& left, R&& right) const {
    return ScalMult_::template eval<left_is_consumable, right_is_consumable>(
        std::forward<L>(left), std::forward<R>(right));
  }

  /// Multiply right to left and scale the result

  /// Multiply the right tile to the left.
  /// \tparam R The right-hand tile argument type
  /// \param left The left-hand tile argument
  /// \param right The right-hand tile argument
  /// \return The product of `left` and `right`.
  template <typename R>
  result_type consume_left(left_type& left, R&& right) const {
    constexpr bool can_consume_left =
        is_consumable_tile<left_type>::value &&
        std::is_same<result_type, left_type>::value;
    constexpr bool can_consume_right =
        right_is_consumable && !(std::is_const<R>::value || can_consume_left);
    return ScalMult_::template eval<can_consume_left, can_consume_right>(
        left, std::forward<R>(right));
  }

  /// Multiply left to right and scale the result

  /// Multiply the left tile to the right, and scale the resulting left
  /// tile.
  /// \tparam L The left-hand tile argument type
  /// \param left The left-hand tile argument
  /// \param right The right-hand tile argument
  /// \return The product of `left` and `right`.
  template <typename L>
  result_type consume_right(L&& left, right_type& right) const {
    constexpr bool can_consume_right =
        is_consumable_tile<right_type>::value &&
        std::is_same<result_type, right_type>::value;
    constexpr bool can_consume_left =
        left_is_consumable && !(std::is_const<L>::value || can_consume_right);
    return ScalMult_::template eval<can_consume_left, can_consume_right>(
        std::forward<L>(left), right);
  }

};  // class ScalMult

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TILE_OP_MULT_H__INCLUDED
