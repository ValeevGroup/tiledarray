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
 *  subt.h
 *  May 8, 2013
 *
 */

#ifndef TILEDARRAY_TILE_OP_SUBT_H__INCLUDED
#define TILEDARRAY_TILE_OP_SUBT_H__INCLUDED

#include <TiledArray/tile_op/tile_interface.h>
#include <TiledArray/zero_tensor.h>
#include "../tile_interface/clone.h"
#include "../tile_interface/permute.h"
#include "../tile_interface/scale.h"

namespace TiledArray {
namespace detail {

/// Tile subtraction operation

/// This subtraction operation will subtract the content two tiles, and
/// accepts an optional permute argument.
/// \tparam Result The result tile type
/// \tparam Left The left-hand argument base type
/// \tparam Right The right-hand argument base type
/// \tparam LeftConsumable If `true`, the left-hand tile is a temporary and
/// may be consumed
/// \tparam RightConsumable If `true`, the right-hand tile is a temporary
/// and may be consumed
/// \note Input tiles can be consumed only if their type matches the result
/// type.
template <typename Result, typename Left, typename Right, bool LeftConsumable,
          bool RightConsumable>
class Subt {
 public:
  typedef Subt<Result, Left, Right, LeftConsumable, RightConsumable> Subt_;
  typedef Left left_type;      ///< Left-hand argument base type
  typedef Right right_type;    ///< Right-hand argument base type
  typedef Result result_type;  ///< The result tile type

  /// Indicates whether it is *possible* to consume the left tile
  static constexpr bool left_is_consumable =
      LeftConsumable && std::is_same<result_type, left_type>::value;
  /// Indicates whether it is *possible* to consume the right tile
  static constexpr bool right_is_consumable =
      RightConsumable && std::is_same<result_type, right_type>::value;

 private:
  // Permuting tile evaluation function
  // These operations cannot consume the argument tile since this operation
  // requires temporary storage space.

  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  static result_type eval(const left_type& first, const right_type& second,
                          const Perm& perm) {
    using TiledArray::subt;
    return subt(first, second, perm);
  }

  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  static result_type eval(ZeroTensor, const right_type& second,
                          const Perm& perm) {
    using TiledArray::neg;
    return neg(second, perm);
  }

  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  static result_type eval(const left_type& first, ZeroTensor,
                          const Perm& perm) {
    using TiledArray::permute;
    return permute(first, perm);
  }

  // Non-permuting tile evaluation functions
  // The compiler will select the correct functions based on the
  // consumability of the arguments.

  template <bool LC, bool RC,
            typename std::enable_if<!(LC || RC)>::type* = nullptr>
  static result_type eval(const left_type& first, const right_type& second) {
    using TiledArray::subt;
    return subt(first, second);
  }

  template <bool LC, bool RC, typename std::enable_if<LC>::type* = nullptr>
  static result_type eval(left_type& first, const right_type& second) {
    using TiledArray::subt_to;
    return subt_to(std::move(first), second);
  }

  template <bool LC, bool RC,
            typename std::enable_if<!LC && RC>::type* = nullptr>
  static result_type eval(const left_type& first, right_type& second) {
    using TiledArray::subt_to;
    return subt_to(std::move(second), first, -1);
  }

  template <bool LC, bool RC, typename std::enable_if<!RC>::type* = nullptr>
  static result_type eval(ZeroTensor, const right_type& second) {
    using TiledArray::neg;
    return neg(second);
  }

  template <bool LC, bool RC, typename std::enable_if<RC>::type* = nullptr>
  static result_type eval(ZeroTensor, right_type& second) {
    using TiledArray::neg_to;
    return neg_to(std::move(second));
  }

  template <bool LC, bool RC, typename std::enable_if<!LC>::type* = nullptr>
  static result_type eval(const left_type& first, ZeroTensor) {
    TiledArray::Clone<result_type, left_type> clone;
    return clone(first);
  }

  template <bool LC, bool RC, typename std::enable_if<LC>::type* = nullptr>
  static result_type eval(left_type& first, ZeroTensor) {
    return first;
  }

 public:
  /// Subtract-and-permute operator

  /// Compute the difference of two tiles and permute the result. One of the
  /// argument tiles may be replaced with `ZeroTensor` argument, in which
  /// case the argument's element values are assumed to be `0`.
  /// \tparam L The left-hand tile argument type
  /// \tparam R The right-hand tile argument type
  /// \param left The left-hand tile argument
  /// \param right The right-hand tile argument
  /// \param perm The permutation applied to the result tile
  /// \return The permuted and scaled difference of `left` and `right`.
  template <
      typename L, typename R, typename Perm,
      typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
  result_type operator()(L&& left, R&& right, const Perm& perm) const {
    return eval(std::forward<L>(left), std::forward<R>(right), perm);
  }

  /// Subtract operator

  /// Compute the difference of two tiles. One of the argument tiles may be
  /// replaced with `ZeroTensor` argument, in which case the argument's
  /// element values are assumed to be `0`.
  /// \tparam L The left-hand tile argument type
  /// \tparam R The right-hand tile argument type
  /// \param left The left-hand tile argument
  /// \param right The right-hand tile argument
  /// \return The scaled difference of `left` and `right`.
  template <typename L, typename R>
  result_type operator()(L&& left, R&& right) const {
    return Subt_::template eval<left_is_consumable, right_is_consumable>(
        std::forward<L>(left), std::forward<R>(right));
  }

  /// Subtract right to left

  /// Subtract the right tile to the left. The right tile may be replaced
  /// with `ZeroTensor` argument, in which case the argument's element
  /// values are assumed to be `0`.
  /// \tparam R The right-hand tile argument type
  /// \param left The left-hand tile argument
  /// \param right The right-hand tile argument
  /// \return The difference of `left` and `right`.
  template <typename R>
  result_type consume_left(left_type& left, R&& right) const {
    constexpr bool can_consume_left =
        is_consumable_tile<left_type>::value &&
        std::is_same<result_type, left_type>::value;
    constexpr bool can_consume_right =
        right_is_consumable && !(std::is_const<R>::value || can_consume_left);
    return Subt_::template eval<can_consume_left, can_consume_right>(
        left, std::forward<R>(right));
  }

  /// Subtract left to right

  /// Subtract the left tile to the right. The left tile may be replaced
  /// with `ZeroTensor` argument, in which case the argument's element
  /// values are assumed to be `0`.
  /// \tparam L The left-hand tile argument type
  /// \param left The left-hand tile argument
  /// \param right The right-hand tile argument
  /// \return The difference of `left` and `right`.
  template <typename L>
  result_type consume_right(L&& left, right_type& right) const {
    constexpr bool can_consume_right =
        is_consumable_tile<right_type>::value &&
        std::is_same<result_type, right_type>::value;
    constexpr bool can_consume_left =
        left_is_consumable && !(std::is_const<L>::value || can_consume_right);
    return Subt_::template eval<can_consume_left, can_consume_right>(
        std::forward<L>(left), right);
  }

};  // class Subt

/// Tile scale-subtraction operation

/// This subtraction operation will subtract the content two tiles and apply
/// a permutation to the result tensor. If no permutation is given or the
/// permutation is null, then the result is not permuted.
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
class ScalSubt {
 public:
  typedef ScalSubt<Result, Left, Right, Scalar, LeftConsumable,
                   RightConsumable>
      ScalSubt_;               ///< This class type
  typedef Left left_type;      ///< Left-hand argument base type
  typedef Right right_type;    ///< Right-hand argument base type
  typedef Scalar scalar_type;  ///< Scaling factor type
  typedef Result result_type;  ///< The result tile type

  static constexpr bool left_is_consumable =
      LeftConsumable && std::is_same<result_type, left_type>::value;
  static constexpr bool right_is_consumable =
      RightConsumable && std::is_same<result_type, right_type>::value;

 private:
  scalar_type factor_;

  // Permuting tile evaluation function
  // These operations cannot consume the argument tile since this operation
  // requires temporary storage space.

  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type eval(const left_type& first, const right_type& second,
                   const Perm& perm) const {
    using TiledArray::subt;
    return subt(first, second, factor_, perm);
  }

  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type eval(ZeroTensor, const right_type& second,
                   const Perm& perm) const {
    using TiledArray::scale;
    return scale(second, -factor_, perm);
  }

  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type eval(const left_type& first, ZeroTensor, const Perm& perm) const {
    using TiledArray::scale;
    return scale(first, factor_, perm);
  }

  // Non-permuting tile evaluation functions
  // The compiler will select the correct functions based on the
  // consumability of the arguments.

  template <bool LC, bool RC,
            typename std::enable_if<!(LC || RC)>::type* = nullptr>
  result_type eval(const left_type& first, const right_type& second) const {
    using TiledArray::subt;
    return subt(first, second, factor_);
  }

  template <bool LC, bool RC, typename std::enable_if<LC>::type* = nullptr>
  result_type eval(left_type& first, const right_type& second) const {
    using TiledArray::subt_to;
    return subt_to(std::move(first), second, factor_);
  }

  template <bool LC, bool RC,
            typename std::enable_if<!LC && RC>::type* = nullptr>
  result_type eval(const left_type& first, right_type& second) const {
    using TiledArray::subt_to;
    return subt_to(std::move(second), first, -factor_);
  }

  template <bool LC, bool RC, typename std::enable_if<!RC>::type* = nullptr>
  result_type eval(ZeroTensor, const right_type& second) const {
    using TiledArray::scale;
    return scale(second, -factor_);
  }

  template <bool LC, bool RC, typename std::enable_if<RC>::type* = nullptr>
  result_type eval(ZeroTensor, right_type& second) const {
    using TiledArray::scale_to;
    return scale_to(std::move(second), -factor_);
  }

  template <bool LC, bool RC, typename std::enable_if<!LC>::type* = nullptr>
  result_type eval(const left_type& first, ZeroTensor) const {
    using TiledArray::scale;
    return scale(first, factor_);
  }

  template <bool LC, bool RC, typename std::enable_if<LC>::type* = nullptr>
  result_type eval(left_type& first, ZeroTensor) const {
    using TiledArray::scale_to;
    return scale_to(std::move(first), factor_);
  }

 public:
  // Compiler generated functions
  ScalSubt(const ScalSubt_&) = default;
  ScalSubt(ScalSubt_&&) = default;
  ~ScalSubt() = default;
  ScalSubt_& operator=(const ScalSubt_&) = default;
  ScalSubt_& operator=(ScalSubt_&&) = default;

  /// Constructor

  /// \param factor The scaling factor applied to result tiles
  explicit ScalSubt(const Scalar factor) : factor_(factor) {}

  /// Scale-subtract-and-permute operator

  /// Compute the scaled difference of two tiles and permute the result.
  /// One of the argument tiles may be replaced with `ZeroTensor` argument,
  /// in which case the argument's element values are assumed to be `0`.
  /// \tparam L The left-hand tile argument type
  /// \tparam R The right-hand tile argument type
  /// \param left The left-hand tile argument
  /// \param right The right-hand tile argument
  /// \param perm The permutation applied to the result tile
  /// \return The permuted and scaled difference of `left` and `right`.
  template <
      typename L, typename R, typename Perm,
      typename = std::enable_if_t<TiledArray::detail::is_permutation_v<Perm>>>
  result_type operator()(L&& left, R&& right, const Perm& perm) const {
    return eval(std::forward<L>(left), std::forward<R>(right), perm);
  }

  /// Scale-and-subtract operator

  /// Compute the scaled difference of two tiles. One of the argument tiles
  /// may be replaced with `ZeroTensor` argument, in which case the
  /// argument's element values are assumed to be `0`.
  /// \tparam L The left-hand tile argument type
  /// \tparam R The right-hand tile argument type
  /// \param left The left-hand tile argument
  /// \param right The right-hand tile argument
  /// \return The scaled difference of `left` and `right`.
  template <typename L, typename R>
  result_type operator()(L&& left, R&& right) const {
    return ScalSubt_::template eval<left_is_consumable, right_is_consumable>(
        std::forward<L>(left), std::forward<R>(right));
  }

  /// Subtract right to left and scale the result

  /// Subtract the right tile to the left. The right tile may be replaced
  /// with `ZeroTensor` argument, in which case the argument's element
  /// values are assumed to be `0`.
  /// \tparam R The right-hand tile argument type
  /// \param left The left-hand tile argument
  /// \param right The right-hand tile argument
  /// \return The difference of `left` and `right`.
  template <typename R>
  result_type consume_left(left_type& left, R&& right) const {
    constexpr bool can_consume_left =
        is_consumable_tile<left_type>::value &&
        std::is_same<result_type, left_type>::value;
    constexpr bool can_consume_right =
        right_is_consumable && !(std::is_const<R>::value || can_consume_left);
    return ScalSubt_::template eval<can_consume_left, can_consume_right>(
        left, std::forward<R>(right));
  }

  /// Subtract left to right and scale the result

  /// Subtract the left tile to the right, and scale the resulting left
  /// tile. The left tile may be replaced with `ZeroTensor` argument, in
  /// which case the argument's element values are assumed to be `0`.
  /// \tparam L The left-hand tile argument type
  /// \param left The left-hand tile argument
  /// \param right The right-hand tile argument
  /// \return The difference of `left` and `right`.
  template <typename L>
  result_type consume_right(L&& left, right_type& right) const {
    constexpr bool can_consume_right =
        is_consumable_tile<right_type>::value &&
        std::is_same<result_type, right_type>::value;
    constexpr bool can_consume_left =
        left_is_consumable && !(std::is_const<L>::value || can_consume_right);
    return ScalSubt_::template eval<can_consume_left, can_consume_right>(
        std::forward<L>(left), right);
  }

};  // class ScalSubt

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TILE_OP_SUBT_H__INCLUDED
