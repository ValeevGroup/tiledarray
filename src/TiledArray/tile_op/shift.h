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
 *  shift.h
 *  June 7, 2015
 *
 */

#ifndef TILEDARRAY_TILE_OP_SHIFT_H__INCLUDED
#define TILEDARRAY_TILE_OP_SHIFT_H__INCLUDED

#include "../tile_interface/permute.h"
#include "../tile_interface/shift.h"

#include <algorithm>
#include <type_traits>

namespace TiledArray {
namespace detail {

/// Tile shift operation

/// This tile operation will shift the range of the tile and/or apply a
/// permutation to the result tensor.
/// \tparam Result The tile result type
/// \tparam Arg The argument type
/// \tparam Consumable If `true`, the tile is a temporary and may be
/// consumed
/// \note Input tiles can be consumed only if their type matches the result
/// type.
/// \tparam Share If `true`, a zero shift of a non-consumable argument hands
/// the argument through (a shallow copy, for a tile type like TA::Tensor whose
/// copies share the data) instead of cloning it, as Noop does: the result
/// then shares the argument's data (see is_pass_through()). This is the op of
/// a block of a const array, whose tiles its consumer must never consume.
template <typename Result, typename Arg, bool Consumable, bool Share = false>
class Shift {
 public:
  typedef Shift<Result, Arg, Consumable, Share> Shift_;  ///< This object type
  typedef Arg argument_type;                             ///< The argument type
  typedef Result result_type;  ///< The result tile type

  /// Indicates whether it is *possible* to consume the left tile
  static constexpr bool is_consumable =
      Consumable && std::is_same<result_type, argument_type>::value;

  /// true if this op would hand a non-consumable argument through unchanged,
  /// so that its result shares the argument's data: a zero shift in sharing
  /// mode (see \p Share)
  bool is_pass_through() const noexcept {
    if constexpr (Share && std::is_same_v<result_type, argument_type>)
      return is_zero_shift();
    else
      return false;
  }

 private:
  container::svector<long> range_shift_;

  bool is_zero_shift() const noexcept {
    return std::all_of(range_shift_.begin(), range_shift_.end(),
                       [](long s) { return s == 0l; });
  }

  // Permuting tile evaluation function
  // These operations cannot consume the argument tile since this operation
  // requires temporary storage space.
  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type eval(const argument_type& arg, const Perm& perm) const {
    TiledArray::Permute<result_type, argument_type> permute;
    TiledArray::ShiftTo<result_type, result_type> shift_to;
    result_type result = permute(arg, perm);
    shift_to(result, range_shift_);
    return result;
  }

  // Non-permuting tile evaluation functions
  // The compiler will select the correct functions based on the
  // consumability of the arguments.

  template <bool C, typename = void>
  auto eval(const argument_type& arg) const {
    if constexpr (Share && std::is_same_v<result_type, argument_type>) {
      // nothing to relocate: hand the tile through (a shallow copy for a
      // shallow-copy tile type) instead of cloning it
      if (is_zero_shift()) return result_type(arg);
    }
    TiledArray::Shift<result_type, argument_type> shift;
    return shift(arg, range_shift_);
  }

  template <bool C, typename = typename std::enable_if<C>::type>
  auto eval(argument_type& arg) const {
    TiledArray::ShiftTo<result_type, argument_type> shift_to;
    return shift_to(std::move(arg), range_shift_);
  }

  template <bool C, typename = typename std::enable_if<C>::type>
  auto eval(argument_type&& arg) const {
    TiledArray::ShiftTo<result_type, argument_type> shift_to;
    return shift_to(std::move(arg), range_shift_);
  }

 public:
  // Compiler generated functions
  Shift() = delete;
  Shift(const Shift_&) = default;
  Shift(Shift_&&) = default;
  ~Shift() = default;
  Shift& operator=(const Shift_&) = default;
  Shift& operator=(Shift_&&) = default;

  /// Default constructor

  /// Construct a no operation that does not permute the result tile
  template <typename IntegralRange,
            typename = std::enable_if_t<
                TiledArray::detail::is_integral_range_v<IntegralRange>>>
  Shift(IntegralRange&& range_shift)
      : range_shift_(std::forward<IntegralRange>(range_shift).begin(),
                     std::forward<IntegralRange>(range_shift).end()) {}

  /// Shift and permute operator

  /// \param arg The tile argument
  /// \param perm The permutation applied to the result tile
  /// \return A permuted and shifted copy of `arg`
  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type operator()(const argument_type& arg, const Perm& perm) const {
    return eval(arg, perm);
  }

  /// Consuming shift operation

  /// \tparam A The tile argument type
  /// \param arg The tile argument
  /// \return A shifted copy of `arg`
  template <typename A>
  result_type operator()(A&& arg) const {
    return Shift_::template eval<is_consumable>(std::forward<A>(arg));
  }

  /// Explicit consuming shift operation

  /// \tparam A The tile argument type
  /// \param arg The tile argument
  /// \return In-place shifted `arg`
  template <typename A>
  result_type consume(A& arg) const {
    constexpr bool can_consume =
        is_consumable_tile<argument_type>::value &&
        std::is_same<result_type, argument_type>::value;
    return Shift_::template eval<can_consume>(arg);
  }

};  // class Shift

/// Tile shift operation

/// This tile operation will shift the range of the tile and/or apply a
/// permutation to the result tensor.
/// \tparam Result The result type
/// \tparam Arg The argument type
/// \tparam Scalar The scaling factor type
/// \tparam Consumable Flag that is \c true when Arg is consumable
template <typename Result, typename Arg, typename Scalar, bool Consumable>
class ScalShift {
 public:
  typedef ScalShift<Result, Arg, Scalar, Consumable>
      ScalShift_;              ///< This object type
  typedef Arg argument_type;   ///< The argument type
  typedef Scalar scalar_type;  ///< The scaling factor type
  typedef Result result_type;  ///< The result tile type

  static constexpr bool is_consumable =
      Consumable && std::is_same<result_type, argument_type>::value;

 private:
  container::svector<long> range_shift_;  ///< Range shift array
  scalar_type factor_;                    ///< Scaling factor

 public:
  // Permuting tile evaluation function
  // These operations cannot consume the argument tile since this operation
  // requires temporary storage space.
  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type eval(const argument_type& arg, const Perm& perm) const {
    using TiledArray::scale;
    using TiledArray::shift_to;
    result_type result = scale(arg, factor_, perm);
    return shift_to(std::move(result), range_shift_);
  }

  // Non-permuting tile evaluation functions
  // The compiler will select the correct functions based on the
  // consumability of the arguments.

  template <bool C>
  typename std::enable_if<!C, result_type>::type eval(
      const argument_type& arg) const {
    using TiledArray::scale;
    using TiledArray::shift_to;
    result_type result = scale(arg, factor_);
    return shift_to(std::move(result), range_shift_);
  }

  template <bool C>
  typename std::enable_if<C, result_type>::type eval(argument_type& arg) const {
    using TiledArray::scale_to;
    using TiledArray::shift_to;
    return shift_to(scale_to(std::move(arg), factor_), range_shift_);
  }

 public:
  // Compiler generated functions
  ScalShift() = delete;
  ScalShift(const ScalShift_&) = default;
  ScalShift(ScalShift_&&) = default;
  ~ScalShift() = default;
  ScalShift_& operator=(const ScalShift_&) = default;
  ScalShift_& operator=(ScalShift_&&) = default;

  /// Default constructor

  /// Construct a no operation that does not permute the result tile
  template <typename IntegralRange,
            typename = std::enable_if_t<
                TiledArray::detail::is_integral_range_v<IntegralRange>>>
  ScalShift(IntegralRange&& range_shift, const scalar_type factor)
      : range_shift_(std::forward<IntegralRange>(range_shift).begin(),
                     std::forward<IntegralRange>(range_shift).end()),
        factor_(factor) {}

  /// Shift and permute operator

  /// \param arg The tile argument
  /// \param perm The permutation applied to the result tile
  /// \return A permuted and shifted copy of `arg`
  template <typename Perm, typename = std::enable_if_t<
                               TiledArray::detail::is_permutation_v<Perm>>>
  result_type operator()(const argument_type& arg, const Perm& perm) const {
    return eval(arg, perm);
  }

  /// Consuming shift operation

  /// \tparam A The tile argument type
  /// \param arg The tile argument
  /// \return A shifted copy of `arg`
  template <typename A>
  result_type operator()(A&& arg) const {
    return ScalShift_::template eval<is_consumable>(std::forward<A>(arg));
  }

  /// Explicit consuming shift operation

  /// \param arg The tile argument
  /// \return In-place shifted `arg`
  result_type consume(argument_type& arg) const {
    constexpr bool can_consume =
        is_consumable_tile<argument_type>::value &&
        std::is_same<result_type, argument_type>::value;
    return ScalShift_::template eval<can_consume>(arg);
  }

};  // class ScalShift

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TILE_OP_SHIFT_H__INCLUDED
