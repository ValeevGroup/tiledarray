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
 *  scal.h
 *  June 20, 2013
 *
 */

#ifndef TILEDARRAY_TILE_OP_SCAL_H__INCLUDED
#define TILEDARRAY_TILE_OP_SCAL_H__INCLUDED

#include <TiledArray/tile_op/tile_interface.h>
#include <type_traits>
#include "../tile_interface/scale.h"

namespace TiledArray {
namespace detail {

/// Tile scaling operation

/// This scaling operation will scale the content a tile and apply a
/// permutation to the result tensor. If no permutation is given or the
/// permutation is null, then the result is not permuted.
/// \tparam Result The result type
/// \tparam Arg The argument type
/// \tparam Scalar The scaling factor type
/// \tparam Consumable Flag that is \c true when Arg is consumable
template <typename Result, typename Arg, typename Scalar, bool Consumable>
class Scal {
 public:
  typedef Scal<Result, Arg, Scalar, Consumable> Scal_;  ///< This object type
  typedef Arg argument_type;                            ///< The argument type
  typedef Scalar scalar_type;  ///< The scaling factor type
  typedef Result result_type;  ///< The result tile type

  static constexpr bool is_consumable =
      Consumable && std::is_same<result_type, argument_type>::value;

 private:
  scalar_type factor_;  ///< Scaling factor

  // Permuting tile evaluation function
  // These operations cannot consume the argument tile since this operation
  // requires temporary storage space.
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  result_type eval(const Arg& arg, const Perm& perm) const {
    using TiledArray::scale;
    return scale(arg, factor_, perm);
  }

  // Non-permuting tile evaluation functions
  // The compiler will select the correct functions based on the
  // consumability of the arguments.

  template <bool C, typename std::enable_if<!C>::type* = nullptr>
  result_type eval(const argument_type& arg) const {
    using TiledArray::scale;
    return scale(arg, factor_);
  }

  template <bool C, typename std::enable_if<C>::type* = nullptr>
  result_type eval(argument_type& arg) const {
    using TiledArray::scale_to;
    return scale_to(arg, factor_);
  }

 public:
  // Compiler generated functions
  Scal(const Scal_&) = default;
  Scal(Scal_&&) = default;
  ~Scal() = default;
  Scal_& operator=(const Scal_&) = default;
  Scal_& operator=(Scal_&&) = default;

  /// Constructor

  /// Construct a scaling operation that scales the result tensor
  /// \param factor The scaling factor for the operation
  explicit Scal(const scalar_type factor) : factor_(factor) {}

  /// Scale and permute operator

  /// \param arg The tile argument
  /// \param perm The permutation applied to the result tile
  /// \return A permuted and scaled copy of `arg`
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  result_type operator()(const argument_type& arg, const Perm& perm) const {
    return eval(arg, perm);
  }

  /// Consuming scale operation

  /// \tparam A The tile argument type
  /// \param arg The tile argument
  /// \return A scaled copy of `arg`
  template <typename A>
  result_type operator()(A&& arg) const {
    return Scal_::template eval<is_consumable>(std::forward<A>(arg));
  }

  /// Explicit consuming scale operation

  /// \param arg The tile argument
  /// \return In-place scaled `arg`
  result_type consume(argument_type& arg) const {
    constexpr bool can_consume =
        is_consumable_tile<argument_type>::value &&
        std::is_same<result_type, argument_type>::value;
    return Scal_::template eval<can_consume>(arg);
  }

  void set_factor(const scalar_type factor) { factor_ = factor; }

};  // class Scal

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TILE_OP_SCAL_H__INCLUDED
