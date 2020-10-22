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
 *  noop.h
 *  June 27, 2013
 *
 */

#ifndef TILEDARRAY_TILE_OP_NOOP_H__INCLUDED
#define TILEDARRAY_TILE_OP_NOOP_H__INCLUDED

#include "../tile_interface/clone.h"
#include "../tile_interface/permute.h"

namespace TiledArray {
namespace detail {

/// Tile no operation (noop)

/// This no operation will return the original or apply a permutation to the
/// result tensor. If no permutation is given or the permutation is null,
/// then the result is not permuted.
/// \tparam Result The result tile type
/// \tparam Arg The argument type
/// \tparam Consumable Flag that is \c true when Arg is consumable
template <typename Result, typename Arg, bool Consumable>
class Noop {
 public:
  typedef Noop<Result, Arg, Consumable> Noop_;  ///< This object type
  typedef Arg argument_type;                    ///< The argument type
  typedef Result result_type;                   ///< The result tile type

  static constexpr bool is_consumable = Consumable;

 private:
  // Permuting tile evaluation function
  // These operations cannot consume the argument tile since this operation
  // requires temporary storage space.
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  result_type eval(const Arg& arg, const Perm& perm) const {
    TiledArray::Permute<Result, Arg> permute;
    return permute(arg, perm);
  }

  // Non-permuting tile evaluation functions
  // The compiler will select the correct functions based on the
  // consumability of the arguments.

  template <bool C, typename std::enable_if<!C>::type* = nullptr>
  result_type eval(const Arg& arg) const {
    TiledArray::Clone<Result, Arg> clone;
    return clone(arg);
  }

  template <bool C, typename std::enable_if<C>::type* = nullptr>
  result_type eval(Arg& arg) const {
    return arg;
  }

 public:
  /// Permute operator

  /// \tparam A The tile argument type
  /// \param arg The tile argument
  /// \param perm The permutation applied to the result tile
  /// \return A permuted copy of `arg`
  template <typename Perm,
            typename = std::enable_if_t<detail::is_permutation_v<Perm>>>
  result_type operator()(const argument_type& arg, const Perm& perm) const {
    return eval(arg, perm);
  }

  /// Clone operator

  /// \tparam A The tile argument type
  /// \param arg The tile argument
  /// \return A clone of the `arg`
  template <typename A>
  result_type operator()(A&& arg) const {
    return Noop_::template eval<is_consumable>(arg);
  }

  /// Pass-through operations (shallow copy)

  /// \param arg The tile argument
  /// \return `arg`
  result_type consume(argument_type& arg) const {
    constexpr bool can_consume =
        is_consumable_tile<argument_type>::value &&
        std::is_same<result_type, argument_type>::value;
    return Noop_::template eval<can_consume>(arg);
  }

};  // class Noop

}  // namespace detail
}  // namespace TiledArray

#endif  // TILEDARRAY_TILE_OP_NOOP_H__INCLUDED
