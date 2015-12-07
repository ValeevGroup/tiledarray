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
 *  neg.h
 *  May 7, 2013
 *
 */

#ifndef TILEDARRAY_TILE_OP_NEG_H__INCLUDED
#define TILEDARRAY_TILE_OP_NEG_H__INCLUDED

#include <TiledArray/tile_op/tile_interface.h>

namespace TiledArray {

  /// Tile negation operation

  /// This negation operation will negate the content a tile and apply a
  /// permutation to the result tensor. If no permutation is given or the
  /// permutation is null, then the result is not permuted.
  /// \tparam Result The result type
  /// \tparam Arg The argument type
  /// \tparam Consumable Flag that is \c true when Arg is consumable
  template <typename Arg, bool Consumable>
  class Neg {
  public:
    typedef Neg<Arg, Consumable> Neg_; ///< This object type
    typedef Arg argument_type; ///< The argument type
    typedef decltype(neg(std::declval<argument_type>()))
        result_type; ///< The result tile type

    static constexpr bool is_consumable =
        Consumable && std::is_same<result_type, argument_type>::value;

  private:

    // Permuting tile evaluation function
    // These operations cannot consume the argument tile since this operation
    // requires temporary storage space.

    result_type eval(const Arg& arg, const Permutation& perm) const {
      using TiledArray::neg;
      return neg(arg, perm);
    }

    // Non-permuting tile evaluation functions
    // The compiler will select the correct functions based on the
    // consumability of the arguments.

    template <bool C, typename std::enable_if<!C>::type* = nullptr>
    static result_type eval(const Arg& arg) {
      using TiledArray::neg;
      return neg(arg);
    }

    template <bool C, typename std::enable_if<C>::type* = nullptr>
    static result_type eval(Arg& arg) {
      using TiledArray::neg_to;
      return neg_to(arg);
    }

  public:

    /// Negate and permute operator

    /// \tparam A The tile argument type
    /// \param arg The tile argument
    /// \param perm The permutation applied to the result tile
    /// \return A permuted and negated copy of `arg`
    template <typename A>
    result_type operator()(A&& arg, const Permutation& perm) const {
      return eval(arg, perm);
    }

    /// Consuming negate operation

    /// \tparam A The tile argument type
    /// \param arg The tile argument
    /// \return A negated copy of `arg`
    template <typename A>
    result_type operator()(A&& arg) const {
      return Neg_::template eval<is_consumable>(arg);
    }

    /// Explicit consuming negate operation

    /// \tparam A The tile argument type
    /// \param arg The tile argument
    /// \return In-place negated `arg`
    template <typename A>
    result_type consume(A& arg) const {
      return Neg_::template eval<is_consumable_tile<Arg>::value>(arg);
    }

  }; // class Neg

} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_NEG_H__INCLUDED
