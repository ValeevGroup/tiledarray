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

#include <TiledArray/tile_op/unary_interface.h>

namespace TiledArray {
  namespace math {

    /// Tile negation operation

    /// This negation operation will negate the content a tile and apply a
    /// permutation to the result tensor. If no permutation is given or the
    /// permutation is null, then the result is not permuted.
    /// \tparam Result The result type
    /// \tparam Arg The argument type
    /// \tparam Consumable Flag that is \c true when Arg is consumable
    template <typename Result, typename Arg, bool Consumable>
    class Neg : public UnaryInterface<Neg<Result, Arg, Consumable> > {
    public:
      typedef Neg<Result, Arg, Consumable> Neg_; ///< This object type
      typedef UnaryInterface<Neg_> UnaryInterface_;
      typedef typename UnaryInterface_::argument_type argument_type; ///< The argument type
      typedef typename UnaryInterface_::result_type result_type; ///< The result tile type

      /// Default constructor

      /// Construct a negation operation that does not permute the result tile
      Neg() : UnaryInterface_() { }

      /// Permute constructor

      /// Construct a negation operation that permutes the result tensor
      /// \param perm The permutation to apply to the result tile
      Neg(const Permutation& perm) : UnaryInterface_(perm) { }

      /// Copy constructor

      /// \param other The negation operation object to be copied
      Neg(const Neg_& other) : UnaryInterface_(other) { }

      /// Copy assignment

      /// \param other The negation operation object to be copied
      /// \return A reference to this object
      Neg_& operator=(const Neg_& other) {
        UnaryInterface_::operator =(other);
        return *this;
      }

      // Import interface from base class
      using UnaryInterface_::operator();

      // Permuting tile evaluation function
      // These operations cannot consume the argument tile since this operation
      // requires temporary storage space.

      result_type permute(const Arg& arg) const {
        return TiledArray::neg(arg, UnaryInterface_::permutation());
      }

      // Non-permuting tile evaluation functions
      // The compiler will select the correct functions based on the consumability
      // of the arguments.

      template <bool C>
      static typename madness::enable_if_c<!C, result_type>::type
      no_permute(const Arg& arg) {
        return TiledArray::neg(arg);
      }

      template <bool C>
      static typename madness::enable_if_c<C, result_type>::type
      no_permute(Arg& arg) {
        return TiledArray::neg_to(arg);
      }

    }; // class Neg

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_NEG_H__INCLUDED
