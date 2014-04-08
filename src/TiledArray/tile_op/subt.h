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

#include <TiledArray/tile_op/permute.h>
#include <TiledArray/tile_op/binary_interface.h>

namespace TiledArray {
  namespace math {

    /// Tile subtraction operation

    /// This subtraction operation will compute the difference of two tiles and
    /// apply a permutation to the result tensor. If no permutation is given or
    /// the permutation is null, then the result is not permuted.
    /// \tparam Result The result type
    /// \tparam Left The left-hand argument type
    /// \tparam Right The right-hand argument type
    /// \tparam LeftConsumable A flag that is \c true when the left-hand
    /// argument is consumable.
    /// \tparam RightConsumable A flag that is \c true when the right-hand
    /// argument is consumable.
    template <typename Result, typename Left, typename Right, bool LeftConsumable,
        bool RightConsumable>
    class Subt : public BinaryInterface<Subt<Result, Left, Right, LeftConsumable,
        RightConsumable> >
    {
    public:
      typedef Subt<Result, Left, Right, LeftConsumable, RightConsumable> Subt_; ///< This object type
      typedef BinaryInterface<Subt_> BinaryInterface_; ///< Interface base class type
      typedef typename BinaryInterface_::first_argument_type first_argument_type; ///< The left-hand argument type
      typedef typename BinaryInterface_::second_argument_type second_argument_type; ///< The right-hand argument type
      typedef typename BinaryInterface_::zero_left_type zero_left_type; ///< Zero left-hand tile type
      typedef typename BinaryInterface_::zero_right_type zero_right_type; ///< Zero right-hand tile type
      typedef typename BinaryInterface_::result_type result_type; ///< The result tile type

      /// Default constructor

      /// Construct an subtraction operation that does not permute the result tile
      Subt() : BinaryInterface_() { }

      /// Permute constructor

      /// Construct an subtraction operation that permutes the result tensor
      /// \param perm The permutation to apply to the result tile
      explicit Subt(const Permutation& perm) : BinaryInterface_(perm) { }

      /// Copy constructor

      /// \param other The subtraction operation object to be copied
      Subt(const Subt_& other) : BinaryInterface_(other) { }

      /// Copy assignment

      /// \param other The subtraction operation object to be copied
      /// \return A reference to this object
      Subt_& operator=(const Subt_& other) {
        BinaryInterface_::operator =(other);
        return *this;
      }

      using BinaryInterface_::operator();

      // Permuting tile evaluation function
      // These operations cannot consume the argument tile since this operation
      // requires temporary storage space.

      result_type permute(first_argument_type first, second_argument_type second) const {
        return first.subt(second, BinaryInterface_::permutation());
      }

      result_type permute(zero_left_type, second_argument_type second) const {
        return second.neg(BinaryInterface_::permutation());
      }

      result_type permute(first_argument_type first, zero_right_type) const {
        return first.permute(BinaryInterface_::permutation());
      }

      // Non-permuting tile evaluation functions
      // The compiler will select the correct functions based on the consumability
      // of the arguments.

      template <bool LC, bool RC>
      static typename madness::enable_if_c<!(LC || RC), result_type>::type
      no_permute(first_argument_type first, second_argument_type second) {
        return first.subt(second);
      }

      template <bool LC, bool RC>
      static typename madness::enable_if_c<LC, result_type>::type
      no_permute(Left& first, second_argument_type second) {
        return first.subt_to(second);
      }

      template <bool LC, bool RC>
      static typename madness::enable_if_c<!LC && RC, result_type>::type
      no_permute(first_argument_type first, Right& second) {
        return second.subt_to(first, -1);
      }

      template <bool LC, bool RC>
      static typename madness::enable_if_c<!RC, result_type>::type
      no_permute(zero_left_type, second_argument_type second) {
        return second.neg();
      }

      template <bool LC, bool RC>
      static typename madness::enable_if_c<RC, result_type>::type
      no_permute(zero_left_type, Right& second) {
        return second.neg_to();
      }

      template <bool LC, bool RC>
      static typename madness::enable_if_c<!LC, result_type>::type
      no_permute(first_argument_type first, zero_right_type) {
        return first.clone();
      }

      template <bool LC, bool RC>
      static typename madness::enable_if_c<LC, result_type>::type
      no_permute(Left& first, zero_right_type) {
        return first;
      }

    }; // class Subt

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_SUBT_H__INCLUDED
