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
 *  scal_subt.h
 *  May 8, 2013
 *
 */

#ifndef TILEDARRAY_TILE_OP_SCAL_SUBT_H__INCLUDED
#define TILEDARRAY_TILE_OP_SCAL_SUBT_H__INCLUDED

#include <TiledArray/tile_op/binary_interface.h>

namespace TiledArray {
  namespace math {

    /// Tile subtraction and scale operation

    /// This subtraction operation will add the content two tiles, then scale and
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
    class ScalSubt : public BinaryInterface<ScalSubt<Result, Left, Right,
        LeftConsumable, RightConsumable> >
    {
    public:
      typedef ScalSubt<Result, Left, Right, LeftConsumable, RightConsumable> ScalSubt_; ///< This object type
      typedef BinaryInterface<ScalSubt_> BinaryInterface_; ///< Interface base class type
      typedef typename BinaryInterface_::first_argument_type first_argument_type; ///< The left-hand argument type
      typedef typename BinaryInterface_::second_argument_type second_argument_type; ///< The right-hand argument type
      typedef typename BinaryInterface_::zero_left_type zero_left_type; ///< Zero left-hand tile type
      typedef typename BinaryInterface_::zero_right_type zero_right_type; ///< Zero right-hand tile type
      typedef typename BinaryInterface_::result_type result_type; ///< The result tile type
      typedef typename TiledArray::detail::scalar_type<result_type>::type scalar_type; ///< Scalar type

    private:

      scalar_type factor_; ///< The scaling factor

    public:
      /// Default constructor

      /// Construct a subtraction operation that does not permute the result
      /// tile and has a scaling factor of 1.
      ScalSubt() : BinaryInterface_(), factor_(1) { }

      /// Permute constructor

      /// Construct a subtraction operation that scales the result tensor
      /// \param factor The scaling factor for the operation [default = 1]
      explicit ScalSubt(const scalar_type factor) :
        BinaryInterface_(), factor_(factor)
      { }

      /// Permute constructor

      /// Construct a subtraction operation that permutes and scales the result
      /// tile.
      /// \param perm The permutation to apply to the result tile
      /// \param factor The scaling factor for the operation [default = 1]
      explicit ScalSubt(const Permutation& perm, const scalar_type factor = scalar_type(1)) :
        BinaryInterface_(perm), factor_(factor)
      { }

      /// Copy constructor

      /// \param other The subtraction operation object to be copied
      ScalSubt(const ScalSubt_& other) : BinaryInterface_(other), factor_(other.factor_) { }

      /// Copy assignment

      /// \param other The subtraction operation object to be copied
      /// \return A reference to this object
      ScalSubt_& operator=(const ScalSubt_& other) {
        BinaryInterface_::operator =(other);
        factor_ = other.factor_;
        return *this;
      }

      using BinaryInterface_::operator();

      // Permuting tile evaluation function
      // These operations cannot consume the argument tile since this operation
      // requires temporary storage space.

      result_type permute(first_argument_type first, second_argument_type second) const {
        return first.subt(second, factor_, BinaryInterface_::permutation());
      }

      result_type permute(zero_left_type, second_argument_type second) const {
        return second.scale(-factor_, BinaryInterface_::permutation());
      }

      result_type permute(first_argument_type first, zero_right_type) const {
        return first.scale(factor_, BinaryInterface_::permutation());
      }

      // Non-permuting tile evaluation functions
      // The compiler will select the correct functions based on the consumability
      // of the arguments.

      template <bool LC, bool RC>
      typename madness::enable_if_c<!(LC || RC), result_type>::type
      no_permute(first_argument_type first, second_argument_type second) const {
        return first.subt(second, factor_);
      }

      template <bool LC, bool RC>
      typename madness::enable_if_c<LC, result_type>::type
      no_permute(Left& first, second_argument_type second) const {
        return first.subt_to(second, factor_);
      }

      template <bool LC, bool RC>
      typename madness::enable_if_c<!LC && RC, result_type>::type
      no_permute(first_argument_type first, Right& second) const {
        return second.subt_to(first, -factor_);
      }


      template <bool LC, bool RC>
      typename madness::enable_if_c<!RC, result_type>::type
      no_permute(zero_left_type, second_argument_type second) const {
        return second.scale(-factor_);
      }

      template <bool LC, bool RC>
      typename madness::enable_if_c<RC, result_type>::type
      no_permute(zero_left_type, Right& second) const {
        return second.scale_to(-factor_);
      }

      template <bool LC, bool RC>
      typename madness::enable_if_c<!LC, result_type>::type
      no_permute(first_argument_type first, zero_right_type) const {
        return first.scale(factor_);
      }

      template <bool LC, bool RC>
      typename madness::enable_if_c<LC, result_type>::type
      no_permute(Left& first, zero_right_type) const {
        return first.scale_to(factor_);
      }

    }; // class ScalSubt

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_SCAL_SUBT_H__INCLUDED
