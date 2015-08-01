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
 *  scal_add.cpp
 *  May 8, 2013
 *
 */

#ifndef TILEDARRAY_TILE_OP_SCAL_ADD_H__INCLUDED
#define TILEDARRAY_TILE_OP_SCAL_ADD_H__INCLUDED

#include <TiledArray/tile_op/binary_interface.h>

namespace TiledArray {
  namespace math {

    /// Tile addition and scale operation

    /// This addition operation will add the content two tiles, then scale and
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
    class ScalAdd : public BinaryInterface<ScalAdd<Result, Left, Right,
        LeftConsumable, RightConsumable> >
    {
    public:
      typedef ScalAdd<Result, Left, Right, LeftConsumable, RightConsumable> ScalAdd_; ///< This object type
      typedef BinaryInterface<ScalAdd_> BinaryInterface_; ///< Interface base class type
      typedef typename BinaryInterface_::first_argument_type first_argument_type; ///< The left-hand argument type
      typedef typename BinaryInterface_::second_argument_type second_argument_type; ///< The right-hand argument type
      typedef typename BinaryInterface_::result_type result_type; ///< The result tile type
      typedef typename TiledArray::detail::scalar_type<result_type>::type scalar_type; ///< Scalar type

    private:

      scalar_type factor_; ///< The scaling factor

    public:
      /// Default constructor

      /// Construct an addition operation that does not permute the result tile
      /// and has a scaling factor of 1.
      ScalAdd() : BinaryInterface_(), factor_(1) { }

      /// Permute constructor

      /// Construct an addition operation that scales the result tensor
      /// \param factor The scaling factor for the operation [default = 1]
      explicit ScalAdd(const scalar_type factor) :
        BinaryInterface_(), factor_(factor)
      { }

      /// Permute constructor

      /// Construct an addition operation that permutes and scales the result tensor
      /// \param perm The permutation to apply to the result tile
      /// \param factor The scaling factor for the operation [default = 1]
      explicit ScalAdd(const Permutation& perm, const scalar_type factor = scalar_type(1)) :
        BinaryInterface_(perm), factor_(factor)
      { }

      /// Copy constructor

      /// \param other The addition operation object to be copied
      ScalAdd(const ScalAdd_& other) : BinaryInterface_(other), factor_(other.factor_) { }

      /// Copy assignment

      /// \param other The addition operation object to be copied
      /// \return A reference to this object
      ScalAdd_& operator=(const ScalAdd_& other) {
        BinaryInterface_::operator =(other);
        factor_ = other.factor_;
        return *this;
      }

      using BinaryInterface_::operator();

      // Permuting tile evaluation function
      // These operations cannot consume the argument tile since this operation
      // requires temporary storage space.

      result_type permute_op(first_argument_type first, second_argument_type second) const {
        return add(first, second, factor_, BinaryInterface_::permutation());
      }

      result_type permute_op(ZeroTensor, second_argument_type second) const {
        return scale(second, factor_, BinaryInterface_::permutation());
      }

      result_type permute_op(first_argument_type first, ZeroTensor) const {
        return scale(first, factor_, BinaryInterface_::permutation());
      }

      // Non-permuting tile evaluation functions
      // The compiler will select the correct functions based on the consumability
      // of the arguments.

      template <bool LC, bool RC>
      typename std::enable_if<!(LC || RC), result_type>::type
      no_permute_op(first_argument_type first, second_argument_type second) const {
        return add(first, second, factor_);
      }

      template <bool LC, bool RC>
      typename std::enable_if<LC, result_type>::type
      no_permute_op(Left& first, second_argument_type second) const {
        return add_to(first, second, factor_);
      }

      template <bool LC, bool RC>
      typename std::enable_if<!LC && RC, result_type>::type
      no_permute_op(first_argument_type first, Right& second) const {
        return add_to(second, first, factor_);
      }


      template <bool LC, bool RC>
      typename std::enable_if<!RC, result_type>::type
      no_permute_op(ZeroTensor, second_argument_type second) const {
        return scale(second, factor_);
      }

      template <bool LC, bool RC>
      typename std::enable_if<RC, result_type>::type
      no_permute_op(ZeroTensor, Right& second) const {
        return scale_to(second, factor_);
      }

      template <bool LC, bool RC>
      typename std::enable_if<!LC, result_type>::type
      no_permute_op(first_argument_type first, ZeroTensor) const {
        return scale(first, factor_);
      }

      template <bool LC, bool RC>
      typename std::enable_if<LC, result_type>::type
      no_permute_op(Left& first, ZeroTensor) const {
        return scale_to(first, factor_);
      }

    }; // class ScalAdd

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_SCAL_ADD_H__INCLUDED
