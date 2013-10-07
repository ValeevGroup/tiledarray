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
 *  add.h
 *  May 7, 2013
 *
 */

#ifndef TILEDARRAY_TILE_OP_ADD_H__INCLUDED
#define TILEDARRAY_TILE_OP_ADD_H__INCLUDED

#include <TiledArray/tile_op/permute.h>
#include <TiledArray/tile_op/binary_interface.h>

namespace TiledArray {
  namespace math {

    /// Tile addition operation

    /// This addition operation will add the content two tiles and apply a
    /// permutation to the result tensor. If no permutation is given or the
    /// permutation is null, then the result is not permuted.
    /// \tparam Result The result type
    /// \tparam Left The left-hand argument type
    /// \tparam Right The right-hand argument type
    /// \tparam LeftConsumable A flag that is \c true when the left-hand
    /// argument is consumable.
    /// \tparam RightConsumable A flag that is \c true when the right-hand
    /// argument is consumable.
    template <typename Result, typename Left, typename Right, bool LeftConsumable,
        bool RightConsumable>
    class Add : public BinaryInterface<Add<Result, Left, Right, LeftConsumable,
        RightConsumable>, LeftConsumable, RightConsumable>
    {
    public:
      typedef Add<Result, Left, Right, LeftConsumable, RightConsumable> Add_; ///< This object type
      typedef BinaryInterface<Add_, LeftConsumable, RightConsumable> BinaryInterface_; ///< Interface base class type
      typedef typename BinaryInterface_::first_argument_type first_argument_type; ///< The left-hand argument type
      typedef typename BinaryInterface_::second_argument_type second_argument_type; ///< The right-hand argument type
      typedef typename BinaryInterface_::zero_left_type zero_left_type; ///< Zero left-hand tile type
      typedef typename BinaryInterface_::zero_right_type zero_right_type; ///< Zero right-hand tile type
      typedef typename BinaryInterface_::result_type result_type; ///< The result tile type

    private:
      Permutation perm_; ///< The result permutation

      // Make friends with base classes
      friend class BinaryInterface<Add_, LeftConsumable, RightConsumable>;
      friend class BinaryInterfaceBase<Add_, LeftConsumable, RightConsumable>;

      // Element operation functor types

      typedef Plus<typename Left::value_type, typename Right::value_type,
          typename Result::value_type> plus_op;


      // Permuting tile evaluation function
      // These operations cannot consume the argument tile since this operation
      // requires temporary storage space.

      result_type permute(const Left& first, const Right& second) const {
        result_type result;
        TiledArray::math::permute(result, perm_, first, second, plus_op());
        return result;
      }

      result_type permute(zero_left_type, const Right& second) const {
        result_type result;
        TiledArray::math::permute(result, perm_, second);
        return result;
      }

      result_type permute(const Left& first, zero_right_type) const {
        result_type result;
        TiledArray::math::permute(result, perm_, first);
        return result;
      }

      // Non-permuting tile evaluation functions
      // The compiler will select the correct functions based on the consumability
      // of the arguments.

      template <bool LC, bool RC>
      static typename madness::disable_if_c<(LC && std::is_same<Result, Left>::value) ||
          (RC && std::is_same<Result, Right>::value), result_type>::type
      no_permute(const Left& first, const Right& second) {
        return first + second;
      }

      template <bool LC, bool RC>
      static typename madness::enable_if_c<LC && std::is_same<Result, Left>::value, result_type>::type
      no_permute(Left& first, const Right& second) {
        first += second;
        return first;
      }

      template <bool LC, bool RC>
      static typename madness::enable_if_c<(RC && std::is_same<Result, Right>::value) &&
          (!(LC && std::is_same<Result, Left>::value)), result_type>::type
      no_permute(const Left& first, Right& second) {
        second += first;
        return second;
      }

      template <bool LC, bool RC>
      static typename madness::disable_if_c<RC, result_type>::type
      no_permute(zero_left_type, const Right& second) {
        return second.clone();
      }

      template <bool LC, bool RC>
      static typename madness::enable_if_c<RC, result_type>::type
      no_permute(zero_left_type, Right& second) {
        return second;
      }

      template <bool LC, bool RC>
      static typename madness::disable_if_c<LC, result_type>::type
      no_permute(const Left& first, zero_right_type) {
        return first.clone();
      }

      template <bool LC, bool RC>
      static typename madness::enable_if_c<LC, result_type>::type
      no_permute(Left& first, zero_right_type) {
        return first;
      }

    public:
      /// Default constructor

      /// Construct an addition operation that does not permute the result tile
      Add() : perm_() { }

      /// Permute constructor

      /// Construct an addition operation that permutes the result tensor
      /// \param perm The permutation to apply to the result tile
      Add(const Permutation& perm) : perm_(perm) { }

      /// Copy constructor

      /// \param other The addition operation object to be copied
      Add(const Add_& other) : perm_(other.perm_) { }

      /// Copy assignment

      /// \param other The addition operation object to be copied
      /// \return A reference to this object
      Add_& operator=(const Add_& other) {
        perm_ = other.perm_;
        return *this;
      }

      // Import interface from base class
      using BinaryInterface_::operator();

    }; // class Add

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_ADD_H__INCLUDED
