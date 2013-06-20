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

#include <TiledArray/tile_op/permute.h>
#include <TiledArray/tensor.h>

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
    class ScalAdd {
    public:
      typedef ScalAdd<Result, Left, Right, false, false> ScalAdd_; ///< This object type
      typedef typename madness::if_c<LeftConsumable, Left&, const Left&>::type first_argument_type; ///< The left-hand argument type
      typedef typename madness::if_c<RightConsumable, Right&, const Right&>::type second_argument_type; ///< The right-hand argument type
      typedef const ZeroTensor<typename Left::value_type>& zero_left_type; ///< Zero left-hand tile type
      typedef const ZeroTensor<typename Right::value_type>& zero_right_type; ///< Zero right-hand tile type
      typedef Result result_type; ///< The result tile type
      typedef typename TiledArray::detail::scalar_type<Result>::type scalar_type; ///< Scalar type

    private:
      Permutation perm_; ///< The result permutation
      scalar_type factor_; ///< The scaling factor

      // Element operation functor types

      typedef ScalPlus<typename Left::value_type,
          typename Right::value_type, typename Result::value_type> scal_plus_op;
      typedef ScalPlusAssign<typename Left::value_type,
          typename Right::value_type> scal_plus_assign_left_op;
      typedef ScalPlusAssign<typename Right::value_type,
          typename Left::value_type> scal_plus_assign_right_op;
      typedef Scale<typename Left::value_type> scale_left_op;
      typedef Scale<typename Right::value_type> scale_right_op;

      // Permuting tile evaluation function
      // These operations cannot consume the argument tile since this operation
      // requires temporary storage space.

      result_type permute(first_argument_type first, second_argument_type second) const {
        result_type result;
        TiledArray::math::permute(result, perm_, first, second, scal_plus_op(factor_));
        return result;
      }

      result_type permute(zero_left_type, second_argument_type second) const {
        result_type result;
        TiledArray::math::permute(result, perm_, second, scale_right_op(factor_));
        return result;
      }

      result_type permute(first_argument_type first, zero_right_type) const {
        result_type result;
        TiledArray::math::permute(result, perm_, first, scale_left_op(factor_));
        return result;
      }

      // Non-permuting tile evaluation functions
      // The compiler will select the correct functions based on the consumability
      // of the arguments.

      template <bool LC, bool RC>
      typename madness::disable_if_c<(LC && std::is_same<Result, Left>::value) ||
          (RC && std::is_same<Result, Right>::value), result_type>::type
      no_permute(first_argument_type first, second_argument_type second) const {
        return result_type(first.range(), first.data(), second.data(),
            scal_plus_op(factor_));
      }

      template <bool LC, bool RC>
      typename madness::enable_if_c<LC && std::is_same<Result, Left>::value, result_type>::type
      no_permute(first_argument_type first, second_argument_type second) const {
        vector_assign(first.size(), second.data(), first.data(),
            scal_plus_assign_left_op(factor_));
        return first;
      }

      template <bool LC, bool RC>
      typename madness::enable_if_c<(RC && std::is_same<Result, Right>::value) &&
          (!(LC && std::is_same<Result, Left>::value)), result_type>::type
      no_permute(first_argument_type first, second_argument_type second) const {
        vector_assign(second.size(), first.data(), second.data(),
            scal_plus_assign_right_op(factor_));
        return second;
      }


      template <bool LC, bool RC>
      typename madness::disable_if_c<RC, result_type>::type
      no_permute(zero_left_type, second_argument_type second) const {
        return second * factor_;
      }

      template <bool LC, bool RC>
      typename madness::enable_if_c<RC, result_type>::type
      no_permute(zero_left_type, second_argument_type second) const {
        second *= factor_;
        return second;
      }

      template <bool LC, bool RC>
      typename madness::disable_if_c<LC, result_type>::type
      no_permute(first_argument_type first, zero_right_type) const {
        return first * factor_;
      }

      template <bool LC, bool RC>
      typename madness::enable_if_c<LC, result_type>::type
      no_permute(first_argument_type first, zero_right_type) const {
        first *= factor_;
        return first;
      }

    public:
      /// Default constructor

      /// Construct an addition operation that does not permute the result tile
      /// and has a scaling factor of 1.
      ScalAdd() : perm_(), factor_(1) { }

      /// Permute constructor

      /// Construct an addition operation that scales the result tensor
      /// \param factor The scaling factor for the operation [default = 1]
      ScalAdd(const scalar_type factor) :
        perm_(), factor_(factor)
      { }

      /// Permute constructor

      /// Construct an addition operation that permutes and scales the result tensor
      /// \param perm The permutation to apply to the result tile
      /// \param factor The scaling factor for the operation [default = 1]
      ScalAdd(const Permutation& perm, const scalar_type factor = scalar_type(1)) :
        perm_(perm), factor_(factor)
      { }

      /// Copy constructor

      /// \param other The addition operation object to be copied
      ScalAdd(const ScalAdd_& other) : perm_(other.perm_), factor_(other.factor_) { }

      /// Copy assignment

      /// \param other The addition operation object to be copied
      /// \return A reference to this object
      ScalAdd_& operator=(const ScalAdd_& other) {
        perm_ = other.perm_;
        factor_ = other.factor_;
        return *this;
      }

      /// Add and scale two non-zero tiles and possibly permute

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return The scaled sum and permutation of \c first and \c second
      result_type operator()(first_argument_type first, second_argument_type second) const {
        TA_ASSERT(first.range() == second.range());

        if(perm_.dim() > 1)
          return permute(first, second);

        return no_permute<LeftConsumable, RightConsumable>(first, second);
      }

      /// Add and scale a zero tile to a non-zero tiles and possibly permute

      /// \param second The right-hand argument
      /// \return The scaled sum and permutation of \c first and \c second
      result_type operator()(zero_left_type first, second_argument_type second) const {
        if(perm_.dim() > 1)
          return permute(first, second);

        return no_permute<LeftConsumable, RightConsumable>(first, second);
      }

      /// Add and scale a non-zero tiles to a zero tile and possibly permute

      /// \param first The left-hand argument
      /// \return The scaled sum and permutation of \c first and \c second
      result_type operator()(first_argument_type first, zero_right_type second) const {
        if(perm_.dim() > 1)
          return permute(first, second);

        return no_permute<LeftConsumable, RightConsumable>(first, second);
      }
    }; // class ScalAdd

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_SCAL_ADD_H__INCLUDED
