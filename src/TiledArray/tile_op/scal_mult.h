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
 *  scal_mult.h
 *  May 8, 2013
 *
 */

#ifndef TILEDARRAY_TILE_OP_SCAL_MULT_H__INCLUDED
#define TILEDARRAY_TILE_OP_SCAL_MULT_H__INCLUDED

#include <TiledArray/tile_op/permute.h>
#include <TiledArray/tensor.h>

namespace TiledArray {
  namespace math {

    /// Tile multiplication and scale operation

    /// This multiplication operation will multiply the content two tiles, then
    /// scale and apply a permutation to the result tensor. If no permutation is
    /// given or the permutation is null, then the result is not permuted.
    /// \tparam Result The result type
    /// \tparam Left The left-hand argument type
    /// \tparam Right The right-hand argument type
    /// \tparam LeftConsumable A flag that is \c true when the left-hand
    /// argument is consumable.
    /// \tparam RightConsumable A flag that is \c true when the right-hand
    /// argument is consumable.
    /// \tparam Enabler Used to disambiguate specialization
    template <typename Result, typename Left, typename Right, bool LeftConsumable,
        bool RightConsumable, typename Enabler = void>
    class ScalMult {
    public:
      typedef ScalMult<Result, Left, Right, false, false> ScalMult_; ///< This object type
      typedef const Left& first_argument_type; ///< The left-hand argument type
      typedef const Right& second_argument_type; ///< The right-hand argument type
      typedef Result result_type; ///< The result tile type
      typedef typename TiledArray::detail::scalar_type<Result>::type scalar_type; ///< Scalar type

    private:
      Permutation perm_; ///< The result permutation
      scalar_type factor_; ///< The scaling factor

    public:
      /// Default constructor

      /// Construct an multiplication operation that does not permute the result tile
      /// and has a scaling factor of 1.
      ScalMult() : perm_(), factor_(1) { }

      /// Permute constructor

      /// Construct an multiplication operation that scales the result tensor
      /// \param factor The scaling factor for the operation [default = 1]
      ScalMult(const scalar_type factor) :
        perm_(), factor_(factor)
      { }

      /// Permute constructor

      /// Construct an multiplication operation that permutes and scales the result tensor
      /// \param perm The permutation to apply to the result tile
      /// \param factor The scaling factor for the operation [default = 1]
      ScalMult(const Permutation& perm, const scalar_type factor = scalar_type(1)) :
        perm_(perm), factor_(factor)
      { }

      /// Copy constructor

      /// \param other The multiplication operation object to be copied
      ScalMult(const ScalMult_& other) : perm_(other.perm_), factor_(other.factor_) { }

      /// Copy assignment

      /// \param other The multiplication operation object to be copied
      /// \return A reference to this object
      ScalMult_& operator=(const ScalMult_& other) {
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

        const TiledArray::detail::ScalMultiplies<typename Left::value_type,
            typename Right::value_type, typename Result::value_type> op(factor_);

        result_type result;
        if(perm_.dim())
          permute(result, perm_, first, second, op);
        else
          result = result_type(first.range(), first.begin(), second.begin(), op);

        return result;
      }
    }; // class ScalMult

    /// Tile multiplication and scale operation

    /// This multiplication operation will multiply the content two tiles, then
    /// scale and apply a permutation to the result tensor. If no permutation is
    /// given or the permutation is null, then the result is not permuted.
    /// \tparam Result The result type
    /// \tparam Right The right-hand argument type
    /// \tparam RightConsumable A flag that is \c true when the right-hand
    /// argument is consumable.
    /// \note This specialization assumes the left hand tile is consumable
    template <typename Result, typename Right, bool RightConsumable>
    class ScalMult<Result, Result, Right, true, RightConsumable, void> {
    public:
      typedef ScalMult<Result, Result, Right, true, false> ScalMult_; ///< This object type
      typedef Result first_argument_type; ///< The left-hand argument type
      typedef const Right& second_argument_type; ///< The right-hand argument type
      typedef Result result_type; ///< The result tile type
      typedef typename TiledArray::detail::scalar_type<Result>::type scalar_type; ///< Scalar type

    private:
      Permutation perm_; ///< The result permutation
      scalar_type factor_; ///< The scaling factor

    public:
      /// Default constructor

      /// Construct an multiplication operation that does not permute the result tile
      /// and has a scaling factor of 1.
      ScalMult() : perm_(), factor_(1) { }

      /// Permute constructor

      /// Construct an multiplication operation that scales the result tensor
      /// \param factor The scaling factor for the operation [default = 1]
      ScalMult(const scalar_type factor) :
        perm_(), factor_(factor)
      { }

      /// Permute constructor

      /// Construct an multiplication operation that permutes and scales the result tensor
      /// \param perm The permutation to apply to the result tile
      /// \param factor The scaling factor for the operation [default = 1]
      ScalMult(const Permutation& perm, const scalar_type factor = scalar_type(1)) :
        perm_(perm), factor_(factor)
      { }

      /// Copy constructor

      /// \param other The multiplication operation object to be copied
      ScalMult(const ScalMult_& other) : perm_(other.perm_), factor_(other.factor_) { }

      /// Copy assignment

      /// \param other The multiplication operation object to be copied
      /// \return A reference to this object
      ScalMult_& operator=(const ScalMult_& other) {
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

        const TiledArray::detail::ScalMultiplies<typename Result::value_type,
            typename Right::value_type, typename Result::value_type> op(factor_);

        if(perm_.dim()) {
          result_type result;
          permute(result, perm_, first, second, op);
          return result;
        } else {
          const std::size_t end = first.size();
          for(std::size_t i = 0ul; i < end; ++i)
            first[i] = op(first[i], second[i]);
          return first;
        }
      }
    }; // class ScalMult


    /// Tile multiplication and scale operation

    /// This multiplication operation will multiply the content two tiles, then
    /// scale and apply a permutation to the result tensor. If no permutation is
    /// given or the permutation is null, then the result is not permuted.
    /// \tparam Result The result type
    /// \tparam Left The left-hand argument type
    /// \tparam LeftConsumable A flag that is \c true when the left-hand
    /// argument is consumable.
    /// \note This specialization assumes the right-hand tile is consumable
    template <typename Result, typename Left, bool LeftConsumable>
    class ScalMult<Result, Left, Result, LeftConsumable, true,
        typename madness::disable_if_c<LeftConsumable && std::is_same<Result, Left>::value>::type>
    {
    public:
      typedef ScalMult<Result, Left, Result, true, false> ScalMult_; ///< This object type
      typedef const Left& first_argument_type; ///< The left-hand argument type
      typedef Result second_argument_type; ///< The right-hand argument type
      typedef Result result_type; ///< The result tile type
      typedef typename TiledArray::detail::scalar_type<Result>::type scalar_type; ///< Scalar type

    private:
      Permutation perm_; ///< The result permutation
      scalar_type factor_; ///< The scaling factor

    public:
      /// Default constructor

      /// Construct an multiplication operation that does not permute the result tile
      /// and has a scaling factor of 1
      ScalMult() : perm_(), factor_(1) { }

      /// Permute constructor

      /// Construct an multiplication operation that scales the result tensor
      /// \param factor The scaling factor for the operation [default = 1]
      ScalMult(const scalar_type factor) :
        perm_(), factor_(factor)
      { }

      /// Permute constructor

      /// Construct an multiplication operation that permutes and scales the result tensor
      /// \param perm The permutation to apply to the result tile
      /// \param factor The scaling factor for the operation [default = 1]
      ScalMult(const Permutation& perm, const scalar_type factor = scalar_type(1)) :
        perm_(perm), factor_(factor)
      { }

      /// Copy constructor

      /// \param other The multiplication operation object to be copied
      ScalMult(const ScalMult_& other) : perm_(other.perm_), factor_(other.factor_) { }

      /// Copy assignment

      /// \param other The multiplication operation object to be copied
      /// \return A reference to this object
      ScalMult_& operator=(const ScalMult_& other) {
        perm_ = other.perm_;
        factor_ = other.factor_;
        return *this;
      }

      /// Multiply two non-zero tiles and possibly permute

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return The scaled sum and permutation of \c first and \c second
      result_type operator()(first_argument_type first, second_argument_type second) const {
        TA_ASSERT(first.range() == second.range());

        const TiledArray::detail::ScalMultiplies<typename Left::value_type,
            typename Result::value_type, typename Result::value_type> op(factor_);

        if(perm_.dim()) {
          result_type result;
          permute(result, perm_, first, second, op);
          return result;
        } else {
          const std::size_t end = first.size();
          for(std::size_t i = 0ul; i < end; ++i)
            second[i] = op(first[i], second[i]);
          return second;
        }
      }
    }; // class ScalMult

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_SCAL_MULT_H__INCLUDED
