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
    /// \tparam Enabler Used to disambiguate specialization
    template <typename Result, typename Left, typename Right, bool LeftConsumable,
        bool RightConsumable, typename Enabler = void>
    class ScalAdd {
    public:
      typedef ScalAdd<Result, Left, Right, false, false> ScalAdd_; ///< This object type
      typedef const Left& first_argument_type; ///< The left-hand argument type
      typedef const Right& second_argument_type; ///< The right-hand argument type
      typedef const ZeroTensor<typename Left::value_type>& zero_left_type; ///< Zero left-hand tile type
      typedef const ZeroTensor<typename Right::value_type>& zero_right_type; ///< Zero right-hand tile type
      typedef Result result_type; ///< The result tile type
      typedef typename TiledArray::detail::scalar_type<Result>::type scalar_type; ///< Scalar type

    private:
      Permutation perm_; ///< The result permutation
      scalar_type factor_; ///< The scaling factor

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

        const TiledArray::detail::ScalPlus<typename Left::value_type,
            typename Right::value_type, typename Result::value_type> op(factor_);

        result_type result;
        if(perm_.dim() > 1)
          permute(result, perm_, first, second, op);
        else
          result = result_type(first.range(), first.begin(), second.begin(), op);

        return result;
      }

      /// Add and scale a zero tile to a non-zero tiles and possibly permute

      /// \param second The right-hand argument
      /// \return The scaled sum and permutation of \c first and \c second
      result_type operator()(zero_left_type, second_argument_type second) const {
        const TiledArray::detail::Scale<typename Right::value_type> op(factor_);

        result_type result;
        if(perm_.dim() > 1)
          permute(result, perm_, second, op); // permute
        else
          result = result_type(second.range(), second.begin(), op); // no permute

        return result;
      }

      /// Add and scale a non-zero tiles to a zero tile and possibly permute

      /// \param first The left-hand argument
      /// \return The scaled sum and permutation of \c first and \c second
      result_type operator()(first_argument_type first, zero_right_type) const {
        const TiledArray::detail::Scale<typename Left::value_type> op(factor_);

        result_type result;
        if(perm_.dim() > 1)
          permute(result, perm_, first, op); // permute
        else
          result = result_type(first.range(), first.begin(), op); // No permute

        return result;
      }
    }; // class ScalAdd

    /// Tile addition and scale operation

    /// This addition operation will add the content two tiles, then scale and
    /// apply a permutation to the result tensor. If no permutation is given or
    /// the permutation is null, then the result is not permuted.
    /// \tparam Result The result type
    /// \tparam Right The right-hand argument type
    /// \tparam RightConsumable A flag that is \c true when the right-hand
    /// argument is consumable.
    /// \note This specialization assumes the left hand tile is consumable
    template <typename Result, typename Right, bool RightConsumable>
    class ScalAdd<Result, Result, Right, true, RightConsumable, void> {
    public:
      typedef ScalAdd<Result, Result, Right, true, false> ScalAdd_; ///< This object type
      typedef Result first_argument_type; ///< The left-hand argument type
      typedef const Right& second_argument_type; ///< The right-hand argument type
      typedef const ZeroTensor<typename Result::value_type>& zero_left_type; ///< Zero left-hand tile type
      typedef const ZeroTensor<typename Right::value_type>& zero_right_type; ///< Zero right-hand tile type
      typedef Result result_type; ///< The result tile type
      typedef typename TiledArray::detail::scalar_type<Result>::type scalar_type; ///< Scalar type

    private:
      Permutation perm_; ///< The result permutation
      scalar_type factor_; ///< The scaling factor

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

        const TiledArray::detail::ScalPlus<typename Result::value_type,
            typename Right::value_type, typename Result::value_type> op(factor_);

        if(perm_.dim() > 1) {
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

      /// Add and scale a zero tile to a non-zero tiles and possibly permute

      /// \param second The right-hand argument
      /// \return The scaled sum and permutation of \c first and \c second
      result_type operator()(zero_left_type, second_argument_type second) const {
        const TiledArray::detail::Scale<typename Right::value_type> op(factor_);

        result_type result;
        if(perm_.dim() > 1)
          permute(result, perm_, second, op); // permute
        else
          result = result_type(second.range(), second.begin(), op); // no permute

        return result;
      }

      /// Add and scale a non-zero tiles to a zero tile and possibly permute

      /// \param first The left-hand argument
      /// \return The scaled sum and permutation of \c first and \c second
      result_type operator()(first_argument_type first, zero_right_type) const {
        if(perm_.dim() > 1) {
          const TiledArray::detail::Scale<typename Result::value_type> op(factor_);

          result_type result;
          permute(result, perm_, first, op); // permute
          return result;
        }

        first *= factor_;
        return first; // No permute
      }
    }; // class ScalAdd


    /// Tile addition and scale operation

    /// This addition operation will add the content two tiles, then scale and
    /// apply a permutation to the result tensor. If no permutation is given or
    /// the permutation is null, then the result is not permuted.
    /// \tparam Result The result type
    /// \tparam Left The left-hand argument type
    /// \tparam LeftConsumable A flag that is \c true when the left-hand
    /// argument is consumable.
    /// \note This specialization assumes the right-hand tile is consumable
    template <typename Result, typename Left, bool LeftConsumable>
    class ScalAdd<Result, Left, Result, LeftConsumable, true,
        typename madness::disable_if_c<LeftConsumable && std::is_same<Result, Left>::value>::type>
    {
    public:
      typedef ScalAdd<Result, Left, Result, true, false> ScalAdd_; ///< This object type
      typedef const Left& first_argument_type; ///< The left-hand argument type
      typedef Result second_argument_type; ///< The right-hand argument type
      typedef const ZeroTensor<typename Left::value_type>& zero_left_type; ///< Zero left-hand tile type
      typedef const ZeroTensor<typename Result::value_type>& zero_right_type; ///< Zero right-hand tile type
      typedef Result result_type; ///< The result tile type
      typedef typename TiledArray::detail::scalar_type<Result>::type scalar_type; ///< Scalar type

    private:
      Permutation perm_; ///< The result permutation
      scalar_type factor_; ///< The scaling factor

    public:
      /// Default constructor

      /// Construct an addition operation that does not permute the result tile
      /// and has a scaling factor of 1
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

      /// Add two non-zero tiles and possibly permute

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return The scaled sum and permutation of \c first and \c second
      result_type operator()(first_argument_type first, second_argument_type second) const {
        TA_ASSERT(first.range() == second.range());

        const TiledArray::detail::ScalPlus<typename Left::value_type,
            typename Result::value_type, typename Result::value_type> op(factor_);

        if(perm_.dim() > 1) {
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

      /// Add and scale a zero tile to a non-zero tiles and possibly permute

      /// \param second The right-hand argument
      /// \return The scaled sum and permutation of \c first and \c second
      result_type operator()(zero_left_type, second_argument_type second) const {
        if(perm_.dim() > 1) {
          const TiledArray::detail::Scale<typename Result::value_type> op(factor_);

          result_type result;
          permute(result, perm_, second, op); // permute
          return result;
        }

        second *= factor_;
        return second; // no permute
      }

      /// Add and scale a non-zero tiles to a zero tile and possibly permute

      /// \param first The left-hand argument
      /// \return The scaled sum and permutation of \c first and \c second
      result_type operator()(first_argument_type first, zero_right_type) const {
        const TiledArray::detail::Scale<typename Left::value_type> op(factor_);

        result_type result;
        if(perm_.dim() > 1)
          permute(result, perm_, first, op); // permute
        else
          result = result_type(first.range(), first.begin(), op); // No permute

        return result;
      }
    }; // class ScalAdd

  }  // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_SCAL_ADD_H__INCLUDED
