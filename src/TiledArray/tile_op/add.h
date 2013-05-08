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
 */

#ifndef TILEDARRAY_TILE_OP_ADD_H__INCLUDED
#define TILEDARRAY_TILE_OP_ADD_H__INCLUDED

#include <TiledArray/tile_op/permute.h>
#include <TiledArray/tensor.h>

namespace TiledArray {
  namespace math {

    // Forward declarations
    template <typename Result, typename Left, typename Right, bool LeftConsumable, bool RightConsumable>
    class Add;

    /// Tile addition operation

    /// This addition operation will add the content two tiles and apply a
    /// permutation to the result tensor. If no permutation is given or the
    /// permutation is null, then the result is not permuted.
    /// \tparam Result The result type
    /// \tparam Left The left-hand argument type
    /// \tparam Right The right-hand argument type
    template <typename Result, typename Left, typename Right>
    class Add<Result, Left, Right, false, false> {
    public:
      typedef Add<Result, Left, Right, false, false> Add_; ///< This object type
      typedef const Left& first_argument_type; ///< The left-hand argument type
      typedef const Right& second_argument_type; ///< The right-hand argument type
      typedef const ZeroTensor<typename Left::value_type>& zero_left_type; ///< Zero left-hand tile type
      typedef const ZeroTensor<typename Right::value_type>& zero_right_type; ///< Zero right-hand tile type
      typedef Result result_type; ///< The result tile type

    private:
      Permutation perm_; ///< The result permutation

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
      Add operator=(const Add_& other) {
        perm_ = other.perm_;
        return *this;
      }

      /// Add two non-zero tiles and possibly permute

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return The sum and permutation of the first and second
      result_type operator()(first_argument_type first, second_argument_type second) const {
        TA_ASSERT(first.range() == second.range());

        TiledArray::detail::Plus<typename Left::value_type,
            typename Right::value_type, typename Result::value_type> op;

        result_type result;
        if(perm_.dim())
          permute(result, perm_, first, second, op);
        else
          result = result_type(first.range(), first.begin(), second.begin(), op);

        return result;
      }

      /// Add a zero tile to a non-zero tiles and possibly permute

      /// \param first The left-hand argument, a zero tile
      /// \param second The right-hand argument
      /// \return The sum and permutation of the first and second
      result_type operator()(zero_left_type, second_argument_type second) const {
        result_type result;
        if(perm_.dim())
          permute(result, perm_, second); // permute
        else
          result = result_type(second.range(), second.begin()); // no permute

        return result;
      }

      /// Add a non-zero tiles to a zero tile and possibly permute

      /// \param first The left-hand argument
      /// \param second The right-hand argument, a zero tile
      /// \return The sum and permutation of the first and second
      result_type operator()(first_argument_type first, zero_right_type) const {
        result_type result;
        if(perm_.dim())
          permute(result, perm_, first); // permute
        else
          result = result_type(first.range(), first.begin()); // No permute

        return result;
      }
    }; // class Add

    /// Tile addition operation

    /// This addition operation will add the content two tiles and apply a
    /// permutation to the result tensor. If no permutation is given or the
    /// permutation is null, then the result is not permuted.
    /// \tparam Result The result type
    /// \tparam Left The left-hand argument type
    /// \tparam Right The right-hand argument type
    /// \note This specialization assumes the left-hand tile is consumable
    template <typename Result, typename Left, typename Right, bool RightConsumable>
    class Add<Result, Left, Right, true, RightConsumable> {
    public:
      typedef Add<Result, Left, Right, true, false> Add_; ///< This object type
      typedef Left first_argument_type; ///< The left-hand argument type
      typedef const Right& second_argument_type; ///< The right-hand argument type
      typedef const ZeroTensor<typename Left::value_type>& zero_left_type; ///< Zero left-hand tile type
      typedef const ZeroTensor<typename Right::value_type>& zero_right_type; ///< Zero right-hand tile type
      typedef Result result_type; ///< The result tile type
      typedef TiledArray::detail::Plus<typename Left::value_type,
          typename Right::value_type, typename Result::value_type> op_type; ///< The operation applied to the arguments

    private:
      Permutation perm_; ///< The result permutation

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
      Add operator=(const Add_& other) {
        perm_ = other.perm_;
        return *this;
      }

      /// Add two non-zero tiles and possibly permute

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return The sum and permutation of the first and second
      result_type operator()(first_argument_type first, second_argument_type second) const {
        TA_ASSERT(first.range() == second.range());

        TiledArray::detail::Plus<typename Left::value_type,
            typename Right::value_type, typename Result::value_type> op;

        if(perm_.dim()) {
          result_type result;
          permute(result, perm_, first, second, op);
          return result;
        } else {
          first += second;
          return first;
        }
      }

      /// Add a zero tile to a non-zero tiles and possibly permute

      /// \param first The left-hand argument, a zero tile
      /// \param second The right-hand argument
      /// \return The sum and permutation of the first and second
      result_type operator()(zero_left_type, second_argument_type second) const {
        result_type result;
        if(perm_.dim())
          permute(result, perm_, second); // permute
        else
          result = result_type(second.range(), second.begin()); // no permute

        return result;
      }

      /// Add a non-zero tiles to a zero tile and possibly permute

      /// \param first The left-hand argument
      /// \param second The right-hand argument, a zero tile
      /// \return The sum and permutation of the first and second
      result_type operator()(first_argument_type first, zero_right_type) const {
        if(perm_.dim()) {
          result_type result;
          permute(result, perm_, first); // permute
          return result;
        } else
          return first; // No permute
      }
    }; // class Add


    /// Tile addition operation

    /// This addition operation will add the content two tiles and apply a
    /// permutation to the result tensor. If no permutation is given or the
    /// permutation is null, then the result is not permuted.
    /// \tparam Result The result type
    /// \tparam Left The left-hand argument type
    /// \tparam Right The right-hand argument type
    /// \note This specialization assumes the right-hand tile is consumable
    template <typename Result, typename Left, typename Right>
    class Add<Result, Left, Right, false, true> {
    public:
      typedef Add<Result, Left, Right, true, false> Add_; ///< This object type
      typedef const Left& first_argument_type; ///< The left-hand argument type
      typedef Right second_argument_type; ///< The right-hand argument type
      typedef const ZeroTensor<typename Left::value_type>& zero_left_type; ///< Zero left-hand tile type
      typedef const ZeroTensor<typename Right::value_type>& zero_right_type; ///< Zero right-hand tile type
      typedef Result result_type; ///< The result tile type

    private:
      Permutation perm_; ///< The result permutation

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
      Add operator=(const Add_& other) {
        perm_ = other.perm_;
        return *this;
      }

      /// Add two non-zero tiles and possibly permute

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return The sum and permutation of the first and second
      result_type operator()(first_argument_type first, second_argument_type second) const {
        TA_ASSERT(first.range() == second.range());

        TiledArray::detail::Plus<typename Left::value_type,
            typename Right::value_type, typename Result::value_type> op;

        if(perm_.dim()) {
          result_type result;
          permute(result, perm_, first, second, op);
          return result;
        } else {
          second += first;
          return second;
        }
      }

      /// Add a zero tile to a non-zero tiles and possibly permute

      /// \param first The left-hand argument, a zero tile
      /// \param second The right-hand argument
      /// \return The sum and permutation of the first and second
      result_type operator()(zero_left_type, second_argument_type second) const {
        result_type result;
        if(perm_.dim())
          permute(result, perm_, second); // permute
        else
          return second; // no permute

        return result;
      }

      /// Add a non-zero tiles to a zero tile and possibly permute

      /// \param first The left-hand argument
      /// \param second The right-hand argument, a zero tile
      /// \return The sum and permutation of the first and second
      result_type operator()(first_argument_type first, zero_right_type) const {
        result_type result;
        if(perm_.dim())
          permute(result, perm_, first); // permute
        else
          result = result_type(first.range(), first.begin()); // No permute

        return result;
      }
    }; // class Add

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_ADD_H__INCLUDED
