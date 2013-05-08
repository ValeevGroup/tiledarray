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
    class Subt;

    /// Tile subtraction operation

    /// This subtraction operation will compute the difference of two tiles and
    /// apply a permutation to the result tensor. If no permutation is given or
    /// the permutation is null, then the result is not permuted.
    /// \tparam Result The result type
    /// \tparam Left The left-hand argument type
    /// \tparam Right The right-hand argument type
    template <typename Result, typename Left, typename Right>
    class Subt<Result, Left, Right, false, false> {
    public:
      typedef Subt<Result, Left, Right, false, false> Subt_; ///< This object type
      typedef const Left& first_argument_type; ///< The left-hand argument type
      typedef const Right& second_argument_type; ///< The right-hand argument type
      typedef const ZeroTensor<typename Left::value_type>& zero_left_type; ///< Zero left-hand tile type
      typedef const ZeroTensor<typename Right::value_type>& zero_right_type; ///< Zero right-hand tile type
      typedef Result result_type; ///< The result tile type

    private:
      Permutation perm_; ///< The result permutation

    public:
      /// Default constructor

      /// Construct an subtraction operation that does not permute the result tile
      Subt() : perm_() { }

      /// Permute constructor

      /// Construct an subtraction operation that permutes the result tensor
      /// \param perm The permutation to apply to the result tile
      Subt(const Permutation& perm) : perm_(perm) { }

      /// Copy constructor

      /// \param other The subtraction operation object to be copied
      Subt(const Subt_& other) : perm_(other.perm_) { }

      /// Copy assignment

      /// \param other The subtraction operation object to be copied
      /// \return A reference to this object
      Subt operator=(const Subt_& other) {
        perm_ = other.perm_;
        return *this;
      }

      /// Subtract two non-zero tiles and possibly permute

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return The difference and permutation of the first and second
      result_type operator()(first_argument_type first, second_argument_type second) const {
        TA_ASSERT(first.range() == second.range());

        TiledArray::detail::Minus<typename Left::value_type,
            typename Right::value_type, typename Result::value_type> op;

        result_type result;
        if(perm_.dim())
          permute(result, perm_, first, second, op);
        else
          result = result_type(first.range(), first.begin(), second.begin(), op);

        return result;
      }

      /// Subtract a zero tile from a non-zero tiles and possibly permute

      /// \param first The left-hand argument, a zero tile
      /// \param second The right-hand argument
      /// \return The difference and permutation of \c first and \c second
      result_type operator()(zero_left_type, second_argument_type second) const {
        TiledArray::detail::Negate<typename Right::value_type,
            typename Result::value_type> op;

        result_type result;
        if(perm_.dim())
          permute(result, perm_, second, op); // permute
        else
          result = result_type(second.range(), second.begin(), op); // no permute

        return result;
      }

      /// Subtract a non-zero tiles from a zero tile and possibly permute

      /// \param first The left-hand argument
      /// \param second The right-hand argument, a zero tile
      /// \return The difference and permutation of \c first and \c second
      result_type operator()(first_argument_type first, zero_right_type) const {
        result_type result;
        if(perm_.dim())
          permute(result, perm_, first); // permute
        else
          result = result_type(first.range(), first.begin()); // No permute

        return result;
      }
    }; // class Add

    /// Tile subtraction operation

    /// This subtraction operation will compute the difference of two tiles and
    /// apply a permutation to the result tensor. If no permutation is given or
    /// the permutation is null, then the result is not permuted.
    /// \tparam Result The result type
    /// \tparam Left The left-hand argument type
    /// \tparam Right The right-hand argument type
    /// \note This specialization assumes the left-hand tile is consumable
    template <typename Result, typename Left, typename Right, bool RightConsumable>
    class Subt<Result, Left, Right, true, RightConsumable> {
    public:
      typedef Subt<Result, Left, Right, true, false> Subt_; ///< This object type
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

      /// Construct an subtraction operation that does not permute the result tile
      Subt() : perm_() { }

      /// Permute constructor

      /// Construct an subtraction operation that permutes the result tensor
      /// \param perm The permutation to apply to the result tile
      Subt(const Permutation& perm) : perm_(perm) { }

      /// Copy constructor

      /// \param other The subtraction operation object to be copied
      Subt(const Subt_& other) : perm_(other.perm_) { }

      /// Copy assignment

      /// \param other The subtraction operation object to be copied
      /// \return A reference to this object
      Subt operator=(const Subt_& other) {
        perm_ = other.perm_;
        return *this;
      }

      /// Add two non-zero tiles and possibly permute

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return The difference and permutation of \c first and \c second
      result_type operator()(first_argument_type first, second_argument_type second) const {
        TA_ASSERT(first.range() == second.range());

        TiledArray::detail::Minus<typename Left::value_type,
            typename Right::value_type, typename Result::value_type> op;

        if(perm_.dim()) {
          result_type result;
          permute(result, perm_, first, second, op);
          return result;
        } else {
          first -= second;
          return first;
        }
      }

      /// Add a zero tile from a non-zero tiles and possibly permute

      /// \param first The left-hand argument, a zero tile
      /// \param second The right-hand argument
      /// \return The difference and permutation of \c first and \c second
      result_type operator()(zero_left_type, second_argument_type second) const {
        TiledArray::detail::Negate<typename Right::value_type,
            typename Result::value_type> op;

        result_type result;
        if(perm_.dim())
          permute(result, perm_, second, op); // permute
        else
          result = result_type(second.range(), second.begin(), op); // no permute

        return result;
      }

      /// Subtract a non-zero tiles from a zero tile and possibly permute

      /// \param first The left-hand argument
      /// \param second The right-hand argument, a zero tile
      /// \return The difference and permutation of \c first and \c second
      result_type operator()(first_argument_type first, zero_right_type) const {
        if(perm_.dim()) {
          result_type result;
          permute(result, perm_, first); // permute
          return result;
        } else
          return first; // No permute
      }
    }; // class Subt


    /// Tile subtraction operation

    /// This subtraction operation will compute the difference of two tiles and
    /// apply a permutation to the result tensor. If no permutation is given or
    /// the permutation is null, then the result is not permuted.
    /// \tparam Result The result type
    /// \tparam Left The left-hand argument type
    /// \tparam Right The right-hand argument type
    /// \note This specialization assumes the right-hand tile is consumable
    template <typename Result, typename Left, typename Right>
    class Subt<Result, Left, Right, false, true> {
    public:
      typedef Subt<Result, Left, Right, true, false> Subt_; ///< This object type
      typedef const Left& first_argument_type; ///< The left-hand argument type
      typedef Right second_argument_type; ///< The right-hand argument type
      typedef const ZeroTensor<typename Left::value_type>& zero_left_type; ///< Zero left-hand tile type
      typedef const ZeroTensor<typename Right::value_type>& zero_right_type; ///< Zero right-hand tile type
      typedef Result result_type; ///< The result tile type

    private:
      Permutation perm_; ///< The result permutation

    public:
      /// Default constructor

      /// Construct an subtraction operation that does not permute the result tile
      Subt() : perm_() { }

      /// Permute constructor

      /// Construct a subtraction operation that permutes the result tensor
      /// \param perm The permutation to apply to the result tile
      Subt(const Permutation& perm) : perm_(perm) { }

      /// Copy constructor

      /// \param other The subtraction operation object to be copied
      Subt(const Subt_& other) : perm_(other.perm_) { }

      /// Copy assignment

      /// \param other The subtraction operation object to be copied
      /// \return A reference to this object
      Subt operator=(const Subt_& other) {
        perm_ = other.perm_;
        return *this;
      }

      /// Subtract two non-zero tiles and possibly permute

      /// \param first The left-hand argument
      /// \param second The right-hand argument
      /// \return The difference and permutation of \c first and \c second
      result_type operator()(first_argument_type first, second_argument_type second) const {
        TiledArray::detail::Minus<typename Left::value_type,
            typename Right::value_type, typename Result::value_type> op;

        TA_ASSERT(first.range() == second.range());
        if(perm_.dim()) {
          result_type result;
          permute(result, perm_, first, second, op);
          return result;
        } else {
          const std::size_t end = second.size();
          for(std::size_t i = 0ul; i < end; ++i)
            second[i] = op(first[i], second[i]);
          return second;
        }
      }

      /// Subtract a zero tile from a non-zero tiles and possibly permute

      /// \param first The left-hand argument, a zero tile
      /// \param second The right-hand argument
      /// \return The difference and permutation of \c first and \c second
      result_type operator()(zero_left_type, second_argument_type second) const {
        TiledArray::detail::Negate<typename Right::value_type,
            typename Result::value_type> op;

        result_type result;
        if(perm_.dim())
          permute(result, perm_, second, op); // permute
        else {
          const std::size_t end = second.size();
          for(std::size_t i = 0ul; i < end; ++i)
            second[i] = op(second[i]);
          return second; // no permute
        }

        return result;
      }

      /// Subtract a non-zero tiles from a zero tile and possibly permute

      /// \param first The left-hand argument
      /// \param second The right-hand argument, a zero tile
      /// \return The difference and permutation of \c first and \c second
      result_type operator()(first_argument_type first, zero_right_type) const {
        result_type result;
        if(perm_.dim())
          permute(result, perm_, first); // permute
        else
          result = result_type(first.range(), first.begin()); // No permute

        return result;
      }
    }; // class Subt

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_ADD_H__INCLUDED
