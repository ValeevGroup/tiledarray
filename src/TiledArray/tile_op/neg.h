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

#include <TiledArray/tile_op/permute.h>
#include <TiledArray/tensor.h>
#include <Eigen/Core>

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
    class Neg {
    public:
      typedef Neg<Result, Arg, Consumable> Neg_; ///< This object type
      typedef const Arg& argument_type; ///< The left-hand argument type
      typedef Result result_type; ///< The result tile type

    private:
      Permutation perm_; ///< The result permutation

    public:
      /// Default constructor

      /// Construct a negation operation that does not permute the result tile
      Neg() : perm_() { }

      /// Permute constructor

      /// Construct a negation operation that permutes the result tensor
      /// \param perm The permutation to apply to the result tile
      Neg(const Permutation& perm) : perm_(perm) { }

      /// Copy constructor

      /// \param other The negation operation object to be copied
      Neg(const Neg_& other) : perm_(other.perm_) { }

      /// Copy assignment

      /// \param other The negation operation object to be copied
      /// \return A reference to this object
      Neg_& operator=(const Neg_& other) {
        perm_ = other.perm_;
        return *this;
      }

      /// Negate a tile and possibly permute

      /// \param arg The argument
      /// \return The sum and permutation of \c arg
      result_type operator()(argument_type arg) const {
        TiledArray::detail::Negate<typename Result::value_type,
            typename Arg::value_type> op;

        result_type result;
        if(perm_.dim() > 1) {
          permute(result, perm_, arg, op);
        } else {
          result = -arg;
        }

        return result;
      }
    }; // class Neg

    /// Tile negation operation

    /// This negation operation will negate the content a tile and apply a
    /// permutation to the result tensor. If no permutation is given or the
    /// permutation is null, then the result is not permuted.
    /// \tparam Result The result type
    /// \note This specialization assumes the tile is consumable
    template <typename Result>
    class Neg<Result, Result, true> {
    public:
      typedef Neg<Result, Result, true> Neg_; ///< This object type
      typedef Result argument_type; ///< The argument type
      typedef Result result_type; ///< The result tile type

    private:
      Permutation perm_; ///< The result permutation

      static inline void negate(typename argument_type::value_type& value) {
        value = -value;
      }

    public:
      /// Default constructor

      /// Construct a negation operation that does not permute the result tile
      Neg() : perm_() { }

      /// Permute constructor

      /// Construct a negation operation that permutes the result tensor
      /// \param perm The permutation to apply to the result tile
      Neg(const Permutation& perm) : perm_(perm) { }

      /// Copy constructor

      /// \param other The negation operation object to be copied
      Neg(const Neg_& other) : perm_(other.perm_) { }

      /// Copy assignment

      /// \param other The negation operation object to be copied
      /// \return A reference to this object
      Neg_& operator=(const Neg_& other) {
        perm_ = other.perm_;
        return *this;
      }

      /// Negate a tile and possibly permute

      /// \param arg The argument
      /// \return The negative and permutation of \c arg
      result_type operator()(argument_type arg) const {
        if(perm_.dim() > 1) {
          result_type result;
          permute(result, perm_, arg,
              TiledArray::detail::Negate<typename Result::value_type,
              typename Result::value_type>());
          return result;
        }

        math::vector_assign(arg.size(), arg.data(),
            TiledArray::detail::NegateAssign<typename Result::value_type>());
        return arg;
      }
    }; // class Neg



  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_NEG_H__INCLUDED
