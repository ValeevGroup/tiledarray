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
 *  scal_neg.h
 *  May 9, 2013
 *
 */

#ifndef TILEDARRAY_TILE_OP_SCAL_NEG_H__INCLUDED
#define TILEDARRAY_TILE_OP_SCAL_NEG_H__INCLUDED

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
    class ScalNeg {
    public:
      typedef ScalNeg<Result, Arg, Consumable> ScalNeg_; ///< This object type
      typedef const Arg& argument_type; ///< The left-hand argument type
      typedef Result result_type; ///< The result tile type
      typedef typename TiledArray::detail::scalar_type<Result>::type scalar_type; ///< Scalar type

    private:
      Permutation perm_; ///< The result permutation
      scalar_type factor_; ///< Scaling factor

    public:
      /// Default constructor

      /// Construct an negation operation that does not permute the result tile
      /// and has a scaling factor of 1.
      ScalNeg() : perm_(), factor_(1) { }

      /// Permute constructor

      /// Construct an negation operation that scales the result tensor
      /// \param factor The scaling factor for the operation
      ScalNeg(const scalar_type factor) :
        perm_(), factor_(factor)
      { }

      /// Permute constructor

      /// Construct an negation operation that permutes and scales the result tensor.
      /// \param perm The permutation to apply to the result tile
      /// \param factor The scaling factor for the operation [default = 1]
      ScalNeg(const Permutation& perm, const scalar_type factor = scalar_type(1)) :
        perm_(perm), factor_(factor)
      { }

      /// Copy constructor

      /// \param other The negation operation object to be copied
      ScalNeg(const ScalNeg_& other) : perm_(other.perm_), factor_(other.factor_) { }

      /// Copy assignment

      /// \param other The addition operation object to be copied
      /// \return A reference to this object
      ScalNeg_& operator=(const ScalNeg_& other) {
        perm_ = other.perm_;
        factor_ = other.factor_;
        return *this;
      }

      /// Negate a tile and possibly permute

      /// \param arg The argument
      /// \return The sum and permutation of \c arg
      result_type operator()(argument_type arg) const {
        TiledArray::detail::ScalNegate<typename Result::value_type,
            typename Arg::value_type> op(factor_);

        result_type result;

        if(perm_.dim() > 1) {
          permute(result, perm_, arg, op);
        } else {
          result = result_type(arg.range(), arg.begin(), op);
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
    class ScalNeg<Result, Result, true> {
    public:
      typedef ScalNeg<Result, Result, true> ScalNeg_; ///< This object type
      typedef Result argument_type; ///< The argument type
      typedef Result result_type; ///< The result tile type
      typedef typename TiledArray::detail::scalar_type<Result>::type scalar_type; ///< Scalar type

    private:
      Permutation perm_; ///< The result permutation
      scalar_type factor_; ///< Scaling factor


    public:
      /// Default constructor

      /// Construct an negation operation that does not permute the result tile
      /// and has a scaling factor of 1.
      ScalNeg() : perm_(), factor_(1) { }

      /// Permute constructor

      /// Construct an negation operation that scales the result tensor
      /// \param factor The scaling factor for the operation
      ScalNeg(const scalar_type factor) :
        perm_(), factor_(factor)
      { }

      /// Permute constructor

      /// Construct an negation operation that permutes and scales the result tensor.
      /// \param perm The permutation to apply to the result tile
      /// \param factor The scaling factor for the operation [default = 1]
      ScalNeg(const Permutation& perm, const scalar_type factor = scalar_type(1)) :
        perm_(perm), factor_(factor)
      { }

      /// Copy constructor

      /// \param other The negation operation object to be copied
      ScalNeg(const ScalNeg_& other) : perm_(other.perm_), factor_(other.factor_) { }

      /// Copy assignment

      /// \param other The addition operation object to be copied
      /// \return A reference to this object
      ScalNeg_& operator=(const ScalNeg_& other) {
        perm_ = other.perm_;
        factor_ = other.factor_;
        return *this;
      }

      /// Negate a tile and possibly permute

      /// \param arg The argument
      /// \return The negative and permutation of \c arg
      result_type operator()(argument_type arg) const {
        if(perm_.dim() > 1) {
          TiledArray::detail::ScalNegate<typename Result::value_type,
              typename Result::value_type> op(factor_);
          result_type result;
          permute(result, perm_, arg, op);
          return result;
        }

        typedef Eigen::Matrix<typename Result::value_type, Eigen::Dynamic, 1> arg_matrix_type;
        typedef Eigen::Map<arg_matrix_type, Eigen::AutoAlign> arg_map_type;

        arg_map_type arg_map(arg.data(), arg.size());
        arg_map = -factor_ * arg_map;
        return arg;
      }
    }; // class Neg



  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_SCAL_NEG_H__INCLUDED
