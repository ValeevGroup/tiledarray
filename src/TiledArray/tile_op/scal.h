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
 *  scal.h
 *  June 20, 2013
 *
 */

#ifndef TILEDARRAY_TILE_OP_SCAL_H__INCLUDED
#define TILEDARRAY_TILE_OP_SCAL_H__INCLUDED

#include <TiledArray/tile_op/unary_interface.h>

namespace TiledArray {
  namespace math {

    /// Tile scaling operation

    /// This scaling operation will scale the content a tile and apply a
    /// permutation to the result tensor. If no permutation is given or the
    /// permutation is null, then the result is not permuted.
    /// \tparam Result The result type
    /// \tparam Arg The argument type
    /// \tparam Consumable Flag that is \c true when Arg is consumable
    template <typename Result, typename Arg, bool Consumable>
    class Scal : public UnaryInterface<Scal<Result, Arg, Consumable> > {
    public:
      typedef Scal<Result, Arg, Consumable> Scal_; ///< This object type
      typedef UnaryInterface<Scal_> UnaryInterface_;
      typedef typename UnaryInterface_::argument_type argument_type; ///< The argument type
      typedef typename UnaryInterface_::result_type result_type; ///< The result tile type
      typedef typename TiledArray::detail::scalar_type<result_type>::type scalar_type; ///< Scalar type

    private:

      scalar_type factor_; ///< Scaling factor

    public:
      /// Default constructor

      /// Construct a scaling operation that does not permute the result tile
      /// and has a scaling factor of 1.
      Scal() : UnaryInterface_(), factor_(1) { }

      /// Permute constructor

      /// Construct a scaling operation that scales the result tensor
      /// \param factor The scaling factor for the operation
      explicit Scal(const scalar_type factor) :
        UnaryInterface_(), factor_(factor)
      { }

      /// Permute constructor

      /// Construct a scaling operation that permutes and scales the result tensor.
      /// \param perm The permutation to apply to the result tile
      /// \param factor The scaling factor for the operation
      Scal(const Permutation& perm, const scalar_type factor) :
        UnaryInterface_(perm), factor_(factor)
      { }

      /// Copy constructor

      /// \param other The scaling operation object to be copied
      Scal(const Scal_& other) : UnaryInterface_(other), factor_(other.factor_) { }

      /// Copy assignment

      /// \param other The scaling operation object to be copied
      /// \return A reference to this object
      Scal_& operator=(const Scal_& other) {
        UnaryInterface_::operator =(other);
        factor_ = other.factor_;
        return *this;
      }

      // Import interface from base class
      using UnaryInterface_::operator();
      // Permuting tile evaluation function
      // These operations cannot consume the argument tile since this operation
      // requires temporary storage space.

      result_type permute_op(const Arg& arg) const {
        using TiledArray::scale;
        return scale(arg, factor_, UnaryInterface_::permutation());
      }

      // Non-permuting tile evaluation functions
      // The compiler will select the correct functions based on the consumability
      // of the arguments.

      template <bool C>
      typename std::enable_if<!C, result_type>::type
      no_permute_op(const Arg& arg) const {
        using TiledArray::scale;
        return scale(arg, factor_);
      }

      template <bool C>
      typename std::enable_if<C, result_type>::type
      no_permute_op(Arg& arg) const {
        using TiledArray::scale_to;
        return scale_to(arg, factor_);
      }

    }; // class Scal

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_SCAL_H__INCLUDED
