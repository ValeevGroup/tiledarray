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
 *  scal_shift.h
 *  June 7, 2015
 *
 */

#ifndef TILEDARRAY_TILE_OP_SCAL_SHIFT_H__INCLUDED
#define TILEDARRAY_TILE_OP_SCAL_SHIFT_H__INCLUDED

#include <TiledArray/tile_op/unary_interface.h>

namespace TiledArray {
  namespace math {

    /// Tile shift operation

    /// This no operation will shift the range of the tile and/or apply a
    /// permutation to the result tensor.
    /// \tparam Result The result type
    /// \tparam Arg The argument type
    /// \tparam Consumable Flag that is \c true when Arg is consumable
    template <typename Result, typename Arg, bool Consumable>
    class ScalShift : public UnaryInterface<ScalShift<Result, Arg, Consumable> >  {
    public:
      typedef ScalShift<Result, Arg, Consumable> ScalShift_; ///< This object type
      typedef UnaryInterface<ScalShift_> UnaryInterface_;
      typedef typename UnaryInterface_::argument_type argument_type; ///< The argument type
      typedef typename UnaryInterface_::result_type result_type; ///< The result tile type
      typedef typename TiledArray::detail::scalar_type<result_type>::type scalar_type; ///< Scalar type

    private:

      std::vector<long> range_shift_; ///< Range shift array
      scalar_type factor_; ///< Scaling factor

    public:

      // Compiler generated functions
      ScalShift() = delete;
      ScalShift(const ScalShift_&) = default;
      ScalShift(ScalShift_&&) = default;
      ~ScalShift() = default;
      ScalShift_& operator=(const ScalShift_&) = delete;
      ScalShift_& operator=(ScalShift_&&) = delete;

      /// Default constructor

      /// Construct a no operation that does not permute the result tile
      ScalShift(const std::vector<long>& range_shift, const scalar_type factor) :
        UnaryInterface_(), range_shift_(range_shift), factor_(factor)
      { }

      /// Permute constructor

      /// Construct a no operation that permutes the result tensor
      /// \param perm The permutation to apply to the result tile
      template <typename Index>
      ScalShift(const Index& range_shift, const Permutation& perm,
          const scalar_type factor) :
        UnaryInterface_(perm),
        range_shift_(TiledArray::detail::size(range_shift), 0l),
        factor_(factor)
      {
        TA_ASSERT(perm);
        TA_ASSERT(perm.dim() == TiledArray::detail::size(range_shift));
        TiledArray::detail::permute_array(perm, range_shift, range_shift_);
      }

      // Import interface from base class
      using UnaryInterface_::operator();

      // Permuting tile evaluation function
      // These operations cannot consume the argument tile since this operation
      // requires temporary storage space.

      result_type permute_op(const Arg& arg) const {
        result_type result = scale(arg, factor_, UnaryInterface_::permutation());
        return shift_to(result, range_shift_);
      }

      // Non-permuting tile evaluation functions
      // The compiler will select the correct functions based on the consumability
      // of the arguments.

      template <bool C>
      typename std::enable_if<!C, result_type>::type
      no_permute_op(const Arg& arg) const {
        result_type result = scale(arg, factor_);
        return shift_to(result, range_shift_);
      }

      template <bool C>
      typename std::enable_if<C, result_type>::type
      no_permute_op(Arg& arg) const {
        scale_to(arg, factor_);
        shift_to(arg, range_shift_);
        return arg;
      }

    }; // class Shift

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_SCAL_SHIFT_H__INCLUDED
