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
 *  shift.h
 *  June 7, 2015
 *
 */

#ifndef TILEDARRAY_TILE_OP_SHIFT_H__INCLUDED
#define TILEDARRAY_TILE_OP_SHIFT_H__INCLUDED

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
    class Shift : public UnaryInterface<Shift<Result, Arg, Consumable> >  {
    public:
      typedef Shift<Result, Arg, Consumable> Shift_; ///< This object type
      typedef UnaryInterface<Shift_> UnaryInterface_;
      typedef typename UnaryInterface_::argument_type argument_type; ///< The argument type
      typedef typename UnaryInterface_::result_type result_type; ///< The result tile type

    private:

      std::vector<long> range_shift_;

    public:

      // Compiler generated functions
      Shift() = delete;
      Shift(const Shift_&) = default;
      Shift(Shift_&&) = default;
      ~Shift() = default;
      Shift& operator=(const Shift_&) = delete;
      Shift& operator=(Shift_&&) = delete;

      /// Default constructor

      /// Construct a no operation that does not permute the result tile
      Shift(const std::vector<long>& range_shift) :
        UnaryInterface_(), range_shift_(range_shift)
      { }

      /// Permute constructor

      /// Construct a no operation that permutes the result tensor
      /// \param perm The permutation to apply to the result tile
      template <typename Index>
      Shift(const Index& range_shift, const Permutation& perm) :
        UnaryInterface_(perm),
        range_shift_(TiledArray::detail::size(range_shift), 0l)
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
        using TiledArray::permute;
        using TiledArray::shift;
        result_type result = permute(arg, UnaryInterface_::permutation());
        return shift(result, range_shift_);
      }

      // Non-permuting tile evaluation functions
      // The compiler will select the correct functions based on the consumability
      // of the arguments.

      template <bool C>
      typename std::enable_if<!C, result_type>::type
      no_permute_op(const Arg& arg) const {
        using TiledArray::shift;
        return shift(arg, range_shift_);
      }

      template <bool C>
      typename std::enable_if<C, result_type>::type
      no_permute_op(Arg& arg) const {
        using TiledArray::shift_to;
        shift_to(arg, range_shift_);
        return arg;
      }

    }; // class Shift

  } // namespace math
} // namespace TiledArray

#endif // TILEDARRAY_TILE_OP_SHIFT_H__INCLUDED
