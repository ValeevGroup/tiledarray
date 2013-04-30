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

#ifndef TILEDARRAY_EXPRESSIONS_SCAL_BINARY_BASE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_SCAL_BINARY_BASE_H__INCLUDED

#include <TiledArray/expressions/binary_base.h>
#include <TiledArray/type_traits.h>

namespace TiledArray {
  namespace expressions {

    template <typename Derived>
    class ScalBinaryBase : BinaryBase<Derived> {
    private:
      typedef BinaryBase<Derived> base;

    public:
      typedef typename Derived::left_exp_type left_exp_type;
      typedef typename Derived::right_exp_type right_exp_type;
      typedef typename Derived::numeric_type numeric_type;
      typedef typename Derived::tensor_type tensor_type;
      typedef typename Derived::scaled_tensor_type scaled_tensor_type;

    private:
      numeric_type factor_; ///< expression scaling factor

    public:

      ScalBinaryBase(const BinaryBase<Derived>& other, const numeric_type factor) :
        base(other), factor_(factor)
      { }

      ScalBinaryBase(const ScalBinaryBase<Derived>& other, const numeric_type factor) :
        base(other), factor_(factor * other.factor())
      { }

      using base::derived;
      using base::left;
      using base::right;

      numeric_type factor() const { return factor_; }

    }; // class ExpScalBinaryBase

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_SCAL_BINARY_BASE_H__INCLUDED
