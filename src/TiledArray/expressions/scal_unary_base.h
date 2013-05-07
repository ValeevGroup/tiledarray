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

#ifndef TILEDARRAY_EXPRESSIONS_SCAL_UNARY_BASE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_SCAL_UNARY_BASE_H__INCLUDED

#include <TiledArray/expressions/unary_base.h>
#include <TiledArray/type_traits.h>

namespace TiledArray {
  namespace expressions {

    template <typename Derived>
    class ScalUnaryBase : UnaryBase<Derived> {
    private:
      typedef BinaryBase<Derived> base; ///< Base class type

    public:
      typedef typename Derived::arg_exp_type arg_exp_type; ///< Argument type
      typedef typename Derived::numeric_type numeric_type; ///< Numeric type

    private:
      numeric_type factor_; ///< expression scaling factor

    public:

      ScalUnaryBase(const UnaryBase<Derived>& other, numberic_type factor) :
        base(other), factor_(factor)
      { }

      ScalUnaryBase(const ScalUnaryBase<Derived>& other, numberic_type factor) :
        base(other), factor_(other.factor_ * factor)
      { }

      using base::derived;
      using base::arg;

      numeric_type factor() const { return factor_; }

    }; // class ScalUnaryBase

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_SCAL_UNARY_BASE_H__INCLUDED
