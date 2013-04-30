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

#ifndef TILEDARRAY_EXPRESSIONS_BINARY_BASE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_BINARY_BASE_H__INCLUDED

#include <TiledArray/expressions/base.h>

namespace TiledArray {
  namespace expressions {

    template <typename Derived>
    class BinaryBase : Base<Derived> {
    private:
      typedef Base<Derived> base;

    public:
      typedef typename Derived::left_exp_type left_exp_type;
      typedef typename Derived::right_exp_type right_exp_type;
      typedef typename Derived::tensor_type tensor_type;
      typedef typename Derived::scaled_tensor_type scaled_tensor_type;

    private:
      left_exp_type left_;
      right_exp_type right_;

    public:
      BinaryBase(const left_exp_type& left, const right_exp_type& right) :
        left_(left), right_(right)
      { }

      BinaryBase(const BinaryBase& other) :
        left_(other.left_), right_(other.right_)
      { }

      using base::derived;

      const left_exp_type& left() const { return left_; }
      const right_exp_type& right() const { return right_; }


    }; // class BinaryBase

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_BINARY_BASE_H__INCLUDED
