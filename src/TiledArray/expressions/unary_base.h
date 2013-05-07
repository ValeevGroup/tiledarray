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

#ifndef TILEDARRAY_EXPRESSIONS_UNARY_BASE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_UNARY_BASE_H__INCLUDED

namespace TiledArray {
  namespace expressions {

    template <typename Derived>
    class UnaryBase : Base<Derived> {
    private:
      typedef Base<Derived> base;

    public:
      typedef typename Derived::arg_exp_type arg_exp_type;
      typedef typename Derived::tensor_type tensor_type;
      typedef typename Derived::scaled_tensor_type scaled_tensor_type;

    private:
      left_exp_type arg_;

    public:
      UnaryBase(const arg_exp_type& arg) :
        arg_(arg)
      { }

      UnaryBase(const UnaryBase<Derived>& other) :
        arg_(other.arg_)
      { }

      using base::derived;

      const arg_exp_type& arg() const { return arg_; }

    }; // class UnaryBase

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_UNARY_BASE_H__INCLUDED
