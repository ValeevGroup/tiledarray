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

#ifndef TILEDARRAY_EXPRESSIONS_EXP_TSR_BASE_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_EXP_TSR_BASE_H__INCLUDED

#include <TiledArray/expressions/exp_base.h>
#include <TiledArray/variable_list.h>

namespace TiledArray {
  namespace expressions {

    template <typename Derived>
    class ExpTsrBase : public ExpBase<Derived> {
    private:
      typedef ExpBase<Derived> base;

    public:
      typedef typename Derived::array_type array_type;
      typedef typename Derived::tensor_type tensor_type;
      typedef typename Derived::scaled_tensor_type scaled_tensor_type;

    private:
      array_type& array_;
      const VariableList vars_;

    public:

      ExpTsrBase(const array_type& array, const VariableList& vars) :
        array_(array), vars_(vars)
      { }

      ExpTsrBase(const ExpTsrBase<Derived>& tensor) :
        array_(tensor.array()), vars_(tensor.vars())
      { }

      template <typename D>
      ExpTsrBase(const ExpTsrBase<D>& tensor) :
        array_(tensor.array()), vars_(tensor.vars())
      { }

      using base::derived;

      array_type& array() const { return array_; }
      const VariableList& vars() const { return vars_; }

    }; // class ExpTsrBase

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_EXP_TSR_BASE_H__INCLUDED
