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

#ifndef TILEDARRAY_EXPRESSIONS_EXP_TSR_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_EXP_TSR_H__INCLUDED

#include <TiledArray/expressions/exp_tsr_base.h>

namespace TiledArray {
  namespace expressions {

    template <typename A>
    class ExpTsr : public ExpTsrBase<ExpTsr<A> > {
    private:
      typedef ExpTsrBase<ExpTsr<A> > base;

    public:
      typedef A array_type;
      typedef ExpTsr<A> tensor_type;
      typedef ExpScalTsr<A> scaled_tensor_type;

      ExpTsr(array_type& array, const VariableList& vars) :
        base(array, vars)
      { }

      ExpTsr(const ExpTsr<A>& tensor) :
        base(tensor)
      { }

      ExpTsr(const ExpScalTsr<A>& tensor) :
        base(tensor.array(), tensor.vars())
      { }

      // Pull base class functions into this class
      using base::derived;
      using base::array;
      using base::vars;

    }; // class ExpTsr

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_EXP_TSR_H__INCLUDED
