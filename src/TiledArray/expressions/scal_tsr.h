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

#ifndef TILEDARRAY_EXPRESSIONS_EXP_SCAL_TSR_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_EXP_SCAL_TSR_H__INCLUDED

#include <TiledArray/expressions/exp_tsr_base.h>

namespace TiledArray {
  namespace expressions {

    template <typename A>
    class ExpScalTsr : public ExpTsrBase<ExpScalTsr<A> > {
    private:
      typedef ExpTsrBase<ExpScalTsr<A> > base;

    public:
      typedef A array_type;
      typedef typename base::numeric_type numeric_type;
      typedef ExpTsr<A> tensor_type;
      typedef ExpScalTsr<A> scaled_tensor_type;

    private:
      numeric_type factor_;

    public:

      template <typename T>
      ExpScalTsr(const ExpScalTsr<T>& tensor, const numeric_type factor) :
        base(tensor.tensor(), tensor.vars()), factor_(tensor.factor() * factor)
      { }

      template <typename T>
      ExpScalTsr(const ExpTsr<T>& tensor, const numeric_type factor) :
        base(tensor.tensor(), tensor.vars()), factor_(factor)
      { }

      // Pull base class functions into this class
      using base::derived;
      using base::array;
      using base::vars;

      numeric_type factor() const { return factor_; }
    }; // class ExpScalTsr

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_EXP_SCAL_TSR_H__INCLUDED
