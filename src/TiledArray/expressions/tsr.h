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
    class Tsr : public ExpTsrBase<Tsr<A> > {
    private:
      typedef ExpTsrBase<Tsr<A> > base;

    public:
      typedef A array_type;
      typedef Tsr<A> tensor_type;
      typedef ScalTsr<A> scaled_tensor_type;

      Tsr(array_type& array, const VariableList& vars) :
        base(array, vars)
      { }

      Tsr(const ExpTsr<A>& tensor) :
        base(tensor)
      { }

      Tsr(const ExpScalTsr<A>& tensor) :
        base(tensor.array(), tensor.vars())
      { }

      // Pull base class functions into this class
      using base::derived;
      using base::array;
      using base::vars;

      Tsr<A>& operator=(Tsr<A>& other) {
        other.eval_to(*this);
        return *this;
      }

      template <typename D>
      Tsr<A>& operator=(const Base<D>& other) {
        other.eval_to(*this);
        return *this;
      }

      template <typename D>
      Tsr<A>& operator+=(const Base<D>& other) {
        Tsr<A> temp(tsr);
        TsrAdd<Tsr<A>, Derived> exp(temp, derived());
        exp.eval_to(tsr);
        return *this;
      }

      template <typename D>
      Tsr<A>& operator-=(const Base<D>& other) {
        Tsr<A> temp(tsr);
        TsrSubt<Tsr<A>, Derived> exp(temp, derived());
        exp.eval_to(tsr);
        return *this;
      }

      template <typename D>
      Tsr<A>& operator*=(const Base<D>& other) {
        Tsr<A> temp(tsr);
        TsrMult<Tsr<A>, Derived> exp(temp, derived());
        exp.eval_to(tsr);
        return *this;
      }

      operator array_type& () { return array_; }
      operator const array_type& () { return arra_; }

      template <typename A>
      void eval_to(Tsr<A>& tsr) {
      }

    }; // class ExpTsr

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_EXP_TSR_H__INCLUDED
