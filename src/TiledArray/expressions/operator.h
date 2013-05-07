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

#ifndef TILEDARRAY_EXPRESSIONS_OPERATOR_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_OPERATOR_H__INCLUDED

#include <TiledArray/expressions/tsr_add.h>
#include <TiledArray/expressions/tsr_subt.h>
#include <TiledArray/expressions/tsr_neg.h>
#include <TiledArray/expressions/tsr_cont.h>
#include <TiledArray/expressions/tsr_mult.h>
#include <TiledArray/expressions/scal_tsr_add.h>
#include <TiledArray/expressions/scal_tsr_subt.h>
#include <TiledArray/expressions/scal_tsr_neg.h>
#include <TiledArray/expressions/scal_tsr_cont.h>
#include <TiledArray/expressions/scal_tsr_mult.h>

namespace TiledArray {
  namespace expressions {

    // Create addition expression
    template <typename ExpLeft, typename ExpRight>
    TsrAdd<ExpLeft, ExpRight> operator+(const Base<ExpLeft>& left, const Base<ExpRight>& right) {
      return TsrAdd<ExpLeft, ExpRight>(left.derived(), right.derived());
    }


    // Create subtraction expression
    template <typename ExpLeft, typename ExpRight>
    TsrSubt<ExpLeft, ExpRight> operator-(const Base<ExpLeft>& left, const Base<ExpRight>& right) {
      return TsrSubt<ExpLeft, ExpRight>(left.derived(), right.derived());
    }


    // Create negate expression
    template <typename Exp>
    TsrNeg<Exp> operator-(const Base<Exp>& arg) {
      return TsrNeg<Exp>(arg.derived());
    }

    // Create multiply expressions
    template <typename ExpLeft, typename ExpRight>
    TsrMult<ExpLeft, ExpRight> multiply(const Base<ExpLeft>& left, const Base<ExpRight>& right) {
      return TsrMult<ExpLeft, ExpRight>(left.derived(), right.derived());
    }

    // Create multiply expressions
    template <typename ExpLeft, typename ExpRight>
    TsrMult<ExpLeft, ExpRight> multiply(const Base<ExpLeft>& left, const Base<ExpRight>& right) {
      return TsrMult<ExpLeft, ExpRight>(left.derived(), right.derived());
    }

    // Create scaled addition expressions
    template <typename Factor, typename Exp>
    typename madness::enable_if<TiledArray::detail::is_numeric<Factor>, typename Exp::scaled_expression_type>::type
    operator*(const Factor factor, const Base<Exp>& exp) {
      return typename Exp::scaled_expression_type(exp.derived(), factor);
    }

    template <typename Exp, typename Factor>
    typename madness::enable_if<TiledArray::detail::is_numeric<Factor>, typename Exp::scaled_expression_type>::type
    operator*(const Base<Exp>& exp, const Factor& factor) {
      return typename Exp::scaled_expression_type(exp.derived(), factor);
    }

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_OPERATOR_H__INCLUDED
