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

#ifndef TILEDARRAY_EXPRESSIONS_SCAL_TSR_NEG_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_SCAL_TSR_NEG_H__INCLUDED

#include <TiledArray/expressions/scal_unary_base.h>

namespace TiledArray {
  namespace expressions {

    template <typename ExpArg>
    class ScalTsrNeg : public ScalUnaryBase<ScalTsrNeg<ExpArg> > {
    private:
      typedef UnaryBase<ScalTsrNeg<ExpArg> > base;

    public:
      typedef ExpArg arg_exp_type;
      typedef typename detail::scalar_type<ExpArg>::type numeric_type;
      typedef NegateTsr<ExpArg> tensor_type;
      typedef ScalNegateTsr<ExpArg> scaled_tensor_type;

      ScalTsrNeg(const TsrNeg<ExpArg>& arg, const numeric_type factor) :
        base(arg, factor)
      { }

      ScalTsrNeg(const ScalTsrNeg<ExpArg>& other, const numeric_type factor) :
        base(other, factor)
      { }

      using base::derived;
      using base::arg;
      using base::factor;

    }; // class ScalTsrNeg

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_SCAL_TSR_NEG_H__INCLUDED
