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

#ifndef TILEDARRAY_EXPRESSIONS_SCAL_TSR_SUBT_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_SCAL_TSR_SUBT_H__INCLUDED

#include <TiledArray/expressions/scal_binary_base.h>

namespace TiledArray {
  namespace expressions {

    template <typename ExpLeft, typename ExpRight>
    class ScalTsrSubt : public ScalBinaryBase<ScalTsrSubt<ExpLeft, ExpRight> > {
    private:
      typedef ExpBinaryBase<ExpScalTsrSubt<ExpLeft, ExpRight> > base;

    public:
      typedef ExpLeft left_exp_type;
      typedef ExpRight right_exp_type;
      typedef typename detail::scalar_type<ExpLeft>::type numeric_type;
      typedef TsrSubt<ExpLeft, ExpRight> tensor_type;
      typedef ScalTsrSubt<ExpLeft, ExpRight>  scaled_tensor_type;

    public:
      ScalTsrSubt(const TsrSubt<ExpLeft, ExpRight>& tensor, numeric_type factor) :
        base(tensor, factor)
      { }

      ScalTsrSubt(const ScalTsrSubt<ExpLeft, ExpRight>& tensor, numeric_type factor) :
        base(tensor, factor)
      { }

      using base::derived;
      using base::left;
      using base::right;
      using base::factor;

    }; // class ExpScalTsrSubt

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_SCAL_TSR_SUBT_H__INCLUDED
