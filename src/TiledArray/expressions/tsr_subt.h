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

#ifndef TILEDARRAY_EXPRESSIONS_TSR_SUBT_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_TSR_SUBT_H__INCLUDED

namespace TiledArray {
  namespace expressions {

    template <typename ExpLeft, typename ExpRight>
    class TsrSubt : public BinaryBase<ExpTsrSubt<ExpLeft, ExpRight> > {
    private:
      typedef BinaryBase<ExpTsrSubt<ExpLeft, ExpRight> > base;

    public:
      typedef ExpLeft left_exp_type;
      typedef ExpRight right_exp_type;
      typedef TsrSubt<ExpLeft, ExpRight> tensor_type;
      typedef ScalTsrSubt<ExpLeft, ExpRight>  scaled_tensor_type;

      TsrSubt(const left_exp_type& left, const right_exp_type& right) :
        base(left, right)
      { }

      TsrSubt(const TsrSubt<ExpLeft, ExpRight>& other) :
        base(other)
      { }

      TsrSubt(const ScalTsrSubt<ExpLeft, ExpRight>& other) :
        base(other)
      { }

      using base::derived;
      using base::left;
      using base::right;

    }; // class ExpTsrSubt

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_TSR_SUBT_H__INCLUDED
