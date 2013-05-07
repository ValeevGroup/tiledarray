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

#ifndef TILEDARRAY_EXPRESSIONS_TSR_CONV_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_TSR_CONV_H__INCLUDED

namespace TiledArray {
  namespace expressions {

    template <typename ExpArg>
    class TsrConv : public UnaryBase<TsrConv<ExpArg> > {
    private:
      typedef UnaryBase<TsrConv<ExpArg> > base;

    public:
      typedef ExpArg arg_exp_type;
      typedef TsrConv<ExpArg> tensor_type;
      typedef ScalTsrConv<ExpArg> scaled_tensor_type;

      TsrConv(const arg_exp_type& arg) :
        base(arg)
      { }

      TsrConv(const TsrConv<ExpArg>& other) :
        base(other)
      { }

      using base::derived;
      using base::arg;

    }; // class TsrConv

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_TSR_CONV_H__INCLUDED
