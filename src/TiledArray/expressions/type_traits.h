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

#ifndef TILEDARRAY_EXPRESSIONS_TYPE_TRAITS_H__INCLUDED
#define TILEDARRAY_EXPRESSIONS_TYPE_TRAITS_H__INCLUDED

#include <TiledArray/expressions/exp_base.h>
#include <TiledArray/type_traits.h>

namespace TiledArray {
  namespace expressions {

    // Type traits for scaled expressions

    template <typename>
    struct is_scaled : public std::false_type { };

    template <typename A>
    struct is_scaled<ScalTsr<A> > : public std::true_type { };

    template <typename D>
    struct is_scaled<ScalBinaryBase<D> > : public std::true_type { };

    template <typename L, typename R>
    struct is_scaled<TsrAdd<L, R> > : public std::true_type { };

    template <typename L, typename R>
    struct is_scaled<ScalTsrSubt<L, R> > : public std::true_type { };

    template <typename L, typename R>
    struct is_scaled<ScalTsrCont<L, R> > : public std::true_type { };

    template <typename L, typename R>
    struct is_scaled<ScalTsrConv<L, R> > : public std::true_type { };

    template <typename D>
    struct is_scaled<ScalUnaryBase<D> > : public std::true_type { };

    template <typename A>
    struct is_scaled<ScalTsrNeg<A> > : public std::true_type { };


    // Type traits for tensors

    template <typename>
    struct is_tensor : public std::false_type { };

    template <typename D>
    struct is_tensor<TsrBase<D> > : public std::true_type { };

    template <typename A>
    struct is_tensor<Tsr<A> > : public std::true_type { };

    template <typename A>
    struct is_tensor<ScalTsr<A> > : public std::true_type { };


    // Type traits for consumable data

    template <typename>
    struct is_consumable : public std::false_type { };

    template <typename D>
    struct is_consumable<BinaryBase<D> > : public std::true_type { };

  }  // namespace expressions
} // namespace TiledArray

#endif // TILEDARRAY_EXPRESSIONS_TYPE_TRAITS_H__INCLUDED
