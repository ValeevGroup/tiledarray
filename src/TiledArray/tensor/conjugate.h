/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2015  Virginia Tech
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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  conjugate.h
 *  Oct 14, 2015
 *
 */

#ifndef TILEDARRAY_TENSOR_COMPLEX_H__INCLUDED
#define TILEDARRAY_TENSOR_COMPLEX_H__INCLUDED

#include <TiledArray/type_traits.h>

namespace TiledArray {
  namespace detail {

    template <typename T,
        typename std::enable_if<! is_complex<T>::value>::type* = nullptr>
    inline T conj(const T t) { return t; }

    template <typename T>
    inline std::complex<T> conj(const std::complex<T> t) { return std::conj(t); }

  } // namespace detail
} // namespace TiledArray

#endif // TILEDARRAY_TENSOR_COMPLEX_H__INCLUDED
