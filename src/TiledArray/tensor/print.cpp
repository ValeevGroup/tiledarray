/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2025  Virginia Tech
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
 *  Eduard Valeyev
 *  Department of Chemistry, Virginia Tech
 *
 *  print.cpp
 *  Mar 14, 2025
 *
 */

#include <TiledArray/tensor/print.ipp>

namespace TiledArray {

namespace detail {

#define TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(T, C)                   \
  template void NDArrayPrinter::print<T, C>(                                  \
      const T* data, const std::size_t order,                                 \
      const Range1::index1_type* extents, const Range1::index1_type* strides, \
      std::basic_ostream<C>&, std::size_t);                                   \
  template std::basic_string<C> NDArrayPrinter::toString<T, C>(               \
      const T* data, const std::size_t order,                                 \
      const Range1::index1_type* extents, const Range1::index1_type* strides);

TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(double, char);
TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(double, wchar_t);
TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(std::complex<double>, char);
TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(std::complex<double>, wchar_t);
TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(float, char);
TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(float, wchar_t);
TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(std::complex<float>, char);
TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(std::complex<float>, wchar_t);
TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(int, char);
TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION(int, wchar_t);

#undef TILEDARRAY_MAKE_NDARRAY_PRINTER_INSTANTIATION

}  // namespace detail

}  // namespace TiledArray
