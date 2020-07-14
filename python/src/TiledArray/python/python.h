/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2020  Virginia Tech
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

#ifndef TA_PYTHON_H
#define TA_PYTHON_H

#pragma GCC diagnostic ignored "-Wregister"

#include <TiledArray/util/vector.h>
#include <TiledArray/size_array.h>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#ifndef TA_PYTHON_MAX_EXPRESSION
#define TA_PYTHON_MAX_EXPRESSION 5
#endif

#if TA_PYTHON_MAX_EXPRESSION < 1
#error "TA_PYTHON_MAX_EXPRESSION must be > 0"
#endif

namespace py = pybind11;

namespace pybind11 {
namespace detail {

  template <typename T, std::size_t N>
  struct type_caster< TiledArray::container::svector<T,N> >
    : list_caster< TiledArray::container::svector<T,N>, T > { };

  template <typename T>
  struct type_caster< TiledArray::detail::SizeArray<T> >
    : list_caster< TiledArray::detail::SizeArray<T>, T > { };

}
}

#endif // TA_PYTHON_H
