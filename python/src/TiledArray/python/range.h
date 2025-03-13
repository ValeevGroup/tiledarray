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

#ifndef TA_PYTHON_RANGE_H
#define TA_PYTHON_RANGE_H

#include "python.h"

#include <TiledArray/range.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace TiledArray {
namespace python {
namespace range {

  inline auto make_range(const std::vector< std::pair<int64_t,int64_t> >& r) {
    return Range(r);
  }

  inline size_t ndim(const Range &r) {
    return r.rank();
  }

  inline size_t size(const Range &r) {
    return r.volume();
  }

  inline auto shape(const Range &r) {
    return r.extent();
  }

  inline auto start(const Range &r) {
    return r.lobound();
  }

  inline auto stop(const Range &r) {
    return r.upbound();
  }

  inline auto slice(const Range &r) {
    py::list s;
    for (size_t i = 0; i < ndim(r); ++i) {
      s.append(
        py::slice(start(r)[i], stop(r)[i], 1)
      );
    }
    return py::tuple(s);
  }

  inline py::str str(const Range &r) {
    std::stringstream ss;
    ss << r;
    return py::str("Range" + ss.str());
  }

  void __init__(py::module m) {

    py::class_<Range>(m, "Range")
      .def(py::init(&make_range))
      .def("__str__", &str)
      .def_property_readonly("ndim", &ndim)
      .def_property_readonly("size", &size)
      .def_property_readonly("shape", &shape)
      .def_property_readonly("start", &start)
      .def_property_readonly("stop",  &stop)
    ;

  }

}
}
}

#endif /* TA_PYTHON_RANGE_H */
