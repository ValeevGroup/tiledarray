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

#ifndef TA_PYTHON_ARRAY_H
#define TA_PYTHON_ARRAY_H

#include "python.h"
#include "expression.h"
#include "trange.h"
#include "range.h"

#include <TiledArray/dist_array.h>
#include <TiledArray/conversions/eigen.h>
#include <vector>
#include <string>

namespace TiledArray {
namespace python {
namespace array {

  // template<typename T>
  // py::array_t<T> make_tile(Tensor<T> &tile) {
  //   auto buffer_info = make_buffer_info(tile);
  //   return py::array_t<T>(
  //     buffer_info.shape,
  //     buffer_info.strides,
  //     (T*)buffer_info.ptr,
  //     py::cast(tile)
  //   );
  // }

  template<typename T>
  auto make_tile(py::buffer data) {
    auto shape = data.request().shape;
    py::array_t<T> tmp(shape);
    int result = py::detail::npy_api::get().PyArray_CopyInto_(tmp.ptr(), data.ptr());
    if (result < 0) throw py::error_already_set();
    return Tensor<T>(Range(shape), tmp.data());
  }

  // std::function<py::buffer(const Range&)>
  void init_tiles(TArray<double> &a, py::object f) {
    py::gil_scoped_release gil;
    a.init_tiles([f](const Range& range) {
      Tensor<double> tile;
      {
        py::gil_scoped_acquire acquire;
        //py::print(f);
        py::buffer buffer = f(range);
        tile = make_tile<double>(buffer);
      }
      return tile;
    });
  }

  template<class ... Trange>
  std::shared_ptr< TArray<double> > make_array(const Trange& ... args, World *world, py::object op) {
    if (!world) {
      world = &get_default_world();
    }
    auto array = std::make_shared< TArray<double> >(*world, trange::make_trange(args...));
    if (!op.is_none()) {
      init_tiles(*array, op);
    }
    return array;
  }

  template<class S = std::vector<size_t> >
  inline S shape(const TArray<double> &a) {
    auto e = a.elements_range().extent();
    S shape(e.size());
    for (size_t i = 0; i < e.size(); ++i) {
      shape[i] = e[i];
    }
    //std::copy(e.begin(), e.end(), shape.begin());
    return shape;
  }

  inline auto world(const TArray<double> &a) {
    return &a.world();
  }

  inline auto trange(const TArray<double> &a) {
    return trange::list(a.trange());
  }

  template<typename T>
  py::buffer_info make_buffer_info(Tensor<T> &tile) {
    std::vector<size_t> strides;
    for (auto s : tile.range().stride()) {
      strides.push_back(sizeof(T)*s);
    }
    return py::buffer_info(
      tile.data(),                              /* Pointer to buffer */
      sizeof(T),                                /* Size of one scalar */
      py::format_descriptor<T>::format(), /* Python struct-style format descriptor */
      tile.range().rank(),                                        /* Number of dimensions */
      tile.range().extent(),                             /* Buffer dimensions */
      strides                            /* Strides (in bytes) for each index */
    );
  }

  // template<class Array>
  // struct Iterator {
  //   std::shared_ptr<Array> array;
  //   typedef typename Array::iterator iterator;
  //   auto operator++() {
  //     return ++iterator;
  //   }
  //   auto operator*() {
  //     auto index = iterator.index();
  //     return std::make_tuple(
  //       std::vector<int64_t>(index.begin(), index.end()),
  //       py::array(
  //       )
  //     );
  //   }
  //   bool operator==(Iterator other) const {
  //     return this->it == other.it;
  //   }
  // };

  inline auto make_iterator(TArray<double> &array) {
    return py::make_iterator(array.begin(), array.end());
  }

  inline void setitem(TArray<double> &array, std::vector<int64_t> idx, py::buffer data) {
    auto tile = make_tile<double>(data);
    array.set(idx, tile);
  }

  template<class Idx>
  inline py::array getitem(const TArray<double> &array, Idx idx) {
    return py::array(make_buffer_info(array.find(idx).get()));
  }

  template<typename T>
  py::buffer_info make_buffer(TArray<T> &a) {
    auto buffer = py::array_t<T>(shape(a));
    for (size_t i = 0; i < a.size(); ++i) {
      //if (a.is_zero(i)) continue;
      auto range = range::slice(a.trange().make_tile_range(i));
      //py::print(i,range);
      buffer[range] = getitem(a,i);
    }
    return buffer.request();
  }


  typedef TArray<double>::reference TileReference;

  py::array get_reference_data(TileReference &r) {
    auto tile = r.get();
    auto shape = tile.range().extent();
    auto base = py::cast(r);
    return py::array_t<double>(shape, tile.data(), base);
  }

  void set_reference_data(TileReference &r, py::buffer data) {
    r = make_tile<double>(data);
  }

  void __init__(py::module m) {

    py::class_< TArray<double>::reference >(m, "TileReference", py::module_local())
      .def_property_readonly("index", &TileReference::index)
      .def_property_readonly("range", &TileReference::make_range)
      .def_property("data", &get_reference_data, &set_reference_data)
      ;

    py::class_< TArray<double>, std::shared_ptr<TArray<double> > >(m, "TArray", py::buffer_protocol())
      .def(py::init())
      .def(
        py::init(&make_array< std::vector<int64_t>, size_t >),
        py::arg("shape"),
        py::arg("block"),
        py::arg("world") = nullptr,
        py::arg("op") = py::none()
      )
      .def(
        py::init(&array::make_array< std::vector< std::vector<int64_t> > >),
        py::arg("trange"),
        py::arg("world") = nullptr,
        py::arg("op") = py::none()
      )
      .def_buffer(&array::make_buffer<double>)
      .def_property_readonly("world", &array::world, py::return_value_policy::reference)
      .def_property_readonly("trange", &array::trange)
      .def_property_readonly("shape", &array::shape<py::tuple>)
      .def("fill", &TArray<double>::fill, py::arg("value"), py::arg("skip_set") = false)
      .def("init", &array::init_tiles)
      .def("__iter__", &array::make_iterator, py::keep_alive<0, 1>()) // Keep object alive while iterator is used */
      .def("__getitem__", &expression::getitem)
      .def("__setitem__", &expression::setitem)
      .def("__getitem__", &array::getitem< std::vector<int64_t> >)
      .def("__setitem__", &array::setitem)
      ;

    // py::class_< Tensor<double>, std::shared_ptr<Tensor<double> > >(m, "Tensor",  py::buffer_protocol())
    //   .def_buffer(&array::make_buffer_info<double>)
    //   ;

  }


}
}
}

#endif // TA_PYTHON_ARRAY_H
