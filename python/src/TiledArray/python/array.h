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

#include "expression.h"
#include "python.h"
#include "range.h"
#include "trange.h"

#include <TiledArray/conversions/eigen.h>
#include <TiledArray/dist_array.h>
#include <string>
#include <vector>

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

template <typename T>
auto make_tile(py::buffer data) {
  auto shape = data.request().shape;
  py::array_t<T> tmp(shape);
  int result =
      py::detail::npy_api::get().PyArray_CopyInto_(tmp.ptr(), data.ptr());
  if (result < 0) throw py::error_already_set();
  return Tensor<T>(Range(shape), tmp.data());
}

// std::function<py::buffer(const Range&)>
template <class Array>
void init_tiles(Array &a, py::object f) {
  py::gil_scoped_release gil;
  auto op = [f](const Range &range) {
    Tensor<double> tile;
    {
      py::gil_scoped_acquire acquire;
      // py::print(f);
      py::buffer buffer = f(range);
      tile = make_tile<double>(buffer);
    }
    return tile;
  };
  a.init_tiles(op);
  a.world().gop.fence();
}

template <class Array, class... Trange>
std::shared_ptr<Array> make_array(const Trange &... args, World *world,
                                  py::object op) {
  if (!world) {
    world = &get_default_world();
  }
  auto array = std::make_shared<Array>(*world, trange::make_trange(args...));
  if (!op.is_none()) {
    init_tiles(*array, op);
  }
  return array;
}

template <class Array, class S = std::vector<size_t> >
inline S shape(const Array &a) {
  auto e = a.elements_range().extent();
  S shape(e.size());
  for (size_t i = 0; i < e.size(); ++i) {
    shape[i] = e[i];
  }
  // std::copy(e.begin(), e.end(), shape.begin());
  return shape;
}

template <class Array>
inline std::vector<std::vector<int64_t> > trange(const Array &a) {
  return trange::list(a.trange());
}

template <typename T>
py::buffer_info make_buffer_info(Tensor<T> &tile) {
  std::vector<size_t> strides;
  for (auto s : tile.range().stride()) {
    strides.push_back(sizeof(T) * s);
  }
  return py::buffer_info(
      tile.data(),                        /* Pointer to buffer */
      sizeof(T),                          /* Size of one scalar */
      py::format_descriptor<T>::format(), /* Python struct-style format
                                             descriptor */
      tile.range().rank(),                /* Number of dimensions */
      tile.range().extent(),              /* Buffer dimensions */
      strides /* Strides (in bytes) for each index */
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

template <class Array>
inline py::iterator make_iterator(Array &array) {
  return py::make_iterator(array.begin(), array.end());
}

template <class Array>
inline void setitem(Array &array, std::vector<int64_t> idx, py::buffer data) {
  auto tile = make_tile<double>(data);
  array.set(idx, tile);
}

template <class Array, class Idx>
inline py::array getitem(const Array &array, Idx idx) {
  auto tile = array.find(idx);
  if (!tile.probe()) {
    auto str = py::str(py::cast(idx));
    throw std::runtime_error("TArray[" + py::cast<std::string>(str) +
                             "] tile is not set");
  }
  return py::array(make_buffer_info(tile.get()));
}

template <class Array>
py::buffer_info make_buffer(Array &a) {
  typedef typename Array::scalar_type T;
  auto buffer = py::array_t<T>(shape(a));
  for (size_t i = 0; i < a.size(); ++i) {
    // if (a.is_zero(i)) continue;
    auto range = range::slice(a.trange().make_tile_range(i));
    // py::print(i,range);
    buffer[range] = getitem(a, i);
  }
  return buffer.request();
}

template <class Array>
using TileReference = typename Array::reference;

template <class Array>
py::array get_reference_data(TileReference<Array> &r) {
  auto tile = r.get();
  auto shape = tile.range().extent();
  auto base = py::cast(r);
  return py::array_t<double>(shape, tile.data(), base);
}

template <class Array>
void set_reference_data(TileReference<Array> &r, py::buffer data) {
  r = make_tile<double>(data);
}

template <class Array>
void make_array_class(py::object m, const char *name) {
  auto PyArray =
      py::class_<Array, std::shared_ptr<Array> >(m, name, py::buffer_protocol())
          .def(py::init())
          .def(py::init(&make_array<Array, std::vector<int64_t>, size_t>),
               py::arg("shape"), py::arg("block"), py::arg("world") = nullptr,
               py::arg("op") = py::none())
          .def(
              py::init(&array::make_array<Array,
                                          std::vector<std::vector<int64_t> > >),
              py::arg("trange"), py::arg("world") = nullptr,
              py::arg("op") = py::none())
          .def_buffer(&array::make_buffer<Array>)
          .def_property_readonly("world", &Array::world,
                                 py::return_value_policy::reference)
          .def_property_readonly("trange", &array::trange<Array>)
          .def_property_readonly("shape", &array::shape<Array, py::tuple>)
          .def("fill", &Array::fill, py::arg("value"),
               py::arg("skip_set") = false)
          .def("init", &array::init_tiles<Array>)
          // Array object needs be alive while iterator is used */
          .def("__iter__", &array::make_iterator<Array>, py::keep_alive<0, 1>())
          .def("__getitem__", &expression::getitem<Array>)
          .def("__setitem__", &expression::setitem<Array>)
          .def("__getitem__", &array::getitem<Array, std::vector<int64_t> >)
          .def("__setitem__", &array::setitem<Array>)
      // ;
      ;

  py::class_<typename Array::reference>(PyArray, "Reference",
                                        py::module_local())
      .def_property_readonly("index", &TileReference<Array>::index)
      .def_property_readonly("range", &TileReference<Array>::make_range)
      .def_property("data", &get_reference_data<Array>,
                    &set_reference_data<Array>);
}

void __init__(py::module m) {
  make_array_class<TArray<double> >(m, "TArray");
  make_array_class<TSpArray<double> >(m, "TSpArray");

  // py::class_< Tensor<double>, std::shared_ptr<Tensor<double> > >(m, "Tensor",
  // py::buffer_protocol())
  //   .def_buffer(&array::make_buffer_info<double>)
  //   ;
}

}  // namespace array
}  // namespace python
}  // namespace TiledArray

#endif  // TA_PYTHON_ARRAY_H
