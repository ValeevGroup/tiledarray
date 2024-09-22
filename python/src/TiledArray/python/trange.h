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

#ifndef TA_PYTHON_TRANGE_H
#define TA_PYTHON_TRANGE_H

#include "python.h"

#include <TiledArray/tiled_range.h>
#include <string>
#include <vector>

namespace TiledArray {
namespace python {
namespace trange {

// template<class ... Args>
// inline TiledRange make_trange(Args ... args);

auto list(const TiledRange &trange) {
  std::vector<std::vector<int64_t> > v;
  for (auto &&tr1 : trange.data()) {
    auto it = tr1.begin();
    v.push_back({it->first});
    for (; it != tr1.end(); ++it) {
      v.back().push_back(it->second);
    }
  }
  return v;
}

inline TiledRange make_trange(std::vector<std::vector<int64_t> > trange) {
  std::vector<TiledRange1> trange1;
  for (auto tr : trange) {
    trange1.emplace_back(tr.begin(), tr.end());
  }
  return TiledRange(trange1.begin(), trange1.end());
}

// template<>
inline TiledRange make_trange(std::vector<int64_t> shape, size_t block) {
  std::vector<TiledRange1> trange1;
  for (size_t i = 0; i < shape.size(); ++i) {
    trange1.emplace_back(TiledRange1::make_uniform(shape[i], block));
  }
  return TiledRange(trange1.begin(), trange1.end());
}

void __init__(py::module m) {
  // py::class_<TiledRange>(m, "TiledRange")
  //   .def(py::init(&make_trange< std::vector< std::vector<int64_t> > >))
  //   ;

  // py::implicitly_convertible< std::vector< std::vector<int64_t> >,
  // TiledRange>();
}

}  // namespace trange
}  // namespace python
}  // namespace TiledArray

#endif  // TA_PYTHON_TRANGE_H
