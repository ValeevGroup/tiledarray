/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
 *  Copyright (C) 2020  Ames Laboratory
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
 *  Ryan Richards
 *  Ames Laboratory
 *
 *  annotation.h
 *  April 20, 2020
 *
 */

#ifndef TILEDARRAY_UTILS_ANNOTATION_H__INCLUDED
#define TILEDARRAY_UTILS_ANNOTATION_H__INCLUDED

#include <algorithm>
#include <string>
#include "TiledArray/error.h"

namespace TiledArray::detail {

inline std::string dummy_annotation(unsigned int ndim) {
  std::ostringstream oss;
  if (ndim > 0) oss << "i0";
  for (unsigned int d = 1; d < ndim; ++d) oss << ",i" << d;
  return oss.str();
}

inline bool is_tot_annotation(const std::string& annotation) {
  return annotation.find(";") != std::string::npos;
}

inline auto split(const std::string& s, char delim) {
  std::stringstream ss(s);
  std::string buffer;
  std::vector<std::string> tokens;
  while (std::getline(ss, buffer, delim))
    tokens.emplace_back(std::move(buffer));
  return tokens;
}

inline auto split_tot_annotation(const std::string& annotation) {
  auto temp = split(annotation, ';');
  TA_ASSERT(temp.size() == 2);  // Must have outer and inner separation
  return std::make_pair(split(temp[0], ','), split(temp[1], ','));
}

}  // namespace TiledArray::detail

#endif  // TILEDARRAY_UTILS_ANNOTATION_H__INCLUDED
