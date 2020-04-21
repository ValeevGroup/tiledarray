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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  tensor.h
 *  Jun 16, 2015
 *
 */

#ifndef TILEDARRAY_INDEX_PARSING_H__INCLUDED
#define TILEDARRAY_INDEX_PARSING_H__INCLUDED
#include "TiledArray/error.h"
#include <algorithm>
#include <string>

namespace TiledArray::detail {

inline bool is_tot_index(const std::string& idx) {
  return idx.find(";") != std::string::npos;
}

inline auto split(const std::string& s, char delim) {
  std::stringstream ss(s);
  std::string buffer;
  std::vector<std::string> tokens;
  while(std::getline(ss, buffer, delim))
    tokens.emplace_back(std::move(buffer));
  return tokens;
}

inline auto split_tot_index(const std::string& idx) {
  auto temp = split(idx, ';');
  TA_ASSERT(temp.size() == 2); // Must have outer and inner separation
  return std::make_pair(split(temp[0], ','), split(temp[1], ','));
}

}

#endif // TILEDARRAY_INDEX_PARSING_H__INCLUDED
