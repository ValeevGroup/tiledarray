/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2018  Virginia Tech
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
 *  Karl Pierce
 *  Department of Chemistry, Virginia Tech
 *
 *  make_trange1.cpp
 *  June 7, 2022
 *
 */

#ifndef TILEDARRAY_COMPUTE_TRANGE1__H
#define TILEDARRAY_COMPUTE_TRANGE1__H

#include "tiledarray.h"

namespace TiledArray {

/// this creates "uniform" TiledRange1 object using same logic as assumed in
/// vector_of_array.h
inline TiledArray::TiledRange1 kmp5_compute_trange1(
    std::size_t range_size, std::size_t target_block_size) {
  if (range_size > 0) {
    std::size_t nblocks =
        (range_size + target_block_size - 1) / target_block_size;
    auto dv = std::div((int)(range_size + nblocks - 1), (int)nblocks);
    auto avg_block_size = dv.quot - 1, num_avg_plus_one = dv.rem + 1;
    std::vector<std::size_t> hashmarks;
    hashmarks.reserve(nblocks + 1);
    auto block_counter = 0;
    for (auto i = 0; i < num_avg_plus_one;
         ++i, block_counter += avg_block_size + 1) {
      hashmarks.push_back(block_counter);
    }
    for (auto i = num_avg_plus_one; i < nblocks;
         ++i, block_counter += avg_block_size) {
      hashmarks.push_back(block_counter);
    }
    hashmarks.push_back(range_size);
    return TA::TiledRange1(hashmarks.begin(), hashmarks.end());
  } else
    return TA::TiledRange1{};
}

}  // namespace TiledArray

#endif  // TILEDARRAY_COMPUTE_TRANGE1__H
