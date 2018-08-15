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
 *  Chong Peng
 *  Department of Chemistry, Virginia Tech
 *  Aug 15, 2018
 *
 */

#ifndef TILEDARRAY_EXTERNAL_CUTT_H__INCLUDED
#define TILEDARRAY_EXTERNAL_CUTT_H__INCLUDED

#include <TiledArray/config.h>


#ifdef TILEDARRAY_HAS_CUDA

#include <vector>
#include <algorithm>

#include <cutt.h>


namespace TiledArray {

/**
 * convert the extent of a Tensor from RowMajor to ColMajor
 *
 * @param extent  the extent of a RowMajor Tensor
 */
inline void extent_to_col_major(std::vector<int> &extent) {
  std::reverse(extent.begin(), extent.end());
}


/**
 * convert the permutation representation of a Tensor from RowMajor to ColMajor
 * @param perm  the permutation of a RowMajor Tensor
 */
inline void permutation_to_col_major(std::vector<int> &perm) {
  int size = perm.size();

  std::vector<int> col_major_perm(size, 0);

  for (int input_index = 0; input_index < size; input_index++) {
    int output_index = perm[input_index];

    // change input and output index to col major
    int input_index_col_major = size - input_index - 1;
    int output_index_col_major = size - output_index - 1;

    col_major_perm[input_index_col_major] = output_index_col_major;
  }

  perm.swap(col_major_perm);
}

} // namespace TiledArray

#endif //  TILEDARRAY_HAS_CUDA

#endif //TILEDARRAY_EXTERNAL_CUTT_H__INCLUDED
