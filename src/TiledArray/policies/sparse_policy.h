/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  sparse_array.h
 *  Jul 9, 2013
 *
 */

#ifndef TILEDARRAY_SPARSE_ARRAY_H__INCLUDED
#define TILEDARRAY_SPARSE_ARRAY_H__INCLUDED

#include <TiledArray/error.h>
#include <TiledArray/sparse_shape.h>

namespace TiledArray {

  // Forward declarations
  template <typename> class Tensor;

  template <typename T>
  class SparsePolicy {
    typedef T element_type;
    typedef Tensor<T> tile_type;
    typedef Tensor<T> eval_type;
    typedef SparseShape shape_type;

    static shape_type default_shape() {
      TA_USER_ASSERT(false,
          "A SparseShape object must be provided to Array constructor.");
      return shape_type();
    }
  }; // class DensePolicy

} // namespace TiledArray

#endif // TILEDARRAY_SPARSE_ARRAY_H__INCLUDED
