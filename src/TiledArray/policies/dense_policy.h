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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  dense_array.h
 *  Jul 9, 2013
 *
 */

#ifndef TILEDARRAY_DENSE_ARRAY_H__INCLUDED
#define TILEDARRAY_DENSE_ARRAY_H__INCLUDED

#include <TiledArray/tiled_range.h>
#include <TiledArray/pmap/pmap.h>
#include <TiledArray/dense_shape.h>

namespace TiledArray {

  // Forward declarations
  class DenseShape;

  class DensePolicy {
    typedef TiledRange trange_type;
    typedef typename trange_type::range_type range_type;
    typedef typename range_type::size_type size_type;
    typedef DenseShape shape_type;
    typedef Pmap pmap_interface;
  }; // class DensePolicy

} // namespace TiledArray

#endif // TILEDARRAY_DENSE_ARRAY_H__INCLUDED
