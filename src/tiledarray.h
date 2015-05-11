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
 */

#ifndef TILEDARRAY_H__INCLUDED
#define TILEDARRAY_H__INCLUDED

#include <TiledArray/madness.h>

// Array class
#include <TiledArray/array.h>

// Tile type headers
#include <TiledArray/tensor.h>
#include <TiledArray/tile.h>

// Array policy classes
#include <TiledArray/policies/dense_policy.h>
#include <TiledArray/policies/sparse_policy.h>

// Expression functionality
#include <TiledArray/expressions/scal_expr.h>
#include <TiledArray/expressions/tsr_expr.h>
#include <TiledArray/expressions/scal_tsr_expr.h>

// Functions for modifying arrays
#include <TiledArray/conversions/sparse_to_dense.h>
#include <TiledArray/conversions/dense_to_sparse.h>
#include <TiledArray/conversions/to_new_tile_type.h>
#include <TiledArray/conversions/truncate.h>
#include <TiledArray/conversions/foreach.h>

// Process maps
#include <TiledArray/pmap/hash_pmap.h>
#include <TiledArray/pmap/replicated_pmap.h>

// Utility functionality
#include <TiledArray/conversions/eigen.h>

#ifdef TILEDARRAY_HAS_ELEMENTAL
#include <TiledArray/elemental.h>
#endif

#endif // TILEDARRAY_H__INCLUDED
