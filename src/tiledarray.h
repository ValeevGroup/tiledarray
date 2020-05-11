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

#include <tiledarray_fwd.h>

#include <TiledArray/external/madness.h>
#include <TiledArray/initialize.h>

// Array class
#include <TiledArray/tensor.h>
#include <TiledArray/tile.h>

// Array policy classes
#include <TiledArray/policies/dense_policy.h>
#include <TiledArray/policies/sparse_policy.h>

// Expression functionality
#include <TiledArray/conversions/dense_to_sparse.h>
#include <TiledArray/conversions/foreach.h>
#include <TiledArray/conversions/make_array.h>
#include <TiledArray/conversions/retile.h>
#include <TiledArray/conversions/sparse_to_dense.h>
#include <TiledArray/conversions/to_new_tile_type.h>
#include <TiledArray/conversions/truncate.h>
#include <TiledArray/expressions/scal_expr.h>
#include <TiledArray/expressions/tsr_expr.h>

// Special Arrays
#include <TiledArray/special/diagonal_array.h>

// Process maps
#include <TiledArray/pmap/hash_pmap.h>
#include <TiledArray/pmap/replicated_pmap.h>

// Utility functionality
#include <TiledArray/conversions/eigen.h>

// Linear algebra
#include <TiledArray/algebra/conjgrad.h>
#include <TiledArray/dist_array.h>

#ifdef TILEDARRAY_HAS_SCALAPACK
#include <TiledArray/conversions/scalapack.h>
#endif

#endif  // TILEDARRAY_H__INCLUDED
