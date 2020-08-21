/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2019  Virginia Tech
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
 *  Edward Valeev
 *  Department of Chemistry, Virginia Tech
 *
 *  eigen.h
 *  Oct 1, 2019
 *
 */

#ifndef TILEDARRAY_EXTERNAL_EIGEN_H__INCLUDED
#define TILEDARRAY_EXTERNAL_EIGEN_H__INCLUDED

//
// Configure Eigen and include Eigen/Core, all dependencies should include this
// before including any Eigen headers
//

#include <TiledArray/config.h>
#include <madness/config.h>

TILEDARRAY_PRAGMA_GCC(diagnostic push)
TILEDARRAY_PRAGMA_GCC(system_header)

////////////////////////////////////////////////
// this duplicates TiledArray_Eigen definitions
#if HAVE_INTEL_MKL
#ifndef EIGEN_USE_MKL  // strangely, defining EIGEN_USE_MKL_ALL does not imply
                       // EIGEN_USE_MKL
#define EIGEN_USE_MKL 1
#endif
#ifndef EIGEN_USE_MKL_ALL
#define EIGEN_USE_MKL_ALL 1
#endif
#else

//# ifndef EIGEN_USE_BLAS
//#  define EIGEN_USE_BLAS 1
//# endif

#ifdef TILEDARRAY_EIGEN_USE_LAPACKE
#ifndef EIGEN_USE_LAPACKE
#define EIGEN_USE_LAPACKE 1
#endif
#ifndef EIGEN_USE_LAPACKE_STRICT
#define EIGEN_USE_LAPACKE_STRICT 1
#endif

#endif
#endif
  
/////////////////////////////////////////////////
// define lapacke types to prevent inclusion of complex.h by
// Eigen/src/misc/lapacke.h
#include <madness/tensor/lapacke_types.h>
#include <Eigen/Core>

#if defined(EIGEN_USE_LAPACKE) || defined(EIGEN_USE_LAPACKE_STRICT)
#if !EIGEN_VERSION_AT_LEAST(3,3,7)
#error "Eigen3 < 3.3.7 with LAPACKE enabled may give wrong eigenvalue results"
#error "Either turn off TILEDARRAY_EIGEN_USE_LAPACKE or use Eigen3 3.3.7"
#endif
#endif

TILEDARRAY_PRAGMA_GCC(diagnostic pop)

#endif  // TILEDARRAY_EXTERNAL_EIGEN_H__INCLUDED
