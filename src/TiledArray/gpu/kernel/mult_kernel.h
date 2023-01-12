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
 *  Aug 21, 2018
 *
 */

#ifndef TILEDARRAY_CUDA_MULT_KERNEL_H__INCLUDED
#define TILEDARRAY_CUDA_MULT_KERNEL_H__INCLUDED

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_CUDA

namespace TiledArray {

/// result[i] = result[i] * arg[i]
void mult_to_cuda_kernel(int *result, const int *arg, std::size_t n,
                         cudaStream_t stream, int device_id);

void mult_to_cuda_kernel(float *result, const float *arg, std::size_t n,
                         cudaStream_t stream, int device_id);

void mult_to_cuda_kernel(double *result, const double *arg, std::size_t n,
                         cudaStream_t stream, int device_id);

/// result[i] = arg1[i] * arg2[i]
void mult_cuda_kernel(int *result, const int *arg1, const int *arg2,
                      std::size_t n, cudaStream_t stream, int device_id);

void mult_cuda_kernel(float *result, const float *arg1, const float *arg2,
                      std::size_t n, cudaStream_t stream, int device_id);

void mult_cuda_kernel(double *result, const double *arg1, const double *arg2,
                      std::size_t n, cudaStream_t stream, int device_id);

}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_CUDA

#endif  // TILEDARRAY_CUDA_MULT_KERNEL_H__INCLUDED
