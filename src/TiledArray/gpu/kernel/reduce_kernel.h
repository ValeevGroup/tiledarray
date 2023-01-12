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
 *  May 08, 2019
 *
 */

#ifndef TILEDARRAY_CUDA_REDUCE_KERNEL_H__INCLUDED
#define TILEDARRAY_CUDA_REDUCE_KERNEL_H__INCLUDED

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_CUDA

namespace TiledArray {

// foreach(i) result *= arg[i]
int product_cuda_kernel(const int *arg, std::size_t n, cudaStream_t stream,
                        int device_id);

float product_cuda_kernel(const float *arg, std::size_t n, cudaStream_t stream,
                          int device_id);

double product_cuda_kernel(const double *arg, std::size_t n,
                           cudaStream_t stream, int device_id);

// foreach(i) result += arg[i]
int sum_cuda_kernel(const int *arg, std::size_t n, cudaStream_t stream,
                    int device_id);

float sum_cuda_kernel(const float *arg, std::size_t n, cudaStream_t stream,
                      int device_id);

double sum_cuda_kernel(const double *arg, std::size_t n, cudaStream_t stream,
                       int device_id);

// foreach(i) result = max(result, arg[i])
int max_cuda_kernel(const int *arg, std::size_t n, cudaStream_t stream,
                    int device_id);

float max_cuda_kernel(const float *arg, std::size_t n, cudaStream_t stream,
                      int device_id);

double max_cuda_kernel(const double *arg, std::size_t n, cudaStream_t stream,
                       int device_id);

// foreach(i) result = min(result, arg[i])
int min_cuda_kernel(const int *arg, std::size_t n, cudaStream_t stream,
                    int device_id);

float min_cuda_kernel(const float *arg, std::size_t n, cudaStream_t stream,
                      int device_id);

double min_cuda_kernel(const double *arg, std::size_t n, cudaStream_t stream,
                       int device_id);

// foreach(i) result = max(result, abs(arg[i]))
int absmax_cuda_kernel(const int *arg, std::size_t n, cudaStream_t stream,
                       int device_id);

float absmax_cuda_kernel(const float *arg, std::size_t n, cudaStream_t stream,
                         int device_id);

double absmax_cuda_kernel(const double *arg, std::size_t n, cudaStream_t stream,
                          int device_id);

// foreach(i) result = min(result, abs(arg[i]))
int absmin_cuda_kernel(const int *arg, std::size_t n, cudaStream_t stream,
                       int device_id);

float absmin_cuda_kernel(const float *arg, std::size_t n, cudaStream_t stream,
                         int device_id);

double absmin_cuda_kernel(const double *arg, std::size_t n, cudaStream_t stream,
                          int device_id);

}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_CUDA

#endif  // TILEDARRAY_CUDA_REDUCE_KERNEL_H__INCLUDED
