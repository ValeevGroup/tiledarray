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
 *  May 8, 2019
 *
 */

#include <TiledArray/device/kernel/reduce_kernel.h>
#include <TiledArray/device/kernel/reduce_kernel_impl.h>


#ifdef TILEDARRAY_HAS_CUDA

namespace TiledArray {

// foreach(i) result *= arg[i]
int product_cuda_kernel(const int *arg, std::size_t n, cudaStream_t stream,
                        int device_id){
  return product_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

float product_cuda_kernel(const float *arg, std::size_t n, cudaStream_t stream,
                          int device_id){
  return product_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

double product_cuda_kernel(const double *arg, std::size_t n, cudaStream_t stream,
                           int device_id){

  return product_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

std::complex<float> product_cuda_kernel(const std::complex<float> *arg, std::size_t n, cudaStream_t stream,
                          int device_id){
  return product_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

std::complex<double> product_cuda_kernel(const std::complex<double> *arg, std::size_t n, cudaStream_t stream,
                           int device_id){

  return product_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

// foreach(i) result += arg[i]
int sum_cuda_kernel(const int *arg, std::size_t n, cudaStream_t stream,
                    int device_id){
  return sum_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

float sum_cuda_kernel(const float *arg, std::size_t n, cudaStream_t stream,
                      int device_id){
  return sum_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

double sum_cuda_kernel(const double *arg, std::size_t n, cudaStream_t stream,
                       int device_id){
  return sum_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

std::complex<float> sum_cuda_kernel(const std::complex<float> *arg, std::size_t n, cudaStream_t stream,
                      int device_id){
  return sum_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

std::complex<double> sum_cuda_kernel(const std::complex<double> *arg, std::size_t n, cudaStream_t stream,
                       int device_id){
  return sum_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

// foreach(i) result = max(result, arg[i])
int max_cuda_kernel(const int *arg, std::size_t n, cudaStream_t stream,
                    int device_id){
  return max_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

float max_cuda_kernel(const float *arg, std::size_t n, cudaStream_t stream,
                      int device_id){
  return max_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

double max_cuda_kernel(const double *arg, std::size_t n, cudaStream_t stream,
                       int device_id){
  return max_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

// foreach(i) result = min(result, arg[i])
int min_cuda_kernel(const int *arg, std::size_t n, cudaStream_t stream,
                    int device_id){
  return min_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

float min_cuda_kernel(const float *arg, std::size_t n, cudaStream_t stream,
                      int device_id){
  return min_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

double min_cuda_kernel(const double *arg, std::size_t n, cudaStream_t stream,
                       int device_id){
  return min_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

// foreach(i) result = max(result, abs(arg[i]))
int absmax_cuda_kernel(const int *arg, std::size_t n, cudaStream_t stream,
                       int device_id){
  return absmax_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

float absmax_cuda_kernel(const float *arg, std::size_t n, cudaStream_t stream,
                         int device_id){
  return absmax_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

double absmax_cuda_kernel(const double *arg, std::size_t n, cudaStream_t stream,
                          int device_id){
  return absmax_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

std::complex<float> absmax_cuda_kernel(const std::complex<float> *arg, std::size_t n, cudaStream_t stream,
                         int device_id){
  return absmax_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

std::complex<double> absmax_cuda_kernel(const std::complex<double> *arg, std::size_t n, cudaStream_t stream,
                          int device_id){
  return absmax_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

// foreach(i) result = min(result, abs(arg[i]))
int absmin_cuda_kernel(const int *arg, std::size_t n, cudaStream_t stream,
                       int device_id){
  return absmin_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

float absmin_cuda_kernel(const float *arg, std::size_t n, cudaStream_t stream,
                         int device_id){
  return absmin_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

double absmin_cuda_kernel(const double *arg, std::size_t n, cudaStream_t stream,
                          int device_id){
  return absmin_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

std::complex<float> absmin_cuda_kernel(const std::complex<float> *arg, std::size_t n, cudaStream_t stream,
                         int device_id){
  return absmin_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

std::complex<double> absmin_cuda_kernel(const std::complex<double> *arg, std::size_t n, cudaStream_t stream,
                          int device_id){
  return absmin_reduce_cuda_kernel_impl(arg, n, stream, device_id);
}

}  // namespace TiledArray

#endif // TILEDARRAY_HAS_CUDA
