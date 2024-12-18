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

#ifndef TILEDARRAY_DEVICE_REDUCE_KERNEL_H__INCLUDED
#define TILEDARRAY_DEVICE_REDUCE_KERNEL_H__INCLUDED

#include <TiledArray/config.h>

#ifdef TILEDARRAY_HAS_DEVICE

#include <complex>

#include <TiledArray/external/device.h>

namespace TiledArray::device {

// foreach(i) result *= arg[i]
int product_kernel(const int* arg, std::size_t n, const Stream& stream);

float product_kernel(const float* arg, std::size_t n, const Stream& stream);

double product_kernel(const double* arg, std::size_t n, const Stream& stream);

std::complex<float> product_kernel(const std::complex<float>* arg,
                                   std::size_t n, const Stream& stream);

std::complex<double> product_kernel(const std::complex<double>* arg,
                                    std::size_t n, const Stream& stream);

// foreach(i) result += arg[i]
int sum_kernel(const int* arg, std::size_t n, const Stream& stream);

float sum_kernel(const float* arg, std::size_t n, const Stream& stream);

double sum_kernel(const double* arg, std::size_t n, const Stream& stream);

std::complex<float> sum_kernel(const std::complex<float>* arg, std::size_t n,
                               const Stream& stream);

std::complex<double> sum_kernel(const std::complex<double>* arg, std::size_t n,
                                const Stream& stream);

// foreach(i) result = max(result, arg[i])
int max_kernel(const int* arg, std::size_t n, const Stream& stream);

float max_kernel(const float* arg, std::size_t n, const Stream& stream);

double max_kernel(const double* arg, std::size_t n, const Stream& stream);

// foreach(i) result = min(result, arg[i])
int min_kernel(const int* arg, std::size_t n, const Stream& stream);

float min_kernel(const float* arg, std::size_t n, const Stream& stream);

double min_kernel(const double* arg, std::size_t n, const Stream& stream);

// foreach(i) result = max(result, abs(arg[i]))
int absmax_kernel(const int* arg, std::size_t n, const Stream& stream);

float absmax_kernel(const float* arg, std::size_t n, const Stream& stream);

double absmax_kernel(const double* arg, std::size_t n, const Stream& stream);

std::complex<float> absmax_kernel(const std::complex<float>* arg, std::size_t n,
                                  const Stream& stream);

std::complex<double> absmax_kernel(const std::complex<double>* arg,
                                   std::size_t n, const Stream& stream);

// foreach(i) result = min(result, abs(arg[i]))
int absmin_kernel(const int* arg, std::size_t n, const Stream& stream);

float absmin_kernel(const float* arg, std::size_t n, const Stream& stream);

double absmin_kernel(const double* arg, std::size_t n, const Stream& stream);

std::complex<float> absmin_kernel(const std::complex<float>* arg, std::size_t n,
                                  const Stream& stream);

std::complex<double> absmin_kernel(const std::complex<double>* arg,
                                   std::size_t n, const Stream& stream);

}  // namespace TiledArray::device

#endif  // TILEDARRAY_HAS_DEVICE

#endif  // TILEDARRAY_DEVICE_REDUCE_KERNEL_H__INCLUDED
