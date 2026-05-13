/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2026  Virginia Tech
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
 *  Ajay Melekamburath
 *  Department of Chemistry, Virginia Tech
 */

#include <TiledArray/device/tensor.h>

#include <complex>

namespace TiledArray {

// Explicit instantiations of the UMTensor class for the standard numeric
// types. Without these, every TU including device/tensor.h would instantiate
// the full TA::Tensor<T, device_um_allocator<T>> class body (~3000 lines of
// templated members) -- the matching `extern template` declarations in
// device/tensor.h suppress that per-TU work and route consumers to the
// symbols defined here.
//
// The list mirrors `src/TiledArray/tensor/tensor.cpp`'s host-side set
// (double, float, complex variants), plus int/long which are cheap to
// instantiate and useful for index-tile use cases. BLAS-bearing free
// functions (`gemm`, `scale`, ...) are still header-defined templates --
// instantiating those for each numeric type would pull in the full
// BLAS++/librett surface here, and the compile-time saving from
// extern-templating them does not justify it. They get instantiated lazily
// in whichever TU actually calls them (typically the test or example TU).

template class Tensor<double, device_um_allocator<double>>;
template class Tensor<float, device_um_allocator<float>>;
template class Tensor<std::complex<double>,
                      device_um_allocator<std::complex<double>>>;
template class Tensor<std::complex<float>,
                      device_um_allocator<std::complex<float>>>;
template class Tensor<int, device_um_allocator<int>>;
template class Tensor<long, device_um_allocator<long>>;

}  // namespace TiledArray

namespace TiledArray::detail {

// Compile-time guarantees on the trait wiring. Run before the test suite
// (and even when BUILD_TESTING=OFF) so a regression here breaks the
// library build instead of being deferred to a test failure.
static_assert(is_device_tile_v<TiledArray::UMTensor<double>>,
              "UMTensor<double> must be tagged as a device tile");
static_assert(is_device_tile_v<TiledArray::UMTensor<float>>,
              "UMTensor<float> must be tagged as a device tile");
static_assert(
    is_device_tile_v<TiledArray::UMTensor<std::complex<double>>>,
    "UMTensor<std::complex<double>> must be tagged as a device tile");
static_assert(is_device_tile_v<TiledArray::Tile<TiledArray::UMTensor<double>>>,
              "Tile<UMTensor<double>> must propagate the device-tile tag");
static_assert(!is_device_tile_v<TiledArray::Tensor<double>>,
              "Plain Tensor<double> must not be tagged as a device tile");

}  // namespace TiledArray::detail

