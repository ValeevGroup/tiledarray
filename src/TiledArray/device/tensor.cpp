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

namespace TiledArray::detail {

// Phase 1 sanity: confirm the is_device_tile specialization fires for the
// allocator alias and propagates through Tile<>.
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

// Explicit instantiations of UMTensor and its tile-op overloads land here in
// Phase 4 once the overload set is in place.

