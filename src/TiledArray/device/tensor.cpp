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
// the full TA::Tensor<T, device_um_allocator<T>> class body.
// Mirrors the host-side set in tensor/tensor.cpp; paired with the
// `extern template` declarations in device/tensor.h.
template class Tensor<double, device_um_allocator<double>>;
template class Tensor<float, device_um_allocator<float>>;
template class Tensor<std::complex<double>,
                      device_um_allocator<std::complex<double>>>;
template class Tensor<std::complex<float>,
                      device_um_allocator<std::complex<float>>>;
template class Tensor<int, device_um_allocator<int>>;
template class Tensor<long, device_um_allocator<long>>;

}  // namespace TiledArray
