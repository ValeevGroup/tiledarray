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
 *  Eduard Valeyev
 *  Department of Chemistry, Virginia Tech
 *  Feb 6, 2018
 *
 */


#include <TiledArray/external/device.h>
#include <TiledArray/device/thrust.h>

#ifdef TILEDARRAY_HAS_CUDA

namespace thrust {
template<>
void resize<double,TiledArray::device_um_allocator<double>>(
    thrust::device_vector<double, TiledArray::device_um_allocator<double>>& dev_vec,
    size_t size) {
    dev_vec.resize(size);
}
template<>
void resize<float,TiledArray::device_um_allocator<float>>(
    thrust::device_vector<float, TiledArray::device_um_allocator<float>>& dev_vec,
    size_t size) {
    dev_vec.resize(size);
}
}

namespace thrust {
template class device_vector<double, TiledArray::device_um_allocator<double>>;
template class device_vector<float, TiledArray::device_um_allocator<float>>;
}

#endif //TILEDARRAY_HAS_CUDA
