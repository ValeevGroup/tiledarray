/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2016  Virginia Tech
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
 *  justus
 *  Department of Chemistry, Virginia Tech
 *
 *  tensor.cpp
 *  Feb 5, 2016
 *
 */

#include "tensor.h"
#include "tensor_interface.h"

namespace TiledArray {

template class Tensor<double, Eigen::aligned_allocator<double> >;
template class Tensor<float, Eigen::aligned_allocator<float> >;
template class Tensor<int, Eigen::aligned_allocator<int> >;
template class Tensor<long, Eigen::aligned_allocator<long> >;
//  template class Tensor<std::complex<double>,
//  Eigen::aligned_allocator<std::complex<double> > >; template class
//  Tensor<std::complex<float>, Eigen::aligned_allocator<std::complex<float> >
//  >;

}  // namespace TiledArray
