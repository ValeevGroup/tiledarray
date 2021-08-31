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
 *  Justus Calvin
 *  Department of Chemistry, Virginia Tech
 *
 *  array_impl.cpp
 *  Feb 5, 2016
 *
 */

#include "array_impl.h"

namespace TiledArray {
namespace detail {

template class ArrayImpl<Tensor<double>, DensePolicy>;
template class ArrayImpl<Tensor<float>, DensePolicy>;
// template class ArrayImpl<Tensor<int>,
//                         DensePolicy>;
// template class ArrayImpl<Tensor<long>,
//                         DensePolicy>;
template class ArrayImpl<Tensor<std::complex<double>>, DensePolicy>;
template class ArrayImpl<Tensor<std::complex<float>>, DensePolicy>;

template class ArrayImpl<Tensor<double>, SparsePolicy>;
template class ArrayImpl<Tensor<float>, SparsePolicy>;
// template class ArrayImpl<Tensor<int>,
//                         SparsePolicy>;
// template class ArrayImpl<Tensor<long>,
//                         SparsePolicy>;
template class ArrayImpl<Tensor<std::complex<double>>, SparsePolicy>;
template class ArrayImpl<Tensor<std::complex<float>>, SparsePolicy>;

}  // namespace detail
}  // namespace TiledArray
