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
 *  dist_array.cpp
 *  Feb 5, 2016
 *
 */

#include "dist_array.h"
#include "expressions/tsr_expr.h"
#include "policies/dense_policy.h"
#include "policies/sparse_policy.h"
#include "tensor/tensor.h"

namespace TiledArray {

template class DistArray<Tensor<double>, DensePolicy>;
template class DistArray<Tensor<float>, DensePolicy>;
// template class DistArray<Tensor<int>,
//                         DensePolicy>;
// template class DistArray<Tensor<long>,
//                         DensePolicy>;
template class DistArray<Tensor<std::complex<double>>, DensePolicy>;
template class DistArray<Tensor<std::complex<float>>, DensePolicy>;

template class DistArray<Tensor<double>, SparsePolicy>;
template class DistArray<Tensor<float>, SparsePolicy>;
// template class DistArray<Tensor<int>,
//                         SparsePolicy>;
// template class DistArray<Tensor<long>,
//                         SparsePolicy>;
template class DistArray<Tensor<std::complex<double>>, SparsePolicy>;
template class DistArray<Tensor<std::complex<float>>, SparsePolicy>;

}  // namespace TiledArray
