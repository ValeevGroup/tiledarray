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

template class Tensor<double>;
template class Tensor<float>;
// template class Tensor<int>;
// template class Tensor<long>;
template class Tensor<std::complex<double>>;
template class Tensor<std::complex<float>>;

}  // namespace TiledArray

// ---------------------------------------------------------------------------
// libxsmm fast-path wrapper for the scale strided GEMM (declared in tensor.h).
// Kept out of tensor.h so the scale call sites need no libxsmm types; forwards
// to detail::libxsmm_gemm_le64 (declared in the lean libxsmm_gemm.h,
// defined in libxsmm_gemm.cpp -- the only TU that includes <libxsmm.h>).
#include "TiledArray/math/libxsmm_gemm.h"

namespace TiledArray::detail {
bool scale_libxsmm_dgemm(bool trans_a, bool trans_b, long m, long n, long k,
                         const double* a, long lda, const double* b, long ldb,
                         double beta, double* c, long ldc) {
  return TiledArray::detail::libxsmm_gemm_le64(
      trans_a, trans_b, m, n, k, /*alpha=*/1.0, a, lda, b, ldb, beta, c, ldc);
}
}  // namespace TiledArray::detail
