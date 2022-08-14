/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2020 Virginia Tech
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
 *  David Williams-Young
 *  Computational Research Division, Lawrence Berkeley National Laboratory
 *
 *  svd.h
 *  Created:    12 June, 2020
 *
 */
#ifndef TILEDARRAY_MATH_LINALG_FORWARD_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_FORWARD_H__INCLUDED

#include <TiledArray/config.h>
#include <type_traits>

#include <blas/util.hh>

namespace TiledArray::math::linalg {

using Op = ::blas::Op;
static constexpr auto NoTranspose = Op::NoTrans;
static constexpr auto Transpose = Op::Trans;
static constexpr auto ConjTranspose = Op::ConjTrans;

/// converts Op to ints in manner useful for bit manipulations
/// NoTranspose -> 0, Transpose->1, ConjTranspose->2
inline auto to_int(Op op) {
  if (op == NoTranspose)
    return 0;
  else if (op == Transpose)
    return 1;
  else  // op == ConjTranspose
    return 2;
}

struct SVD {
  enum Vectors { ValuesOnly, LeftVectors, RightVectors, AllVectors };
};

/// known linear algebra backends
enum LinearAlgebraBackend {
  /// choose the best that's available, taking into consideration the problem
  /// size and # of ranks
  BestAvailable,
  /// LAPACK on rank 0, followed by broadcast
  LAPACK,
  /// ScaLAPACK
  ScaLAPACK,
  /// TTG (currently only provides cholesky and cholesky_linv)
  TTG
};

LinearAlgebraBackend get_linalg_backend();
void set_linalg_backend(LinearAlgebraBackend b);

std::size_t get_linalg_crossover_to_distributed();
void set_linalg_crossover_to_distributed(std::size_t c);

}  // namespace TiledArray::math::linalg

namespace TiledArray {
using TiledArray::math::linalg::ConjTranspose;
using TiledArray::math::linalg::NoTranspose;
using TiledArray::math::linalg::SVD;
using TiledArray::math::linalg::Transpose;
}  // namespace TiledArray

#endif  // TILEDARRAY_MATH_LINALG_FORWARD_H__INCLUDED
