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
 *  Eduard Valeyev
 *
 *  chol.h
 *  Created:    16 October, 2020
 *
 */
#ifndef TILEDARRAY_ALGEBRA_LAPACK_LAPACK_H__INCLUDED
#define TILEDARRAY_ALGEBRA_LAPACK_LAPACK_H__INCLUDED

#include <TiledArray/config.h>
#include <TiledArray/algebra/types.h>
#include <Eigen/Core>

namespace TiledArray::lapack {

template<typename T>
using Vector = Eigen::Matrix<T,Eigen::Dynamic,1,Eigen::ColMajor>;

template<typename T>
using Matrix = Eigen::Matrix<T,Eigen::Dynamic,Eigen::Dynamic,Eigen::ColMajor>;

template<typename T>
void cholesky(Matrix<T> &A);

template<typename T>
void cholesky_linv(Matrix<T> &A);

template<typename T>
void cholesky_solve(Matrix<T> &A, Matrix<T> &X);

template<typename T>
void cholesky_lsolve(TransposeFlag transpose, Matrix<T> &A, Matrix<T> &X);

template<typename T>
void hereig(Matrix<T> &A, Vector<T> &W);

template<typename T>
void svd(Matrix<T> &A, Vector<T> &S, Matrix<T> *U, Matrix<T> *VT);

}

#endif  // TILEDARRAY_ALGEBRA_LAPACK_LAPACK_H__INCLUDED
