/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2014  Virginia Tech
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
 */

#ifndef TILEDARRAY_FWD_H__INCLUDED
#define TILEDARRAY_FWD_H__INCLUDED

namespace Eigen { // Eigen Alligned allocator for TA::Tensor
  template<class>
  class aligned_allocator;
} // namespace Eigen

namespace TiledArray {

  //TiledArray Policy
  class DensePolicy;
  class SparsePolicy;

  // TiledArray Tensors
  template<typename, typename>
  class Tensor;

  typedef Tensor<double, Eigen::aligned_allocator<double> > TensorD;
  typedef Tensor<int, Eigen::aligned_allocator<int> > TensorI;
  typedef Tensor<float, Eigen::aligned_allocator<float> > TensorF;
  typedef Tensor<long, Eigen::aligned_allocator<long> > TensorL;

  // TiledArray Arrays
  template<typename, unsigned int, typename, typename>
  class Array;

  // Dense Array Typedefs
  typedef Array<double, 1, TensorD, DensePolicy> TArray1D;
  typedef Array<double, 2, TensorD, DensePolicy> TArray2D;
  typedef Array<double, 3, TensorD, DensePolicy> TArray3D;
  typedef Array<double, 4, TensorD, DensePolicy> TArray4D;

  typedef Array<int, 1, TensorI, DensePolicy> TArray1I;
  typedef Array<int, 2, TensorI, DensePolicy> TArray2I;
  typedef Array<int, 3, TensorI, DensePolicy> TArray3I;
  typedef Array<int, 4, TensorI, DensePolicy> TArray4I;

  typedef Array<float, 1, TensorF, DensePolicy> TArray1F;
  typedef Array<float, 2, TensorF, DensePolicy> TArray2F;
  typedef Array<float, 3, TensorF, DensePolicy> TArray3F;
  typedef Array<float, 4, TensorF, DensePolicy> TArray4F;

  typedef Array<long, 1, TensorL, DensePolicy> TArray1L;
  typedef Array<long, 2, TensorL, DensePolicy> TArray2L;
  typedef Array<long, 3, TensorL, DensePolicy> TArray3L;
  typedef Array<long, 4, TensorL, DensePolicy> TArray4L;

  // Sparse Array Typedefs
  typedef Array<double, 1, TensorD, SparsePolicy> TArray1D;
  typedef Array<double, 2, TensorD, SparsePolicy> TArray2D;
  typedef Array<double, 3, TensorD, SparsePolicy> TArray3D;
  typedef Array<double, 4, TensorD, SparsePolicy> TArray4D;

  typedef Array<int, 1, TensorI, SparsePolicy> TArray1I;
  typedef Array<int, 2, TensorI, SparsePolicy> TArray2I;
  typedef Array<int, 3, TensorI, SparsePolicy> TArray3I;
  typedef Array<int, 4, TensorI, SparsePolicy> TArray4I;

  typedef Array<float, 1, TensorF, SparsePolicy> TArray1F;
  typedef Array<float, 2, TensorF, SparsePolicy> TArray2F;
  typedef Array<float, 3, TensorF, SparsePolicy> TArray3F;
  typedef Array<float, 4, TensorF, SparsePolicy> TArray4F;

  typedef Array<long, 1, TensorL, SparsePolicy> TArray1L;
  typedef Array<long, 2, TensorL, SparsePolicy> TArray2L;
  typedef Array<long, 3, TensorL, SparsePolicy> TArray3L;
  typedef Array<long, 4, TensorL, SparsePolicy> TArray4L;
} // namespace TiledArray

#endif // TILEDARRAY_FWD_H__INCLUDED
