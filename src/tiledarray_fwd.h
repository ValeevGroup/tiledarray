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


  // TiledArray Tensors
  template<typename, typename>
  class Tensor;

  // Tensor Typedefs
  using TensorD = Tensor<double, Eigen::aligned_allocator<double> >;
  using TensorI = Tensor<int, Eigen::aligned_allocator<int> >;
  using TensorF = Tensor<float, Eigen::aligned_allocator<float> >;

  // TiledArray Arrays
  template<typename, unsigned int, typename>
  class Array;

  // Array Typedefs
  using Array1D = Array<double, 1, TensorD>;
  using Array2D = Array<double, 2, TensorD>;
  using Array3D = Array<double, 3, TensorD>;
  using Array4D = Array<double, 4, TensorD>;

  using Array1I = Array<int, 1, TensorI>;
  using Array2I = Array<int, 2, TensorI>;
  using Array3I = Array<int, 3, TensorI>;
  using Array4I = Array<int, 4, TensorI>;

  using Array1F = Array<float, 1, TensorF>;
  using Array2F = Array<float, 2, TensorF>;
  using Array3F = Array<float, 3, TensorF>;
  using Array4F = Array<float, 4, TensorF>;

} // namespace TiledArray

#endif // TILEDARRAY_FWD_H__INCLUDED
