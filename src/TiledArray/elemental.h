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
 *  Drew Lewis
 *  Department of Chemistry, Virginia Tech
 *
 *  elemental.h
 *  March 23, 2014
 *
 */

#ifndef TILEDARRAY_ELEMENTAL_H__INCLUDED
#define TILEDARRAY_ELEMENTAL_H__INCLUDED

#include <TiledArray/eigen.h>
#include <elemental.hpp>

namespace TiledArray {

  template<typename T, unsigned int DIM, typename Tile>
  elem::DistMatrix<T> array_to_elem(const Array<T,DIM, Tile> &array,
                                       const elem::Grid &grid){
    // Check that the Array is 2-d
    TA_USER_ASSERT(DIM == 2u,
      "TiledArray::array_to_elem(): The array dimension must be 2.");

    // Construct the elemental matrix
    std::vector<std::size_t> sizes = array.trange().elements().size();
    elem::DistMatrix<T> mat(sizes[0], sizes[1], grid);
    elem::Zero(mat);

    //Create the Axpy interface to fill elemental matrix
    elem::AxpyInterface<T> interface;
    // Attach matrix to interface
    interface.Attach(elem::LOCAL_TO_GLOBAL, mat);

    // Get array iterators
    typename Array<T,DIM,Tile>::iterator it = array.begin();
    typename Array<T,DIM,Tile>::iterator end = array.end();

    for(; it != end; ++it){
      // Get tile matrix location info
      const typename Array<T,DIM,Tile>::value_type tile = *it;
      const std::size_t t0start = tile.range().start()[0];
      const std::size_t t1start = tile.range().start()[1];
      const std::size_t t0size = tile.range().size()[0];
      const std::size_t t1size = tile.range().size()[1];

      // Create Eigen RowMajor Map of tile
      const Eigen::Map<
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
        Eigen::AutoAlign> eig_row_map = eigen_map(tile, t0size, t1size);

      // Create ColMajor EigenMatrix from RowMajor Map
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> CMatrix = eig_row_map;

      // Make Elemental local matrix and attach the data.
      elem::Matrix<T> ElemBlock;
      ElemBlock.Attach(CMatrix.rows(), CMatrix.cols(), CMatrix.data(), CMatrix.rows());

      // Attach elem local matrix to elem global matrix
      interface.Axpy(1.0, ElemBlock, t0start, t1start);
    }
    interface.Detach(); // Does communication using elemental

    return mat;
  }

  template<typename T, unsigned int DIM, typename Tile>
  void elem_to_array(Array<T,DIM,Tile> &array, elem::DistMatrix<T> &mat){
    TA_USER_ASSERT(DIM==2u, "TiledArray::elem_to_array(): requires the array to have dimension 2");
    TA_USER_ASSERT((array.trange().elements().extent()[0]==mat.Height()) &&
                   (array.trange().elements().extent()[1] == mat.Width()),
                   "TiledArray::elem_to_array(): requires the shape of the elem matrix and the array to be the same.");

    // Make interface and attach mat
    elem::AxpyInterface<T> interface;
    interface.Attach(elem::GLOBAL_TO_LOCAL, mat);

    // Get iterators to array
    typename Array<T,DIM, Tile>::iterator it = array.begin();
    typename Array<T,DIM, Tile>::iterator end = array.end();

    // Loop over tiles and improperly assign the data to them in column major
    // format.
    for(;it != end; ++it){
      // Get tile matrix location info
      typename Array<T,DIM,Tile>::value_type tile = *it;
      std::size_t t0start = tile.range().start()[0];
      std::size_t t1start = tile.range().start()[1];
      std::size_t t0size = tile.range().size()[0];
      std::size_t t1size = tile.range().size()[1];

      // Make Elemental local matrix and attach the data.
      elem::Matrix<T> ElemBlock;
      ElemBlock.Attach(t0size, t1size, tile.data(), t0size);

      // need to zero tile so Axpy doesn't add to it.
      std::fill(ElemBlock.Buffer(), ElemBlock.Buffer()+t0size*t1size, 0.0);

      // Attach elem local matrix to elem global matrix
      interface.Axpy(1.0, ElemBlock, t0start, t1start);
    }
    interface.Detach(); // Does communication using elemental

    // now we have to go back and fix it so the tiles are row major ordered.
    it = array.begin();
    for(;it != end; ++it){
      // Get tile and size
      typename Array<T,DIM,Tile>::value_type tile = *it;
      std::size_t t0size = tile.range().size()[0];
      std::size_t t1size = tile.range().size()[1];

      // copy to row major matrix
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_mat =
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                   Eigen::AutoAlign > (tile.data(),t0size,t1size);

      // Finally copy the data back into the tile in the correct format.
      std::copy(row_mat.data(), row_mat.data()+row_mat.size(), tile.data());
    }
  }

} // namespace TiledArray

#endif // TILEDARRAY_ELEMENTAL_H__INCLUDED
