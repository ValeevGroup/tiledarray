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

#ifndef TILEDARRAY_EXTERNAL_ELEMENTAL_H__INCLUDED
#define TILEDARRAY_EXTERNAL_ELEMENTAL_H__INCLUDED

#include <TiledArray/config.h>

#if TILEDARRAY_HAS_ELEMENTAL

#if HAVE_ELEMENTAL_H

#include <TiledArray/conversions/eigen.h>
#include <elemental.hpp>

namespace TiledArray {

template <typename Tile>
elem::DistMatrix<typename Tile::value_type> array_to_elem(
    const DistArray<Tile>& array, const elem::Grid& grid) {
  // Check that the Array is 2-d
  TA_USER_ASSERT(array.range().rank() == 2u,
                 "TiledArray::array_to_elem(): The array dimension must be 2.");

  // Construct the elemental matrix
  using T = typename Tile::value_type;
  auto sizes = array.trange().elements_range().extent_data();
  elem::DistMatrix<T> mat(sizes[0], sizes[1], grid);
  elem::Zero(mat);

  // Create the Axpy interface to fill elemental matrix
  elem::AxpyInterface<T> interface;
  // Attach matrix to interface
  interface.Attach(elem::LOCAL_TO_GLOBAL, mat);

  // Get array iterators
  typename DistArray<Tile>::const_iterator it = array.begin();
  typename DistArray<Tile>::const_iterator end = array.end();

  for (; it != end; ++it) {
    // Get tile matrix location info
    const typename DistArray<Tile>::value_type tile = *it;

    // Get tile range data
    const auto* MADNESS_RESTRICT const tile_lower = tile.range().lobound_data();
    const auto* MADNESS_RESTRICT const tile_extent = tile.range().extent_data();
    const std::size_t tile_lower_0 = tile_lower[0];
    const std::size_t tile_lower_1 = tile_lower[1];
    const std::size_t tile_extent_0 = tile_extent[0];
    const std::size_t tile_extent_1 = tile_extent[1];

    // Create Eigen RowMajor Map of tile
    const Eigen::Map<
        const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>,
        Eigen::AutoAlign>
        eig_row_map = eigen_map(tile, tile_extent_0, tile_extent_1);

    // Create ColMajor EigenMatrix from RowMajor Map
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> CMatrix = eig_row_map;

    // Make Elemental local matrix and attach the data.
    elem::Matrix<T> ElemBlock;
    ElemBlock.Attach(CMatrix.rows(), CMatrix.cols(), CMatrix.data(),
                     CMatrix.rows());

    // Attach elem local matrix to elem global matrix
    interface.Axpy(1.0, ElemBlock, tile_lower_0, tile_lower_1);
  }
  interface.Detach();  // Does communication using elemental

  return mat;
}

template <typename Tile>
void elem_to_array(DistArray<Tile>& array,
                   elem::DistMatrix<typename Tile::value_type>& mat) {
  using T = typename Tile::value_type;
  TA_USER_ASSERT(
      array.range().rank() == 2u,
      "TiledArray::elem_to_array(): requires the array to have dimension 2");
  TA_USER_ASSERT(
      (array.trange().elements_range().extent()[0] == mat.Height()) &&
          (array.trange().elements_range().extent()[1] == mat.Width()),
      "TiledArray::elem_to_array(): requires the shape of the elem matrix and "
      "the array to be the same.");

  // Make interface and attach mat
  elem::AxpyInterface<T> interface;
  interface.Attach(elem::GLOBAL_TO_LOCAL, mat);

  // Get iterators to array
  typename DistArray<Tile>::iterator it = array.begin();
  typename DistArray<Tile>::iterator end = array.end();

  // Loop over tiles and improperly assign the data to them in column major
  // format.
  for (; it != end; ++it) {
    // Get tile matrix location info
    typename DistArray<Tile>::value_type tile = *it;

    // Get tile range data
    const auto* MADNESS_RESTRICT const tile_lower = tile.range().lobound_data();
    const auto* MADNESS_RESTRICT const tile_extent = tile.range().extent_data();
    const std::size_t tile_lower_0 = tile_lower[0];
    const std::size_t tile_lower_1 = tile_lower[1];
    const std::size_t tile_extent_0 = tile_extent[0];
    const std::size_t tile_extent_1 = tile_extent[1];

    // Make Elemental local matrix and attach the data.
    elem::Matrix<T> ElemBlock;
    ElemBlock.Attach(tile_extent_0, tile_extent_1, tile.data(), tile_extent_0);

    // need to zero tile so Axpy doesn't add to it.
    std::fill(ElemBlock.Buffer(),
              ElemBlock.Buffer() + tile_extent_0 * tile_extent_1, 0.0);

    // Attach elem local matrix to elem global matrix
    interface.Axpy(1.0, ElemBlock, tile_lower_0, tile_lower_1);
  }
  interface.Detach();  // Does communication using elemental

  // now we have to go back and fix it so the tiles are row major ordered.
  it = array.begin();
  for (; it != end; ++it) {
    // Get tile and size
    typename DistArray<Tile>::value_type tile = *it;

    // Get tile range data
    const auto* MADNESS_RESTRICT const tile_extent = tile.range().extent_data();
    const std::size_t tile_extent_0 = tile_extent[0];
    const std::size_t tile_extent_1 = tile_extent[1];

    // copy to row major matrix
    Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> row_mat =
        Eigen::Map<const Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>,
                   Eigen::AutoAlign>(tile.data(), tile_extent_0, tile_extent_1);

    // Finally copy the data back into the tile in the correct format.
    std::copy(row_mat.data(), row_mat.data() + row_mat.size(), tile.data());
  }
}

}  // namespace TiledArray

#elif HAVE_EL_H  // end of HAVE_ELEMENTAL_H

// pacify clang warnings about tautological comparisons in
// El/macros/GuardAndPayload.h
TILEDARRAY_PRAGMA_CLANG(diagnostic push)
TILEDARRAY_PRAGMA_CLANG(diagnostic ignored "-Wtautological-compare")

#include <El.hpp>

TILEDARRAY_PRAGMA_CLANG(diagnostic pop)

#include <TiledArray/conversions/elemental.h>

#else
#error \
    "TILEDARRAY_HAS_ELEMENTAL set but neither HAVE_EL_H nor HAVE_ELEMENTAL_H set: file an issue at " TILEDARRAY_PACKAGE_URL
#endif

#endif  // TILEDARRAY_HAS_ELEMENTAL

#endif  // TILEDARRAY_EXTERNAL_ELEMENTAL_H__INCLUDED
