/*
 *  This file is a part of TiledArray.
 *  Copyright (C) 2013  Virginia Tech
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
 *  retile.h
 *  May 11, 2020
 */

#ifndef TILEDARRAY_RETILE_H
#define TILEDARRAY_RETILE_H

#include "../algebra/utils.h"
#include "../special/diagonal_array.h"

/// \name Retile function
/// \brief Retiles a tensor with a provided TiledRange

/// Retiles the data of the input tensor by comparing each dimension of its
/// TiledRange with the corresponding dimension of the input TiledRange. If they
/// are not equivalent, a suitably tiled identity matrix is used to convert that
/// dimension to the desired tiling.
/// \param tensor The tensor whose data is to be retiled
/// \param new_trange The desired TiledRange of the output tensor
/// \return A new tensor with appropriately tiled data

namespace TiledArray {

template <typename TileType, typename PolicyType>
auto retile(const DistArray<TileType, PolicyType>& tensor,
            const TiledRange& new_trange) {
  // Make sure ranks match
  auto rank = new_trange.rank();
  auto tensor_rank = tensor.trange().rank();
  assert((rank == tensor_rank) && "TiledRanges are of different ranks");

  // Makes the annotations for the contraction step
  auto annotations =
      [&](std::size_t target_dim) -> std::tuple<std::string, std::string> {
    std::ostringstream final, switcher;
    switcher << "i" << target_dim << ",iX";
    if (target_dim == 0) {
      final << "iX";
    } else {
      final << "i0";
    }
    for (unsigned int d = 1; d < rank; ++d) {
      if (d == target_dim) {
        final << ",iX";
      } else {
        final << ",i" << d;
      }
    }
    return {final.str(), switcher.str()};
  };

  // Check the different dimensions and contract when needed
  using tensor_type = DistArray<TileType, PolicyType>;
  auto output_tensor = tensor;
  for (auto i = 0; i < rank; ++i) {
    if (new_trange.dim(i) != tensor.trange().dim(i)) {
      // Make identity for contraction
      TiledRange retiler{tensor.trange().dim(i), new_trange.dim(i)};
      auto identity = diagonal_array<tensor_type>(tensor.world(), retiler);

      // Make indices for contraction
      auto start = TA::detail::dummy_annotation(rank);
      auto [finish, change] = annotations(i);

      // Retile
      new_tensor(finish) = output_tensor(start) * identity(change);
    }
  }

  return output_tensor;
}

} // namespace TiledArray


#endif  // TILEDARRAY_RETILE_H
