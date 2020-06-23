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
 *  scalapack.h
 *  Created: 7 Feb, 2020
 *
 */

#ifndef TILEDARRAY_CONVERSIONS_TO_SCALAPACK_H__INCLUDED
#define TILEDARRAY_CONVERSIONS_TO_SCALAPACK_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SCALAPACK

#include <TiledArray/dist_array.h>
#include <TiledArray/error.h>
#include <TiledArray/tensor.h>
#include <TiledArray/tiled_range.h>

#include <blacspp/grid.hpp>
#include <blacspp/information.hpp>

#include <scalapackpp/block_cyclic.hpp>
#include <scalapackpp/util/sfinae.hpp>

namespace TiledArray {

template <typename T,
          typename = scalapackpp::detail::enable_if_scalapack_supported_t<T>>
class ScaLAPACKMatrix : public madness::WorldObject<ScaLAPACKMatrix<T>> {
  using world_base_t = madness::WorldObject<ScaLAPACKMatrix<T>>;
  using col_major_mat_t =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  std::shared_ptr<blacspp::Grid> internal_grid_ = nullptr;

  scalapackpp::BlockCyclicDist2D
      bc_dist_;                     ///< Block-cyclic distribution manager
  col_major_mat_t local_mat_;       ///< Local block cyclic buffer
  std::pair<size_t, size_t> dims_;  ///< Dims of the matrix

  void put_tile(const Tensor<T>& tile) {
    // Extract Tile information
    const auto* lo = tile.range().lobound_data();
    const auto* up = tile.range().upbound_data();
    auto tile_map = eigen_map(tile);

    // Extract distribution information
    const auto mb = bc_dist_.mb();
    const auto nb = bc_dist_.nb();

    const auto m = dims_.first;
    const auto n = dims_.second;

    // Loop over 2D BC compatible blocks
    long i_extent, j_extent, i_t = 0;
    for (auto i = lo[0]; i < up[0]; i += i_extent, i_t += i_extent) {
      long j_t = 0;
      for (auto j = lo[1]; j < up[1]; j += j_extent, j_t += j_extent) {
        // Determine indices of start of BC owning block
        const auto i_block_begin = (i / mb) * mb;
        const auto j_block_begin = (j / nb) * nb;

        // Determine indices of end of BC owning block
        const auto i_block_end =
            std::min(static_cast<decltype(i)>(m), i_block_begin + mb);
        const auto j_block_end =
            std::min(static_cast<decltype(j)>(n), j_block_begin + nb);

        // Cut block if necessacary to adhere to tile dimensions
        const auto i_last = std::min(i_block_end, up[0]);
        const auto j_last = std::min(j_block_end, up[1]);

        // Calculate extents of the block to be copied
        i_extent = i_last - i;
        j_extent = j_last - j;

        if (bc_dist_.i_own(i, j)) {
          // Calculate local indices from BC distribution
          auto [i_local, j_local] = bc_dist_.local_indx(i, j);

          // Copy the block into local storage
          local_mat_.block(i_local, j_local, i_extent, j_extent) =
              tile_map.block(i_t, j_t, i_extent, j_extent);

        } else {
          // Send the subblock to a remote rank for processing
          Tensor<T> subblock(tile.block({i, j}, {i_last, j_last}));
          world_base_t::send(owner(i, j), &ScaLAPACKMatrix<T>::put_tile,
                             subblock);
        }

      }  // for (j)
    }    // for (i)

  }  // put_tile

  Tensor<T> extract_submatrix(std::vector<size_t> lo, std::vector<size_t> up) {
    assert(bc_dist_.i_own(lo[0], lo[1]));

    auto [i_st, j_st] = bc_dist_.local_indx(lo[0], lo[1]);

    auto i_extent = up[0] - lo[0];
    auto j_extent = up[1] - lo[1];

    Range range(lo, up);
    Tensor<T> tile(range);

    auto tile_map = eigen_map(tile);

    tile_map = local_mat_.block(i_st, j_st, i_extent, j_extent);

    return tile;

  }  // extract_submatrix

 public:
  /**
   *  \brief Construct and allocate memory for a ScaLAPACK matrix.
   *
   *  @param[in] world MADNESS World context
   *  @param[in] grid  BLACS grid context
   *  @param[in] M     Number of rows in distributed matrix
   *  @param[in] N     Number of columns in distributed matrix
   *  @param[in] MB    Block-cyclic row distribution factor
   *  @param[in] NB    Block-cyclic column distribution factor
   */
  ScaLAPACKMatrix(madness::World& world, const blacspp::Grid& grid, size_t M,
                  size_t N, size_t MB, size_t NB)
      : world_base_t(world), bc_dist_(grid, MB, NB), dims_{M, N} {
    // TODO: Check if world / grid are compatible

    // Determine size of local BC buffer
    auto [Mloc, Nloc] = bc_dist_.get_local_dims(M, N);
    local_mat_.resize(Mloc, Nloc);
    // local_mat_.fill(0);

    world_base_t::process_pending();
  };

  /**
   *  \brief Construct a ScaLAPACK metrix from a TArray
   *
   *  @param[in] array Array to redistribute
   *  @param[in] grid  BLACS grid context
   *  @param[in] MB    Block-cyclic row distribution factor
   *  @param[in] NB    Block-cyclic column distribution factor
   */
  ScaLAPACKMatrix(const TArray<T>& array, const blacspp::Grid& grid, size_t MB,
                  size_t NB)
      : ScaLAPACKMatrix(array.world(), grid, array.trange().dim(0).extent(),
                        array.trange().dim(1).extent(), MB, NB) {
    TA_ASSERT(array.trange().rank() == 2);

    for (auto it = array.begin(); it != array.end(); ++it) put_tile(*it);
    world_base_t::process_pending();
  }

  ScaLAPACKMatrix(const ScaLAPACKMatrix&) = default;
  ScaLAPACKMatrix(ScaLAPACKMatrix&&) = default;

  ScaLAPACKMatrix& operator=(const ScaLAPACKMatrix&) = default;
  ScaLAPACKMatrix& operator=(ScaLAPACKMatrix&&) = default;

  const auto& dist() const { return bc_dist_; }
  const auto& dims() const { return dims_; }
  const auto& local_mat() const { return local_mat_; }
  auto& local_mat() { return local_mat_; }

  inline size_t owner(size_t I, size_t J) const noexcept {
    return blacspp::coordinate_rank(bc_dist_.grid(),
                                    bc_dist_.owner_coordinate(I, J));
  }

  TArray<T> tensor_from_matrix(const TiledRange& trange) {
    auto construct_tile = [&](Tensor<T>& tile, const Range& range) {
      tile = Tensor<T>(range);

      // Extract Tile information
      const auto* lo = tile.range().lobound_data();
      const auto* up = tile.range().upbound_data();
      auto tile_map = eigen_map(tile);

      // Extract distribution information
      const auto mb = bc_dist_.mb();
      const auto nb = bc_dist_.nb();

      const auto m = dims_.first;
      const auto n = dims_.second;

      // Loop over 2D BC compatible blocks
      size_t i_extent, j_extent;
      for (size_t i = lo[0], i_t = 0ul; i < up[0];
           i += i_extent, i_t += i_extent)
        for (size_t j = lo[1], j_t = 0ul; j < up[1];
             j += j_extent, j_t += j_extent) {
          // Determine indices of start of BC owning block
          const decltype(m) i_block_begin = (i / mb) * mb;
          const decltype(n) j_block_begin = (j / nb) * nb;

          // Determine indices of end of BC owning block
          const auto i_block_end = std::min(m, i_block_begin + mb);
          const auto j_block_end = std::min(n, j_block_begin + nb);

          // Cut block if necessacary to adhere to tile dimensions
          const auto i_last = std::min(i_block_end, static_cast<size_t>(up[0]));
          const auto j_last = std::min(j_block_end, static_cast<size_t>(up[1]));

          // Calculate extents of the block to be copied
          i_extent = i_last - i;
          j_extent = j_last - j;

          if (bc_dist_.i_own(i, j)) {
            // Calculate local indices from BC distribution
            auto [i_local, j_local] = bc_dist_.local_indx(i, j);

            // Copy the block into local storage
            tile_map.block(i_t, j_t, i_extent, j_extent) =
                local_mat_.block(i_local, j_local, i_extent, j_extent);

          } else {
            std::vector<size_t> lo{i, j};
            std::vector<size_t> up{i_last, j_last};
            madness::Future<Tensor<T>> remtile_fut = world_base_t::send(
                owner(i, j), &ScaLAPACKMatrix<T>::extract_submatrix, lo, up);

            tile.block(lo, up) = remtile_fut.get();
          }
        }

      return tile.norm();
    };

    return make_array<TArray<T>>(world_base_t::get_world(), trange,
                                 construct_tile);
  }

};  // class ScaLAPACKMatrix

}  // namespace TiledArray

#endif  // TILEDARRAY_HAS_SCALAPACK
#endif  // TILEDARRAY_CONVERSIONS_TO_SCALAPACK_H__INCLUDED
