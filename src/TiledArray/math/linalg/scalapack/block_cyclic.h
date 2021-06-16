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
 *  block_cyclic.h
 *  Created: 7  Feb, 2020
 *  Edited:  13 May, 2020 (DBWY)
 *
 */

#ifndef TILEDARRAY_MATH_LINALG_SCALAPACK_TO_BLOCKCYCLIC_H__INCLUDED
#define TILEDARRAY_MATH_LINALG_SCALAPACK_TO_BLOCKCYCLIC_H__INCLUDED

#include <TiledArray/config.h>
#if TILEDARRAY_HAS_SCALAPACK

#include <TiledArray/conversions/make_array.h>
#include <TiledArray/dist_array.h>
#include <TiledArray/error.h>
#include <TiledArray/tensor.h>
#include <TiledArray/tiled_range.h>

#include <blacspp/grid.hpp>
#include <blacspp/information.hpp>

#include <scalapackpp/block_cyclic.hpp>
#include <scalapackpp/util/type_traits.hpp>

namespace TiledArray::math::linalg::scalapack {

template <typename T,
          typename = scalapackpp::detail::enable_if_scalapack_supported_t<T>>
class BlockCyclicMatrix : public madness::WorldObject<BlockCyclicMatrix<T>> {
  using world_base_t = madness::WorldObject<BlockCyclicMatrix<T>>;
  using col_major_mat_t =
      Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor>;

  std::shared_ptr<blacspp::Grid> internal_grid_ = nullptr;

  scalapackpp::BlockCyclicDist2D
      bc_dist_;                     ///< Block-cyclic distribution manager
  col_major_mat_t local_mat_;       ///< Local block cyclic buffer
  std::pair<size_t, size_t> dims_;  ///< Dims of the matrix

  template <typename Tile,
            typename = std::enable_if_t<
                TiledArray::detail::is_contiguous_tensor_v<Tile>>>
  void put_tile(const Tile& tile) {
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
          Tensor<T> subblock;
          if constexpr (TiledArray::detail::is_ta_tensor_v<Tile>)
            subblock = tile.block({i, j}, {i_last, j_last});
          else {
            auto tile_blk_range = TiledArray::BlockRange(
                TiledArray::detail::make_ta_range(tile.range()), {i, j},
                {i_last, j_last});
            using std::data;
            auto tile_blk_view =
                TiledArray::make_const_map(data(tile), tile_blk_range);
            subblock = tile_blk_view;
          }
          world_base_t::send(
              owner(i, j),
              &BlockCyclicMatrix<T>::template put_tile<decltype(subblock)>,
              subblock);
        }

      }  // for (j)
    }    // for (i)

  }  // put_tile

  template <typename Tile,
            typename = std::enable_if_t<
                TiledArray::detail::is_contiguous_tensor_v<Tile>>>
  Tile extract_submatrix(std::vector<size_t> lo, std::vector<size_t> up) {
    assert(bc_dist_.i_own(lo[0], lo[1]));

    auto [i_st, j_st] = bc_dist_.local_indx(lo[0], lo[1]);

    auto i_extent = up[0] - lo[0];
    auto j_extent = up[1] - lo[1];

    Range range(lo, up);
    Tile tile(range);

    auto tile_map = eigen_map(tile);

    tile_map = local_mat_.block(i_st, j_st, i_extent, j_extent);

    return tile;

  }  // extract_submatrix

 public:
  /**
   *  \brief Construct and allocate memory for a BlockCyclic matrix.
   *
   *  @param[in] world MADNESS World context
   *  @param[in] grid  BLACS grid context
   *  @param[in] M     Number of rows in distributed matrix
   *  @param[in] N     Number of columns in distributed matrix
   *  @param[in] MB    Block-cyclic row distribution factor
   *  @param[in] NB    Block-cyclic column distribution factor
   */
  BlockCyclicMatrix(madness::World& world, const blacspp::Grid& grid, size_t M,
                    size_t N, size_t MB, size_t NB)
      : world_base_t(world), bc_dist_(grid, MB, NB), dims_{M, N} {
    // TODO: Check if world / grid are compatible

    // Determine size of local BC buffer
    auto [Mloc, Nloc] = bc_dist_.get_local_dims(M, N);
    local_mat_.resize(Mloc, Nloc);
    local_mat_.fill(0);

    world_base_t::process_pending();
  };

  /**
   *  \brief Construct a BlockCyclic matrix from a DistArray
   *
   *  @param[in] array Array to redistribute
   *  @param[in] grid  BLACS grid context
   *  @param[in] MB    Block-cyclic row distribution factor
   *  @param[in] NB    Block-cyclic column distribution factor
   */
  template <typename Tile, typename Policy>
  BlockCyclicMatrix(const DistArray<Tile, Policy>& array,
                    const blacspp::Grid& grid, size_t MB, size_t NB)
      : BlockCyclicMatrix(array.world(), grid, array.trange().dim(0).extent(),
                          array.trange().dim(1).extent(), MB, NB) {
    TA_ASSERT(array.trange().rank() == 2);

    for (auto it = array.begin(); it != array.end(); ++it) put_tile(it->get());
    world_base_t::process_pending();
  }

  BlockCyclicMatrix(const BlockCyclicMatrix&) = default;
  BlockCyclicMatrix(BlockCyclicMatrix&&) = default;

  BlockCyclicMatrix& operator=(const BlockCyclicMatrix&) = default;
  BlockCyclicMatrix& operator=(BlockCyclicMatrix&&) = default;

  const auto& dist() const { return bc_dist_; }
  const auto& dims() const { return dims_; }
  const auto& local_mat() const { return local_mat_; }

  auto& dist() { return bc_dist_; }
  auto& dims() { return dims_; }
  auto& local_mat() { return local_mat_; }

  inline size_t owner(size_t I, size_t J) const noexcept {
    return blacspp::coordinate_rank(bc_dist_.grid(),
                                    bc_dist_.owner_coordinate(I, J));
  }

  template <typename Array>
  Array tensor_from_matrix(const TiledRange& trange) const {
    using Tile = typename Array::value_type;
    auto construct_tile = [&](Tile& tile, const Range& range) {
      tile = Tile(range);

      // Extract Tile information
      const auto* lo = tile.range().lobound_data();
      const auto* up = tile.range().upbound_data();
      auto tile_map = eigen_map(tile);

      // Extract distribution information
      const size_t mb = bc_dist_.mb();
      const size_t nb = bc_dist_.nb();

      decltype(mb) m = dims_.first;
      decltype(mb) n = dims_.second;

      // Loop over 2D BC compatible blocks
      size_t i_extent, j_extent;
      for (size_t i = lo[0], i_t = 0ul; i < up[0];
           i += i_extent, i_t += i_extent)
        for (size_t j = lo[1], j_t = 0ul; j < up[1];
             j += j_extent, j_t += j_extent) {
          // Determine indices of start of BC owning block
          decltype(m) i_block_begin = (i / mb) * mb;
          decltype(m) j_block_begin = (j / nb) * nb;

          // Determine indices of end of BC owning block
          const auto i_block_end = std::min(m, i_block_begin + mb);
          const auto j_block_end = std::min(n, j_block_begin + nb);

          // Cut block if necessary to adhere to tile dimensions
          const auto i_last = std::min(i_block_end, static_cast<decltype(m)>(up[0]));
          const auto j_last = std::min(j_block_end, static_cast<decltype(m)>(up[1]));

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
                owner(i, j),
                &BlockCyclicMatrix<T>::template extract_submatrix<Tensor<T>>,
                lo, up);

            if constexpr (TiledArray::detail::is_ta_tensor_v<Tile>)
              tile.block(lo, up) = remtile_fut.get();
            else {
              auto tile_blk_range = TiledArray::BlockRange(
                  TiledArray::detail::make_ta_range(tile.range()), lo, up);
              using std::data;
              auto tile_blk_view =
                  TiledArray::make_map(data(tile), tile_blk_range);
              tile_blk_view = remtile_fut.get();
            }
          }
        }

      return norm(tile);
    };

    return make_array<Array>(world_base_t::get_world(), trange, construct_tile);
  }

};  // class BlockCyclicMatrix

/**
 *  \brief Convert a dense DistArray to block-cyclic storage format
 *
 *  @tparam T Datatype of underlying tile
 *
 *  @param[in]  array   DistArray to be converted to block-cyclic format
 *  @param[in]  grid    BLACS grid context for block-cyclic matrix
 *  @param[in]  MB      Row blocking factor of resulting block-cyclic matrix
 *  @param[in]  NB      Column blocking factor of resulting block-cyclic matrix
 *
 *  @returns    Block-cyclic conversion of input DistArray
 */
template <typename Array>
BlockCyclicMatrix<typename std::remove_cv_t<Array>::element_type>
array_to_block_cyclic(const Array& array, const blacspp::Grid& grid, size_t MB,
                      size_t NB) {
  return BlockCyclicMatrix<typename std::remove_cv_t<Array>::element_type>(
      array, grid, MB, NB);
}

/**
 *  \brief Convert a block-cyclic matrix to DistArray
 *
 *  @tparam Datatype of underlying tile
 *
 *  @param[in]  matrix  Block-cyclic matrix to convert to DistArray
 *  @param[in]  trange  Tiled ranges for the resulting DistArray
 *
 *  @returns DistArray conversion of input block-cyclic matrix
 */
template <typename Array>
std::remove_cv_t<Array> block_cyclic_to_array(
    const BlockCyclicMatrix<typename std::remove_cv_t<Array>::element_type>&
        matrix,
    const TiledRange& trange) {
  return matrix.template tensor_from_matrix<std::remove_cv_t<Array>>(trange);
}

}  // namespace TiledArray::math::linalg::scalapack

#endif  // TILEDARRAY_HAS_SCALAPACK
#endif  // TILEDARRAY_MATH_LINALG_SCALAPACK_TO_BLOCKCYCLIC_H__INCLUDED
