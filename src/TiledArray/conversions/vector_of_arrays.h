//
// Created by Chong Peng on 2019-05-01.
//

#ifndef TILEDARRAY_CONVERSIONS_VECTOR_OF_ARRAYS_H_
#define TILEDARRAY_CONVERSIONS_VECTOR_OF_ARRAYS_H_

#include <tiledarray.h>

namespace TiledArray {

namespace detail {

/// @brief prepends an extra dimension to a TRange

/// The extra dimension will be the leading dimension, and will be blocked by @c block_size

/// @param array_rank extent of the leading dimension of the result
/// @param array_trange the base trange
/// @param block_size blocking range for the new dimension, the dimension being fused
/// @return TiledRange of fused Array object
inline TA::TiledRange fuse_vector_of_tranges(
    std::size_t array_rank,
    const TiledArray::TiledRange& array_trange, std::size_t block_size = 1) {
  /// make the new TiledRange1 for new dimension
  TA::TiledRange1 new_trange1;
  {
    std::vector<std::size_t> new_trange1_v;
    auto range_size = array_rank;
    new_trange1_v.push_back(0);
    for (decltype(range_size) i = block_size; i < range_size; i += block_size) {
      new_trange1_v.push_back(i);
    }
    new_trange1_v.push_back(range_size);
    new_trange1 = TA::TiledRange1(new_trange1_v.begin(), new_trange1_v.end());
  }

  /// make the new range for N+1 Array
  TA::TiledRange new_trange;
  {
    auto old_trange1s = array_trange.data();
    old_trange1s.insert(old_trange1s.begin(), new_trange1);
    new_trange = TA::TiledRange(old_trange1s.begin(), old_trange1s.end());
  }

  return new_trange;
}

/// @brief fuses the Shapes of a vector of Arrays into 1 Shape, with the vector index forming the first mode

/// @param fused_trange the TiledRange of the fused @c arrays
/// @return Shape of fused Array object
template <typename Tile>
TA::DenseShape fuse_vector_of_shapes(
    const std::vector<TA::DistArray<Tile, TA::DensePolicy>>&,
    const TA::TiledRange& fused_trange) {
  return TA::DenseShape(1, fused_trange);
}

/// @brief fuses the Shapes of a vector of Arrays into 1 Shape, with the vector index forming the first mode

/// @param fused_trange the TiledRange of the fused @c arrays
/// @return Shape of fused Array object
template <typename Tile>
TA::DenseShape fuse_vector_of_shapes(madness::World&,
        const std::vector<TA::DistArray<Tile, TA::DensePolicy>>&,
        int,
        const TA::TiledRange& fused_trange) {
  return TA::DenseShape(1, fused_trange);
}

/// @brief fuses the Shapes of a vector of Arrays into 1 Shape, with the vector index forming the first mode

/// @param[in] arrays a vector of DistArray objects; all members of @c arrays must have the same TiledRange
/// @param[in] fused_trange the TiledRange of the fused @c arrays
/// @return Shape of fused Array object
template <typename Tile>
TA::SparseShape<float> fuse_vector_of_shapes(
    const std::vector<TA::DistArray<Tile, TA::SparsePolicy>>& arrays,
    const TA::TiledRange& fused_trange) {
  auto first_tile_in_mode0 = *fused_trange.dim(0).begin();
  const auto block_size =
      first_tile_in_mode0.second - first_tile_in_mode0.first;

  std::size_t ntiles_per_array = arrays[0].trange().tiles_range().volume();
  // precompute tile volumes for later repeated use
  std::vector<size_t> tile_volumes(ntiles_per_array);
  {
    const auto& tiles_range = arrays[0].trange().tiles_range();
    for (auto&& tile_idx : tiles_range) {
      const auto tile_ord = tiles_range.ordinal(tile_idx);
      tile_volumes[tile_ord] =
          arrays[0].trange().make_tile_range(tile_idx).volume();
    }
  }

  TA::Tensor<float> fused_tile_norms(fused_trange.tiles_range());

  // compute norms of fused tiles
  // N.B. tile norms are stored in scaled format, unscale in order to compute norms of fused tiles
  std::size_t narrays = arrays.size();
  size_t fused_tile_ord = 0;
  for (size_t vidx = 0, fused_vidx = 0; vidx < narrays;
       vidx += block_size, ++fused_vidx) {
    // how many arrays actually constribute to this fused tile ... last fused tile may have fewer than block_size
    const auto vblk_size =
        (narrays - vidx) >= block_size ? block_size : narrays - vidx;
    for (size_t tile_ord = 0; tile_ord != ntiles_per_array;
         ++tile_ord, ++fused_tile_ord) {
      float unscaled_fused_tile_norm2 = 0;
      const auto tile_volume = tile_volumes[tile_ord];
      for (size_t v = 0, vv = vidx; v != vblk_size; ++v, ++vv) {
        const auto unscaled_tile_norm =
            arrays[vv].shape().data()[tile_ord] * tile_volume;
        unscaled_fused_tile_norm2 += unscaled_tile_norm * unscaled_tile_norm;
      }
      const auto fused_tile_volume = tile_volume * vblk_size;
      const auto fused_tile_norm =
          std::sqrt(unscaled_fused_tile_norm2) / fused_tile_volume;

      *(fused_tile_norms.data() + fused_tile_ord) = fused_tile_norm;
    }
  }

  auto fused_shapes =
      TA::SparseShape<float>(fused_tile_norms, fused_trange, true);

  return fused_shapes;
}

/// @brief fuses the Shapes of a vector of Arrays into 1 Shape, with the vector index forming the first mode
///
/// @param global_world the world object which the new fused array will live in.
/// @param[in] arrays a vector of DistArray objects; all members of @c arrays must have the same TiledRange
/// @param array_rank Number of tensors in the fused @c arrays (the size of @c arrays on each rank will depend on world.size)
/// @param[in] fused_trange the TiledRange of the fused @c arrays
/// @return Shape of fused Array object
template <typename Tile>
TA::SparseShape<float> fuse_vector_of_shapes(
    madness::World& global_world,
    const std::vector<TA::DistArray<Tile, TA::SparsePolicy>>& arrays,
    const std::size_t array_rank,
    const TA::TiledRange& fused_trange) {
  const std::size_t rank = global_world.rank();
  auto size = global_world.size();
  auto first_tile_in_mode0 = *fused_trange.dim(0).begin();
  const auto block_size =
      first_tile_in_mode0.second - first_tile_in_mode0.first;

  std::size_t ntiles_per_array = arrays[rank].trange().tiles_range().volume();
  // precompute tile volumes for later repeated use
  std::vector<size_t> tile_volumes(ntiles_per_array);
  {
    const auto& tiles_range = arrays[0].trange().tiles_range();
    for (auto&& tile_idx : tiles_range) {
      const auto tile_ord = tiles_range.ordinal(tile_idx);
      tile_volumes[tile_ord] =
          arrays[rank].trange().make_tile_range(tile_idx).volume();
    }
  }

  TA::Tensor<float> fused_tile_norms(fused_trange.tiles_range());

  // compute norms of fused tiles
  // N.B. tile norms are stored in scaled format, unscale in order to compute norms of fused tiles
  std::size_t narrays = array_rank;
  size_t fused_tile_ord = 0;
  double divisor = 1.0 / (double) size;
  for (size_t vidx = 0, fused_vidx = 0; vidx < narrays;
       vidx += block_size, ++fused_vidx) {
    // how many arrays actually constribute to this fused tile ... last fused tile may have fewer than block_size
    const auto vblk_size =
        (narrays - vidx) >= block_size ? block_size : narrays - vidx;
    for (size_t tile_ord = 0; tile_ord != ntiles_per_array;
         ++tile_ord, ++fused_tile_ord) {
      float unscaled_fused_tile_norm2 = 0;
      const auto tile_volume = tile_volumes[tile_ord];
      for (size_t v = 0, vv = vidx; v != vblk_size; ++v, ++vv) {
        if (rank == vv % size) {
          int dim = (int) (vv * divisor);
          const auto unscaled_tile_norm =
              arrays[dim].shape().data()[tile_ord] * tile_volume;
          unscaled_fused_tile_norm2 += unscaled_tile_norm * unscaled_tile_norm;
        }
      }
      const auto fused_tile_volume = tile_volume * vblk_size;
      const auto fused_tile_norm =
          std::sqrt(unscaled_fused_tile_norm2) / fused_tile_volume;

      *(fused_tile_norms.data() + fused_tile_ord) = fused_tile_norm;
    }
  }

  auto fused_shapes =
      TA::SparseShape<float>(global_world, fused_tile_norms, fused_trange, true);

  return fused_shapes;
}

/// @brief extracts the shape of a subarray of a fused array created with fuse_vector_of_arrays

/// @param[in] fused_array a DistArray created with fuse_vector_of_arrays
/// @param[in] i the index of the subarray whose Shape will be extracted (i.e. the index of the corresponding tile of the leading dimension)
/// @param[in] split_trange TiledRange of the target subarray objct
/// @return the Shape of the @c i -th subarray
template <typename Tile>
TA::DenseShape subshape_from_fused_array(
    const TA::DistArray<Tile, TA::DensePolicy>& fused_array,
    const std::size_t i, const TA::TiledRange& split_trange) {
  return TA::DenseShape(1, split_trange);
}

/// @brief extracts the shape of a subarray of a fused array created with fuse_vector_of_arrays

/// @param[in] fused_array a DistArray created with fuse_vector_of_arrays
/// @param[in] i the index of the subarray whose Shape will be extracted
///            (i.e. the index of the corresponding *element* index of the
///            leading dimension)
/// @param[in] split_trange TiledRange of the target subarray object
/// @return the Shape of the @c i -th subarray
template <typename Tile>
TA::SparseShape<float> subshape_from_fused_array(
    const TA::DistArray<Tile, TA::SparsePolicy>& fused_array,
    const std::size_t i, const TA::TiledRange& split_trange) {
  TA_ASSERT(i < fused_array.trange().dim(0).extent());

  std::size_t split_array_ntiles = split_trange.tiles_range().volume();

  TA::Tensor<float> split_tile_norms(split_trange.tiles_range());

  // map element i to its tile index
  const auto tile_idx_of_i = fused_array.trange().dim(0).element_to_tile(i);
  std::size_t offset = tile_idx_of_i * split_array_ntiles;
  const auto tile_of_i = fused_array.trange().dim(0).tile(tile_idx_of_i);
  const float extent_of_tile_of_i = tile_of_i.second - tile_of_i.first;
  auto& shape = fused_array.shape();

  // note that unlike fusion we cannot compute exact norm of the split tile
  // to guarantee upper bound we have to multiply the norms by the number of
  // split tiles in the fused tile; to see why multiplication is necessary think
  // of a tile obtained by fusing 1 nonzero tile with one or more zero tiles.
  const auto* split_tile_begin = shape.data().data() + offset;
  std::transform(split_tile_begin, split_tile_begin + split_array_ntiles,
                 split_tile_norms.data(),
                 [extent_of_tile_of_i](const float& elem) {
                   return elem * extent_of_tile_of_i;
                 });

  auto split_shape =
      TA::SparseShape<float>(split_tile_norms, split_trange, true);

  return split_shape;
}

}  // namespace detail

/// @brief fuses a vector of DistArray objects, each with the same TiledRange into a DistArray with 1 more dimensions

/// The leading dimension of the resulting array is the vector dimension, and will be blocked by @block_size .
///
/// @param[in] arrays a vector of DistArray objects; every element of @c arrays must have the same TiledRange object and live in the same world.
/// @param[in] block_size the block size for the "vector" dimension of the tiled range of the result
/// @return @c arrays fused into a DistArray
/// @note This is a collective function. It assumes that it is invoked across the world in which all subarrays live; the result will live in the same world.
template <typename Tile, typename Policy>
TA::DistArray<Tile, Policy> fuse_vector_of_arrays(
    const std::vector<TA::DistArray<Tile, Policy>>& arrays,
    std::size_t block_size = 1) {
  auto& world = arrays[0].world();
  auto array_trange = arrays[0].trange();
  const auto mode0_extent = arrays.size();

  // make fused tiledrange
  auto fused_trange = detail::fuse_vector_of_tranges(arrays.size(), arrays.at(0).trange(), block_size);
  std::size_t ntiles_per_array = array_trange.tiles_range().volume();

  // make fused shape
  auto fused_shape = detail::fuse_vector_of_shapes(arrays, fused_trange);

  // make fused array
  TA::DistArray<Tile, Policy> fused_array(world, fused_trange, fused_shape);

  /// copy the data from a sequence of tiles
  auto make_tile = [](const TA::Range& range,
                      const std::vector<madness::Future<Tile>>& tiles) {
    TA_ASSERT(range.extent(0) == tiles.size());
    Tile result(range);
    auto* result_ptr = result.data();
    for (auto&& fut_of_tile : tiles) {
      TA_ASSERT(fut_of_tile.probe());
      const auto& tile = fut_of_tile.get();
      const auto* tile_data = tile.data();
      const auto tile_volume = tile.size();
      std::copy(tile_data, tile_data + tile_volume, result_ptr);
      result_ptr += tile_volume;
    }
    return result;
  };

  /// write to blocks of fused_array
  for (auto&& fused_tile_ord : *fused_array.pmap()) {
    if (!fused_array.is_zero(fused_tile_ord)) {
      // convert ordinal of the fused tile to the ordinals of its consituent tiles
      const auto div = std::ldiv(fused_tile_ord, ntiles_per_array);
      const auto tile_idx_mode0 = div.quot;
      // ordinal of the coresponding tile in the arrays
      const auto tile_ord_array = div.rem;

      auto fused_tile_range =
          fused_array.trange().make_tile_range(fused_tile_ord);
      // make a vector of Futures to the input tiles
      std::vector<madness::Future<Tile>> input_tiles;
      input_tiles.reserve(fused_tile_range.extent(0));
      for (size_t v = 0, vidx = tile_idx_mode0 * block_size;
           v != block_size && vidx < mode0_extent; ++v, ++vidx) {
        input_tiles.emplace_back(arrays[vidx].find(tile_ord_array));
      }
      fused_array.set(fused_tile_ord,
                      world.taskq.add(make_tile, std::move(fused_tile_range),
                                      std::move(input_tiles)));
    }
  }

  return fused_array;
}

namespace detail {
/// @brief global view of a distributed vector of local arrays
template <typename Array>
class dist_subarray_vec
    : public madness::WorldObject<dist_subarray_vec<Array>> {
 public:
  using Tile = typename Array::value_type;
  using Policy = typename Array::policy_type;

  /// @param world world object that contains all the worlds which arrays in @c split_array live in
  /// @param array possibly distributed vector of arrays
  /// @param rank total number of Arrays (sum of arrays per process for each processor)
  dist_subarray_vec(madness::World& world, const std::vector<Array>& array, const std::size_t rank)
      : madness::WorldObject<dist_subarray_vec<Array>>(world),
        split_array(array),
        rank_(rank){
    this->process_pending();
  }

  virtual ~dist_subarray_vec() {}

  /// Tile accessor for distributed arrays
  /// @param r index of the requested array in @c split_array
  /// @param i tile index for the @c r array in @c split_array
  /// @return @c i -th tile of the @c r -th array in @c split_array
  template <typename Index>
  madness::Future<Tile> get_tile(int r, Index& i) {
    return split_array.at(r).find(i);
  }

  /// @return Accessor to @c split_array
  const std::vector<Array>& array_accessor() const { return split_array; }

  /// @return number of Array in @c split_array
  unsigned long size() const { return rank_; }

 private:
  const std::vector<Array>& split_array;
  const int rank_;
};
}

/// @brief fuses a vector of DistArray objects, each with the same TiledRange into a DistArray with 1 more dimensions

/// The leading dimension of the resulting array is the vector dimension, and will be blocked by @block_size .
///
/// @param global_world the world in which the result will live and across which this is invoked.
/// @param[in] arrays a vector of DistArray objects; every element of @c arrays must have the same TiledRange object and live in the same world.
/// @param array_rank total number of arrays in a fused @c arrays (sum of @c arrays.size() on each rank)
/// @param[in] block_size the block size for the "vector" dimension of the tiled range of the result
/// @return @c arrays fused into a DistArray
/// @note This is a collective function. It assumes that it is invoked across @c global_world, but the subarrays are "local" to each rank and distributed in round-robin fashion.
///       The result will live in @c global_world.
template <typename Tile, typename Policy>
TA::DistArray<Tile, Policy> fuse_vector_of_arrays(
    madness::World& global_world,
    const std::vector<TA::DistArray<Tile, Policy>>& array_vec,
    const std::size_t array_rank,
    const TiledArray::TiledRange& array_trange, std::size_t block_size = 1) {
  auto size = global_world.size();

  // make instances of array_vec globally accessible
  using Array = TA::DistArray<Tile, Policy>;
  detail::dist_subarray_vec<Array> arrays(global_world, array_vec, array_rank);

  const auto mode0_extent = array_rank;

  // make fused tiledrange
  auto fused_trange = detail::fuse_vector_of_tranges(array_rank,
                                                     array_trange, block_size);
  std::size_t ntiles_per_array = array_trange.tiles_range().volume();

  // make fused shape
  auto fused_shape = detail::fuse_vector_of_shapes(
      global_world, arrays.array_accessor(), array_rank, fused_trange);

  // make fused array
  TA::DistArray<Tile, Policy> fused_array(global_world, fused_trange,
                                          fused_shape);

  /// copy the data from a sequence of tiles
  auto make_tile = [](const TA::Range& range,
                      const std::vector<madness::Future<Tile>>& tiles) {
    TA_ASSERT(range.extent(0) == tiles.size());
    Tile result(range);
    const auto volume = range.volume();
    size_t result_volume = 0;
    auto* result_ptr = result.data();
    for (auto&& fut_of_tile : tiles) {
      TA_ASSERT(fut_of_tile.probe());
      const auto& tile = fut_of_tile.get();
      const auto* tile_data = tile.data();
      const auto tile_volume = tile.size();
      std::copy(tile_data, tile_data + tile_volume, result_ptr);
      result_ptr += tile_volume;
      result_volume += tile_volume;
    }
    TA_ASSERT(volume == result_volume);
    return result;
  };

  /// write to blocks of fused_array
  auto divisor = 1.0 / (double) size;
  for (auto&& fused_tile_ord : *fused_array.pmap()) {
    if (!fused_array.is_zero(fused_tile_ord)) {
      // convert ordinal of the fused tile to the ordinals of its constituent tiles
      const auto div = std::ldiv(fused_tile_ord, ntiles_per_array);
      const auto tile_idx_mode0 = div.quot;
      // ordinal of the corresponding tile in the arrays
      const auto tile_ord_array = div.rem;
      using Index = decltype(tile_ord_array);

      auto fused_tile_range =
          fused_array.trange().make_tile_range(fused_tile_ord);
      // make a vector of Futures to the input tiles
      std::vector<madness::Future<Tile>> input_tiles;
      input_tiles.reserve(fused_tile_range.extent(0));
      for (size_t v = 0, vidx = tile_idx_mode0 * block_size;
           v != block_size && vidx < mode0_extent; ++v, ++vidx) {
        int owner_rank = vidx % size;
        int new_vidx = (int) (vidx * divisor);

        input_tiles.emplace_back(
            arrays.task(owner_rank,
                        &detail::dist_subarray_vec<
                            DistArray<Tile, Policy>>::template get_tile<Index>,
                        new_vidx, tile_ord_array));
      }
      fused_array.set(
          fused_tile_ord,
          global_world.taskq.add(make_tile, std::move(fused_tile_range),
                                 std::move(input_tiles)));
    }
  }

  // keep arrays around until everyone is done
  global_world.gop.fence();

  return fused_array;
}

/// @brief extracts a subarray of a fused array created with fuse_vector_of_arrays

/// @param[in] fused_array a DistArray created with fuse_vector_of_arrays
/// @param[in] i the index of the subarray to be extracted
///            (i.e. the index of the corresponding *element* index of the
///            leading dimension)
/// @param[in] split_trange TiledRange of the split Array object
/// @return the @c i -th subarray
template <typename Tile, typename Policy>
TA::DistArray<Tile, Policy> subarray_from_fused_array(
    const TA::DistArray<Tile, Policy>& fused_array, std::size_t i,
    const TA::TiledRange& split_trange) {
  auto& world = fused_array.world();

  // get the shape of split Array
  auto split_shape =
      detail::subshape_from_fused_array(fused_array, i, split_trange);

  // determine where subtile i starts
  size_t i_offset_in_tile;
  size_t tile_idx_of_i;
  {
    tile_idx_of_i = fused_array.trange().dim(0).element_to_tile(i);
    const auto tile_of_i = fused_array.trange().dim(0).tile(tile_idx_of_i);
    TA_ASSERT(i >= tile_of_i.first && i < tile_of_i.second);
    i_offset_in_tile = i - tile_of_i.first;
  }

  // create split Array object
  TA::DistArray<Tile, Policy> split_array(world, split_trange, split_shape);

  std::size_t split_ntiles = split_trange.tiles_range().volume();

  /// copy the data from tile
  auto make_tile = [i_offset_in_tile](const TA::Range& range,
                                      const Tile& fused_tile) {
    const auto split_tile_volume = range.volume();
    return Tile(range,
                fused_tile.data() + i_offset_in_tile * split_tile_volume);
  };

  /// write to blocks of fused_array
  for (std::size_t index : *split_array.pmap()) {
    if (!split_array.is_zero(index)) {
      std::size_t fused_array_index = tile_idx_of_i * split_ntiles + index;

      split_array.set(
          index, world.taskq.add(make_tile,
                                 split_array.trange().make_tile_range(index),
                                 fused_array.find(fused_array_index)));
    }
  }

  return split_array;
}

/// @brief extracts a subarray of a fused array created with fuse_vector_of_arrays
/// and creates the array in @c local_world.

/// @param[in] local_world The World object where the @i -th subarray is
///             created
/// @param[in] fused_array a DistArray created with fuse_vector_of_arrays
/// @param[in] i the index of the subarray to be extracted
///            (i.e. the index of the corresponding *element* index of the
///            leading dimension)
/// @param[in] split_trange TiledRange of the split Array object
/// @return the @c i -th subarray
template <typename Tile, typename Policy>
TA::DistArray<Tile, Policy> subarray_from_fused_array(
    madness::World& local_world, const TA::DistArray<Tile, Policy>& fused_array,
    std::size_t i, const TA::TiledRange& split_trange) {
  // get the shape of split Array
  auto split_shape =
      detail::subshape_from_fused_array(fused_array, i, split_trange);

  // determine where subtile i starts
  size_t i_offset_in_tile;
  size_t tile_idx_of_i;
  {
    tile_idx_of_i = fused_array.trange().dim(0).element_to_tile(i);
    const auto tile_of_i = fused_array.trange().dim(0).tile(tile_idx_of_i);
    TA_ASSERT(i >= tile_of_i.first && i < tile_of_i.second);
    i_offset_in_tile = i - tile_of_i.first;
  }

  // create split Array object
  TA::DistArray<Tile, Policy> split_array(local_world, split_trange,
                                          split_shape);

  std::size_t split_ntiles = split_trange.tiles_range().volume();

  /// copy the data from tile
  auto make_tile = [i_offset_in_tile](const TA::Range& range,
                                      const Tile& fused_tile) {
    const auto split_tile_volume = range.volume();
    return Tile(range,
                fused_tile.data() + i_offset_in_tile * split_tile_volume);
  };

  /// write to blocks of fused_array
  for (std::size_t index : *split_array.pmap()) {
    if (!split_array.is_zero(index)) {
      std::size_t fused_array_index = tile_idx_of_i * split_ntiles + index;

      split_array.set(
          index, local_world.taskq.add(
                     make_tile, split_array.trange().make_tile_range(index),
                     fused_array.find(fused_array_index)));
    }
  }

  return split_array;
}

}  // namespace TiledArray

#endif  // TILEDARRAY_CONVERSIONS_VECTOR_OF_ARRAYS_H_
