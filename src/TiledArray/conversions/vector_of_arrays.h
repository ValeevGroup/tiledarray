//
// Created by Chong Peng on 2019-05-01.
//

#ifndef TILEDARRAY_CONVERSIONS_VECTOR_OF_ARRAYS_H_
#define TILEDARRAY_CONVERSIONS_VECTOR_OF_ARRAYS_H_

#include <tiledarray.h>

namespace TiledArray {

namespace detail {

/// @brief prepends an extra dimension to a TRange

/// The extra dimension will be the leading dimension, and will be blocked by @c
/// block_size

/// @param array_rank extent of the leading dimension of the result
/// @param array_trange the base trange
/// @param block_size blocking range for the new dimension, the dimension being
/// fused
/// @return TiledRange of fused Array object
inline TiledArray::TiledRange prepend_dim_to_trange(
    std::size_t array_rank, const TiledArray::TiledRange& array_trange,
    std::size_t block_size = 1) {
  /// make the new TiledRange1 for new dimension
  TiledArray::TiledRange1 new_trange1;
  {
    std::vector<std::size_t> new_trange1_v;
    auto range_size = array_rank;
    new_trange1_v.push_back(0);
    for (decltype(range_size) i = block_size; i < range_size; i += block_size) {
      new_trange1_v.push_back(i);
    }
    new_trange1_v.push_back(range_size);
    new_trange1 =
        TiledArray::TiledRange1(new_trange1_v.begin(), new_trange1_v.end());
  }

  /// make the new range for N+1 Array
  TiledArray::TiledRange new_trange;
  {
    auto old_trange1s = array_trange.data();
    old_trange1s.insert(old_trange1s.begin(), new_trange1);
    new_trange =
        TiledArray::TiledRange(old_trange1s.begin(), old_trange1s.end());
  }

  return new_trange;
}

/// @brief fuses the SparseShape objects of a tilewise-round-robin distributed
///        vector of Arrays into single SparseShape object,
///        with the vector index forming the first dimension.
///
/// This is used to fuse shapes of a sequence of N-dimensional arrays
/// into the shape of the fused (N+1)-dimensional array. The sequence is
/// *tiled*, and the tiles are round-robin distributed. Hence for p ranks tile I
/// will reside on processor I%p ; this tile includes shapes of arrays [I*b,
/// (I+1)*b).
///
/// @param global_world the World object in which the new fused array will live
/// @param[in] arrays a vector of DistArray objects; these are local components
///            of a vector distributed tilewise in round-robin manner; each
///            element of @c arrays must have the same TiledRange;
/// @param array_rank Sum of the sizes of @c arrays on each rank
///        (the size of @c arrays on each rank will depend on world.size)
/// @param[in] fused_trange the TiledRange of the fused @c arrays
/// @return SparseShape of fused Array object
/// TODO rename to fuse_tilewise_vector_of_shapes
template <typename Tile>
TiledArray::SparseShape<float> fuse_tilewise_vector_of_shapes(
    madness::World& global_world,
    const std::vector<TiledArray::DistArray<Tile, TiledArray::SparsePolicy>>&
        arrays,
    const std::size_t array_rank, const TiledArray::TiledRange& fused_trange) {
  if (arrays.size() == 0) {
    TiledArray::Tensor<float> fused_tile_norms(fused_trange.tiles_range(), 0.f);
    return TiledArray::SparseShape<float>(global_world, fused_tile_norms,
                                          fused_trange, true);
  }
  const std::size_t rank = global_world.rank();
  auto size = global_world.size();
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

  TiledArray::Tensor<float> fused_tile_norms(
      fused_trange.tiles_range(), 0.f);  // use nonzeroes for local tiles only

  // compute norms of fused tiles
  // N.B. tile norms are stored in scaled format, unscale in order to compute
  // norms of fused tiles
  std::size_t narrays = array_rank;
  size_t fused_tile_ord = 0;
  auto element_offset_in_owner = 0;
  for (size_t vidx = 0, fused_vidx = 0; vidx < narrays;
       vidx += block_size, ++fused_vidx) {
    bool have_rank = (rank == fused_vidx % size);
    // how many arrays actually constribute to this fused tile ... last fused
    // tile may have fewer than block_size
    if (have_rank) {
      const auto vblk_size =
          (narrays - vidx) >= block_size ? block_size : narrays - vidx;
      for (size_t tile_ord = 0; tile_ord != ntiles_per_array;
           ++tile_ord, ++fused_tile_ord) {
        auto array_ptr = arrays.begin() + element_offset_in_owner * vblk_size;
        float unscaled_fused_tile_norm2 = 0;
        const auto tile_volume = tile_volumes[tile_ord];
        for (size_t v = 0, vv = vidx; v != vblk_size; ++v, ++vv) {
          const auto unscaled_tile_norm =
              (*(array_ptr)).shape().data()[tile_ord] * tile_volume;
          unscaled_fused_tile_norm2 += unscaled_tile_norm * unscaled_tile_norm;
          ++array_ptr;
        }
        const auto fused_tile_volume = tile_volume * vblk_size;
        const auto fused_tile_norm =
            std::sqrt(unscaled_fused_tile_norm2) / fused_tile_volume;

        *(fused_tile_norms.data() + fused_tile_ord) = fused_tile_norm;
      }
      element_offset_in_owner += 1;
    } else {
      fused_tile_ord += ntiles_per_array;
    }
  }

  auto fused_shapes = TiledArray::SparseShape<float>(
      global_world, fused_tile_norms, fused_trange, true);

  return fused_shapes;
}

/// @brief fuses the DenseShape objects of a tilewise-round-robin distributed
///        vector of Arrays into single DenseShape object,
///        with the vector index forming the first dimension.
///
/// This is the same as the sparse version above, but far simpler.
///
/// @param global_world the World object in which the new fused array will live
/// @param[in] arrays a vector of DistArray objects; these are local components
///            of a vector distributed tilewise in round-robin manner; each
///            element of @c arrays must have the same TiledRange;
/// @param array_rank Sum of the sizes of @c arrays on each rank
///        (the size of @c arrays on each rank will depend on world.size)
/// @param[in] fused_trange the TiledRange of the fused @c arrays
/// @return DenseShape of fused Array object
template <typename Tile>
TiledArray::DenseShape fuse_tilewise_vector_of_shapes(
    madness::World&,
    const std::vector<TiledArray::DistArray<Tile, TiledArray::DensePolicy>>&
        arrays,
    const std::size_t array_rank, const TiledArray::TiledRange& fused_trange) {
  return TiledArray::DenseShape(1, fused_trange);
}

/// @brief extracts the shape of a slice of a fused array created with
/// fuse_vector_of_arrays

/// @param[in] split_trange the TiledRange object of each "slice" array that was
/// fused via fuse_vector_of_arrays
/// @param[in] shape the shape of a DistArray created with fuse_vector_of_arrays
/// @param[in] tile_idx the tile index of the leading mode that will be sliced
/// off
/// @param[in] split_ntiles the number of tiles in each "slice" array that was
/// fused via fuse_vector_of_arrays
/// @param[in] tile_size the size of the tile of the leading dimension of the
/// fused array
/// @return the Shape of the @c i -th subarray
// TODO rename to tilewise_slice_of_fused_shape
inline TiledArray::SparseShape<float> tilewise_slice_of_fused_shape(
    const TiledArray::TiledRange& split_trange,
    const TiledArray::SparsePolicy::shape_type& shape,
    const std::size_t tile_idx, const std::size_t split_ntiles,
    const std::size_t tile_size) {
  TA_ASSERT(split_ntiles == split_trange.tiles_range().volume());
  TiledArray::Tensor<float> split_tile_norms(split_trange.tiles_range());

  // map element i to its tile index
  std::size_t offset = tile_idx * split_ntiles;

  // note that unlike fusion we cannot compute exact norm of the split tile
  // to guarantee upper bound we have to multiply the norms by the number of
  // split tiles in the fused tile; to see why multiplication is necessary think
  // of a tile obtained by fusing 1 nonzero tile with one or more zero tiles.
  const auto* split_tile_begin = shape.data().data() + offset;
  std::transform(split_tile_begin, split_tile_begin + split_ntiles,
                 split_tile_norms.data(),
                 [tile_size](const float& elem) { return elem * tile_size; });

  auto split_shape =
      TiledArray::SparseShape<float>(split_tile_norms, split_trange, true);
  return split_shape;
}

/// @brief extracts the shape of a subarray of a fused array created with
/// fuse_vector_of_arrays

/// @param[in] fused_array a DistArray created with fuse_vector_of_arrays
/// @param[in] i the index of the subarray whose Shape will be extracted (i.e.
/// the index of the corresponding tile of the leading dimension)
/// @param[in] split_trange TiledRange of the target subarray objct
/// @return the Shape of the @c i -th subarray
inline TiledArray::DenseShape tilewise_slice_of_fused_shape(
    const TiledArray::TiledRange& split_trange,
    const TiledArray::DensePolicy::shape_type& shape,
    const std::size_t tile_idx, const std::size_t split_ntiles,
    const std::size_t tile_size) {
  return TiledArray::DenseShape(tile_size, split_trange);
}
}  // namespace detail

namespace detail {
/// @brief global view of a tilewise-round-robin distributed std::vector of
/// Arrays
template <typename Array>
class dist_subarray_vec
    : public madness::WorldObject<dist_subarray_vec<Array>> {
 public:
  using Tile = typename Array::value_type;
  using Policy = typename Array::policy_type;

  /// @param world world object that contains all the worlds which arrays in @c
  /// split_array live in
  /// @param array possibly distributed vector of arrays
  /// @param rank total number of Arrays (sum of arrays per process for each
  /// processor)
  dist_subarray_vec(madness::World& world, const std::vector<Array>& array,
                    const std::size_t rank)
      : madness::WorldObject<dist_subarray_vec<Array>>(world),
        split_array(array),
        rank_(rank) {
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
}  // namespace detail

/// @brief fuses a vector of DistArray objects, each with the same TiledRange
/// into a DistArray with 1 more dimensions

/// The leading dimension of the resulting array is the vector dimension, and
/// will be blocked by @block_size .
///
/// @param global_world the world in which the result will live and across which
/// this is invoked.
/// @param[in] array_vec a vector of DistArray objects; every element of @c
/// arrays must have the same TiledRange object and live in the same world.
/// @param[in] fused_dim_extent the extent of the resulting (fused) mode; equals
/// the total number of arrays in a fused @c arrays (sum of @c arrays.size() on
/// each rank)
/// @param[in] block_size the block size for the "vector" dimension of the tiled
/// range of the result
/// @return @c arrays fused into a DistArray
/// @note This is a collective function. It assumes that it is invoked across @c
/// global_world, but the subarrays are "local" to each rank and distributed in
/// tilewise-round-robin fashion.
///       The result will live in @c global_world.
/// @sa detail::fuse_tilewise_vector_of_shapes
template <typename Tile, typename Policy>
TiledArray::DistArray<Tile, Policy> fuse_tilewise_vector_of_arrays(
    madness::World& global_world,
    const std::vector<TiledArray::DistArray<Tile, Policy>>& array_vec,
    const std::size_t fused_dim_extent,
    const TiledArray::TiledRange& array_trange,
    std::size_t target_block_size = 1) {
  auto nproc = global_world.size();

  // make instances of array_vec globally accessible
  using Array = TiledArray::DistArray<Tile, Policy>;
  detail::dist_subarray_vec<Array> arrays(global_world, array_vec,
                                          fused_dim_extent);

  std::size_t nblocks =
      (fused_dim_extent + target_block_size - 1) / target_block_size;
  std::size_t block_size = (fused_dim_extent + nblocks - 1) / nblocks;
  // make fused tiledrange
  auto fused_trange =
      detail::prepend_dim_to_trange(fused_dim_extent, array_trange, block_size);
  std::size_t ntiles_per_array = array_trange.tiles_range().volume();

  // make fused shape
  auto fused_shape = detail::fuse_tilewise_vector_of_shapes(
      global_world, arrays.array_accessor(), fused_dim_extent, fused_trange);

  // make fused array
  TiledArray::DistArray<Tile, Policy> fused_array(global_world, fused_trange,
                                                  fused_shape);

  /// copy the data from a sequence of tiles
  auto make_tile = [](const TiledArray::Range& range,
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
  for (auto&& fused_tile_ord : *fused_array.pmap()) {
    if (!fused_array.is_zero(fused_tile_ord)) {
      // convert ordinal of the fused tile to the ordinals of its constituent
      // tiles
      const auto div0 = std::ldiv(fused_tile_ord, ntiles_per_array);
      // index of the 0th mode of this tile
      const auto tile_idx_mode0 = div0.quot;
      // ordinal of the corresponding tile in the unfused array
      const auto tile_ord_array = div0.rem;

      const auto div1 = std::ldiv(tile_idx_mode0, nproc);
      const auto tile_idx_on_owner = div1.quot;
      const auto vector_idx_offset_on_owner = tile_idx_on_owner * block_size;
      const auto owner_rank = div1.rem;

      auto fused_tile_range =
          fused_array.trange().make_tile_range(fused_tile_ord);
      // make a vector of Futures to the input tiles
      std::vector<madness::Future<Tile>> input_tiles;
      input_tiles.reserve(fused_tile_range.extent(0));
      for (size_t v = 0, vidx = tile_idx_mode0 * block_size;
           v != block_size && vidx < fused_dim_extent; ++v, ++vidx) {
        using Index = decltype(tile_ord_array);
        input_tiles.emplace_back(
            arrays.task(owner_rank,
                        &detail::dist_subarray_vec<
                            DistArray<Tile, Policy>>::template get_tile<Index>,
                        vector_idx_offset_on_owner + v, tile_ord_array));
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

/// @brief extracts a subarray of a fused array created with
/// fuse_vector_of_arrays and creates the array in @c local_world.

/// @param[in] local_world The World object where the @i -th subarray is
///             created
/// @param[in] fused_array a DistArray created with fuse_vector_of_arrays
/// @param[in] i the index of the subarray to be extracted
///            (i.e. the index of the corresponding *element* index of the
///            leading dimension)
/// @param[in] tile_of_i tile range information for tile i
/// @param[in] split_trange TiledRange of the split Array object
/// @return the @c i -th subarray
/// @sa detail::tilewise_slice_of_fused_shape
template <typename Tile, typename Policy>
void split_tilewise_fused_array(
    madness::World& local_world,
    const TiledArray::DistArray<Tile, Policy>& fused_array,
    std::size_t tile_idx,
    std::vector<TiledArray::DistArray<Tile, Policy>>& split_arrays,
    const TiledArray::TiledRange& split_trange) {
  TA_ASSERT(tile_idx < fused_array.trange().dim(0).extent());
  auto arrays_size = split_arrays.size();

  // calculate the number of elements in the 0th dimension are in this tile
  auto tile_range = fused_array.trange().dim(0).tile(tile_idx);
  auto tile_size = tile_range.second - tile_range.first;
  std::size_t split_ntiles = split_trange.tiles_range().volume();
  auto& shape = fused_array.shape();

  // Create tile_size arrays and put them into split_arrays
  for (size_t i = tile_range.first; i < tile_range.second; ++i) {
    auto split_shape = detail::tilewise_slice_of_fused_shape(
        split_trange, shape, tile_idx, split_ntiles, tile_size);
    // create split Array object
    TiledArray::DistArray<Tile, Policy> split_array(local_world, split_trange,
                                                    split_shape);
    split_arrays.push_back(split_array);
  }

  /// copy the data from tile
  auto make_tile = [](const TiledArray::Range& range, const Tile& fused_tile,
                      const size_t i_offset_in_tile) {
    const auto split_tile_volume = range.volume();
    return Tile(range,
                fused_tile.data() + i_offset_in_tile * split_tile_volume);
  };

  /// write to blocks of fused_array
  auto split_array_ptr = split_arrays.data();
  for (std::size_t index : *(*split_array_ptr).pmap()) {
    std::size_t fused_array_index = tile_idx * split_ntiles + index;
    if (!fused_array.is_zero(fused_array_index)) {
      for (std::size_t i = tile_range.first, tile_count = 0;
           i < tile_range.second; ++i, ++tile_count) {
        auto& array = *(split_array_ptr + arrays_size + tile_count);
        array.set(index, local_world.taskq.add(
                             make_tile, array.trange().make_tile_range(index),
                             fused_array.find(fused_array_index), tile_count));
      }
    }
  }
  return;
}

}  // namespace TiledArray

#endif  // TILEDARRAY_CONVERSIONS_VECTOR_OF_ARRAYS_H_
