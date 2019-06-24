//
// Created by Chong Peng on 2019-05-01.
//

#ifndef TILEDARRAY_CONVERSIONS_VECTOR_OF_ARRAYS_H_
#define TILEDARRAY_CONVERSIONS_VECTOR_OF_ARRAYS_H_

#include <tiledarray.h>

namespace TiledArray {

namespace detail {

/// @brief fuses the TRanges of a vector of Arrays into 1 TRange, with the vector index forming the first mode

/// The vector dimension will be the leading dimension, and will be blocked by 1.
/// @warning all arrays in the vector must have the same TiledRange
template <typename Tile, typename Policy>
TA::TiledRange fuse_vector_of_tranges(
        const std::vector<TA::DistArray<Tile, Policy>>& arrays,
        std::size_t block_size) {
  std::size_t n_array = arrays.size();
  auto array_trange = arrays[0].trange();

  /// make the new TiledRange1 for new dimension
  TA::TiledRange1 new_trange1;
  {
    std::vector<std::size_t> new_trange1_v;
    auto range_size = arrays.size();
    new_trange1_v.push_back(0);
    for (std::size_t i = block_size; i < range_size; i += block_size) {
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
///
/// @param arrays a vector of DistArray objects; all members of @c arrays must have the same TiledRange
/// @param trange the TiledRange of the fused @c arrays
template <typename Tile>
TA::DenseShape fuse_vector_of_shapes(
    const std::vector<TA::DistArray<Tile, TA::DensePolicy>>& arrays,
    const TA::TiledRange& trange) {
  return TA::DenseShape(1, trange);
}

/// @brief fuses the Shapes of a vector of Arrays into 1 Shape, with the vector index forming the first mode
///
/// @param arrays a vector of DistArray objects; all members of @c arrays must have the same TiledRange
/// @param trange the TiledRange of the fused @c arrays
template <typename Tile>
TA::SparseShape<float> fuse_vector_of_shapes(
    const std::vector<TA::DistArray<Tile, TA::SparsePolicy>>& arrays,
    const TA::TiledRange& trange) {
  std::size_t array_tiles_volume = arrays[0].trange().tiles_range().volume();

  TA::Tensor<float> fused_tile_norms(trange.tiles_range());

  auto copy_shape = [&array_tiles_volume, &fused_tile_norms](
                        std::size_t i, const TA::Tensor<float>& shape) {
    //    std::cout << shape << std::endl;

    std::copy(shape.data(), shape.data() + array_tiles_volume,
              fused_tile_norms.data() + i * array_tiles_volume);
  };

  std::size_t n_array = arrays.size();
  for (std::size_t i = 0; i < n_array; i++) {
    copy_shape(i, arrays[i].shape().data());
  }

  auto fused_shapes = TA::SparseShape<float>(fused_tile_norms, trange, true);

  //  std::cout << fused_shapes << std::endl;
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
    const std::size_t i,
    const TA::TiledRange& split_trange) {
  return TA::DenseShape(1, split_trange);
}

/// @brief extracts the shape of a subarray of a fused array created with fuse_vector_of_arrays

/// @param[in] fused_array a DistArray created with fuse_vector_of_arrays
/// @param[in] i the index of the subarray whose Shape will be extracted (i.e. the index of the corresponding tile of the leading dimension)
/// @param[in] split_trange TiledRange of the target subarray objct
/// @return the Shape of the @c i -th subarray
template <typename Tile>
TA::SparseShape<float> subshape_from_fused_array(
    const TA::DistArray<Tile, TA::SparsePolicy>& fused_array,
    const std::size_t i,
    const TA::TiledRange& split_trange) {

  std::size_t split_array_tiles_volume = split_trange.tiles_range().volume();

  TA::Tensor<float> split_tile_norms(split_trange.tiles_range());

  std::size_t offset = split_array_tiles_volume * i;

  auto& shape = fused_array.shape();

  std::copy(shape.data().data() + offset, shape.data().data() + offset + split_array_tiles_volume,
            split_tile_norms.data());


  auto split_shape = TA::SparseShape<float>(split_tile_norms, split_trange, true);

  //  std::cout << fused_shapes << std::endl;
  return split_shape;
}

}  // namespace detail

/// @brief fuses a vector of DistArray objects, each with the same TiledRange into a DistArray with 1 more dimensions

/// The leading dimension of the resulting array is the vector dimension, and will
/// be blocked by 1.
///
/// @param[in] arrays a vector of DistArray objects; every element of @c arrays must have the same TiledRange object
/// @return @c arrays fused into a DistArray
template <typename Tile, typename Policy>
TA::DistArray<Tile, Policy> fuse_vector_of_arrays(
    const std::vector<TA::DistArray<Tile, Policy>>& arrays,
    std::size_t block_size = 1) {
  auto& world = arrays[0].world();
  auto array_trange = arrays[0].trange();

  // make fused tiledrange
  auto fused_trange = detail::fuse_vector_of_tranges(arrays, block_size);

  // make fused shape
  auto fused_shape = detail::fuse_vector_of_shapes(arrays, fused_trange);

  // make fused arrays
  TA::DistArray<Tile, Policy> fused_array(world, fused_trange, fused_shape);

  std::size_t old_tiles_volume = array_trange.tiles_range().volume();

  /// copy the data from tile
  auto make_tile = [](const TA::Range& range, const Tile& tile) {
    return Tile(range, tile.data());
  };

  /// write to blocks of fused_array
  for (std::size_t index : *fused_array.pmap()) {
    std::size_t array_id = index / old_tiles_volume;
    std::size_t array_index = index - array_id * old_tiles_volume;

    if (!fused_array.is_zero(index)) {
      auto new_tile = world.taskq.add(
          make_tile, fused_array.trange().make_tile_range(index),
          arrays[array_id].find(array_index));
      fused_array.set(index, new_tile);
    }
  }

  //  for(auto& array : arrays){
  //    std::cout << array << std::endl;
  //  }
  //  std::cout << fused_array << std::endl;

  return fused_array;
}


/// @brief extracts a subarray of a fused array created with fuse_vector_of_arrays

/// @param[in] fused_array a DistArray created with fuse_vector_of_arrays
/// @param[in] i the index of the subarray to extract (i.e. the index of the corresponding tile of the leading dimension)
/// @param[in] split_trange TiledRange of the split Array object
/// @return the @c i -th subarray
template <typename Tile, typename Policy>
TA::DistArray<Tile, Policy> subarray_from_fused_array(
    const TA::DistArray<Tile, Policy>& fused_array, std::size_t i, const TA::TiledRange& split_trange) {

  auto& world = fused_array.world();

  // get the shape of split Array
  auto split_shape = detail::subshape_from_fused_array(fused_array, i, split_trange);

  // create split Array object
  TA::DistArray<Tile,Policy> split_array(world, split_trange, split_shape);

  std::size_t split_tiles_volume = split_trange.tiles_range().volume();

  /// copy the data from tile
  auto make_tile = [](const TA::Range& range, const Tile& tile) {
    return Tile(range, tile.data());
  };

  /// write to blocks of fused_array
  for (std::size_t index : *split_array.pmap()) {

    std::size_t fused_array_index = i*split_tiles_volume + index;

    if (!split_array.is_zero(index)) {
      auto new_tile = world.taskq.add(
          make_tile, split_array.trange().make_tile_range(index),
          fused_array.find(fused_array_index));
      split_array.set(index, new_tile);
    }
  }

  return split_array;
}

}  // namespace TiledArray

#endif  // TILEDARRAY_CONVERSIONS_VECTOR_OF_ARRAYS_H_
