#ifndef TILEDARRAY_CONVERSIONS_DENSETOSPARSE_H__INCLUDED
#define TILEDARRAY_CONVERSIONS_DENSETOSPARSE_H__INCLUDED

#include "../dist_array.h"

namespace TiledArray {

/// Function to convert a dense array into a block sparse array

/// If the input array is dense then create a copy by checking the norms of the
/// tiles in the dense array and then cloning the significant tiles into the
/// sparse array.
template <typename Tile, typename ResultPolicy = SparsePolicy,
          typename ArgPolicy>
std::enable_if_t<!is_dense_v<ResultPolicy> && is_dense_v<ArgPolicy>,
                 DistArray<Tile, ResultPolicy>>
to_sparse(DistArray<Tile, ArgPolicy> const &dense_array) {
  typedef DistArray<Tile, ResultPolicy> ArrayType;  // return type
  using ShapeType = typename ResultPolicy::shape_type;
  using ShapeValueType = typename ShapeType::value_type;

  // Constructing a tensor to hold the norm of each tile in the Dense Array
  TiledArray::Tensor<ShapeValueType> tile_norms(
      dense_array.trange().tiles_range(), 0.0);

  const auto end = dense_array.end();
  const auto begin = dense_array.begin();
  for (auto it = begin; it != end; ++it) {
    // write the norm of each local tile to the tensor
    norm(it->get(), tile_norms[it.index()]);
  }

  // Construct a sparse shape the constructor will handle communicating the
  // norms of the local tiles to the other nodes
  ShapeType shape(dense_array.world(), tile_norms, dense_array.trange());

  ArrayType sparse_array(dense_array.world(), dense_array.trange(), shape);

  // Loop over the local dense tiles and if that tile is in the
  // sparse_array set the sparse array tile with a clone so as not to hold
  // a pointer to the original tile.
  for (auto it = begin; it != end; ++it) {
    const auto ix = it.index();
    if (!sparse_array.is_zero(ix)) {
      sparse_array.set(ix, it->get().clone());
    }
  }

  return sparse_array;
}

/// If the array is already sparse return a copy of the array.
template <typename Tile, typename Policy>
std::enable_if_t<!is_dense_v<Policy>, DistArray<Tile, Policy>> to_sparse(
    DistArray<Tile, Policy> const &sparse_array) {
  return sparse_array;
}

}  // namespace TiledArray

#endif /* end of include guard: \
          TILEDARRAY_CONVERSIONS_DENSETOSPARSE_H__INCLUDED */
