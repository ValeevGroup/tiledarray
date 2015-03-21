#pragma once
#ifndef TILEDARRAY_SPARSETODENSE_H__INCLUDED
#define TILEDARRAY_SPARSETODENSE_H__INCLUDED

#include <TiledArray/array.h>

namespace TiledArray {

  /// Function to convert a block sparse array into a dense array.

  template <typename T, unsigned int DIM, typename Tile>
  Array<T, DIM, Tile, DensePolicy>
  to_dense(Array<T, DIM, Tile, SparsePolicy> const& sparse_array) {
      typedef Array<T, DIM, Tile, DensePolicy> ArrayType;
      ArrayType dense_array(sparse_array.get_world(), sparse_array.trange());

      typedef typename ArrayType::pmap_interface pmap_interface;
      std::shared_ptr<pmap_interface> const& pmap = dense_array.get_pmap();

      typename pmap_interface::const_iterator end = pmap->end();

      // iteratate over sparse tiles
      for (typename pmap_interface::const_iterator it = pmap->begin(); it != end;
           ++it) {
          const std::size_t ord = *it;
          if (!sparse_array.is_zero(ord)) {
              // clone because tiles are shallow copied
              Tile tile(sparse_array.find(ord).get().clone());
              dense_array.set(ord, tile);
          } else {
              // This is how Array::set_all_local() sets tiles to a value,
              // This likely means that what ever type Tile is must be
              // constructible from a type T
              dense_array.set(ord, T(0.0));  // This is how Array::set_all_local()
          }
      }

      return dense_array;
  }

  // If array is already dense just use the copy constructor.
  template <typename T, unsigned int DIM, typename Tile>
  Array<T, DIM, Tile, DensePolicy>
  to_dense(Array<T, DIM, Tile, DensePolicy> const& other) {
      return other;
  }

}  // namespace TiledArray

#endif /* end of include guard: TILEDARRAY_SPARSETODENSE_H__INCLUDED */
