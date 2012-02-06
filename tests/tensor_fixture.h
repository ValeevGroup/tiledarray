#ifndef TILEDARRAY_TENSOR_FIXTURE_H__INCLUDED
#define TILEDARRAY_TENSOR_FIXTURE_H__INCLUDED

#include "TiledArray/permute_tensor.h"
#include "TiledArray/tensor.h"

struct PermuteTensorFixture {
  typedef TiledArray::expressions::Tensor<int, TiledArray::StaticRange<GlobalFixture::coordinate_system> > TensorN;
  typedef TensorN::range_type range_type;
  typedef TensorN::range_type::index index;
  typedef TiledArray::Permutation PermN;
  typedef TiledArray::expressions::PermuteTensor<TensorN> PermT;
  typedef PermT::value_type value_type;

  PermuteTensorFixture() : pt(t, p) { }

  // get a unique value for the given index
  static value_type get_value(const index i);

  // make a tile to be permuted
  static TensorN make_tile();

  // make permutation definition object
  static PermN make_perm();

  static const TensorN t;
  static const PermN p;

  PermT pt;
}; // struct PermuteTensorFixture

#endif // TILEDARRAY_TENSOR_FIXTURE_H__INCLUDED
