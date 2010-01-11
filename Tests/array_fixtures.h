#ifndef TILEDARRAY_ARRAY_FIXTURES_H__INCLUDED
#define TILEDARRAY_ARRAY_FIXTURES_H__INCLUDED

#include "TiledArray/coordinate_system.h"
#include "TiledArray/coordinates.h"
#include "TiledArray/array_dim.h"
#include "TiledArray/dense_array.h"
#include "TiledArray/distributed_array.h"

using namespace TiledArray;

struct ArrayDimFixture {
  typedef detail::ArrayDim<std::size_t, 3, LevelTag<0> > ArrayDim3;
  typedef ArrayDim3::index_type index_type;
  typedef ArrayDim3::size_array size_array;

  ArrayDimFixture();
  ~ArrayDimFixture() { }

  size_array s;
  size_array w;
  std::size_t v;
  ArrayDim3 d;
};


struct DenseArrayFixture : public ArrayDimFixture {
  static const std::size_t ndim = 5;
  typedef DenseArray<int, 3> DenseArray3;
  typedef DenseArray<int, ndim> DenseArrayN;
  typedef Permutation<ndim> PermutationN;

  DenseArrayFixture();
  ~DenseArrayFixture() { }

  DenseArray3 da3;
  DenseArrayN daN;
  DenseArrayN daN_p0;
  PermutationN p0;
};

struct DistributedArrayFixture : public ArrayDimFixture {
  typedef detail::ArrayDim<std::size_t, 3, LevelTag<1> > ArrayDim3;
  typedef std::vector<double> data_type;
  typedef DistributedArray<data_type, 3> DistArray3;
  typedef DistArray3::index_type index_type;
  typedef boost::array<std::pair<DistArray3::key_type, data_type>, 24> data_array;
  typedef Range<std::size_t, 3, LevelTag<1>, CoordinateSystem<3> > range_type;

  DistributedArrayFixture();

  double sum_first(const DistArray3& a);
  std::size_t tile_count(const DistArray3& a);

  madness::World* world;
  range_type r;
  data_array data;
  DistArray3 a;
  const DistArray3& ca;
  ArrayDim3 d;
}; // struct DistributedArrayStorageFixture

#endif // TILEDARRAY_ARRAY_FIXTURES_H__INCLUDED
