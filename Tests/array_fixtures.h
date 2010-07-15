#ifndef TILEDARRAY_ARRAY_FIXTURES_H__INCLUDED
#define TILEDARRAY_ARRAY_FIXTURES_H__INCLUDED

#include "TiledArray/types.h"
#include "TiledArray/dense_array.h"
#include "range_fixture.h"

using namespace TiledArray;

struct DenseArrayFixture : public RangeFixture {
  typedef DenseArray<int, GlobalFixture::coordinate_system> DenseArrayN;
  typedef Permutation<GlobalFixture::coordinate_system::dim> PermutationN;

  DenseArrayFixture();
  ~DenseArrayFixture() { }

  static void no_delete(RangeN*) { }

  boost::shared_ptr<RangeN> prange;
  DenseArrayN da;
};

#endif // TILEDARRAY_ARRAY_FIXTURES_H__INCLUDED
