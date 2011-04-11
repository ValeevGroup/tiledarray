#ifndef TILEDARRAY_SHAPE_FIXTURES_H__INCLUDED
#define TILEDARRAY_SHAPE_FIXTURES_H__INCLUDED

#include "TiledArray/range.h"
#include "range_fixture.h"
#include "versioned_pmap_fixture.h"
#include "TiledArray/sparse_shape.h"
#include "TiledArray/dense_shape.h"
#include "TiledArray/pred_shape.h"

struct BaseShapeFixture {
  typedef GlobalFixture::coordinate_system::index index;
  typedef GlobalFixture::coordinate_system::ordinal_index ordinal_index;
  typedef GlobalFixture::coordinate_system::key_type key_type;
  typedef TiledArray::Range<GlobalFixture::coordinate_system> RangeN;
  typedef TiledArray::detail::VersionedPmap<GlobalFixture::coordinate_system::key_type> PmapT;
  typedef TiledArray::Shape<GlobalFixture::coordinate_system> ShapeT;


  // Common data for shape tests.
  BaseShapeFixture() :
      r(index(0), index(5)), m(GlobalFixture::world->size())
  {}

  const RangeN r;
  const PmapT m;
};

struct DenseShapeFixture : public virtual BaseShapeFixture {
  typedef TiledArray::DenseShape<GlobalFixture::coordinate_system> DenseShapeT;

  DenseShapeFixture() :
    ds(r, m)
  { }

  DenseShapeT ds;
};


struct SparseShapeFixture : public virtual BaseShapeFixture {
  typedef TiledArray::SparseShape<GlobalFixture::coordinate_system> SparseShapeT;

  SparseShapeFixture() :
      list(SparseShapeFixture::make_list(r.volume())),
      ss(* GlobalFixture::world, r, m, list.begin(), list.end())
  { }

  static std::vector<std::size_t> make_list(const std::size_t size) {
    std::vector<std::size_t> result;
    result.reserve((size / 3) + 1);
    for(std::size_t i = 0; i < size; ++i) {
      if(i % 3 == 0)
        result.push_back(i);
    }

    return result;
  }

  std::vector<std::size_t> list;
  SparseShapeT ss;
};

struct PredShapeFixture : public virtual BaseShapeFixture {
  struct Even {
    typedef GlobalFixture::coordinate_system::size_array size_array;

    Even(const size_array& w) : weight(w) { }

    bool operator()(const key_type& k) const {
      ordinal_index i = (k.keys() & 1 ? k.key1() : GlobalFixture::coordinate_system::calc_ordinal(k.key2(), weight));
      return (i % 2) == 0;
    }

    template<typename InIter>
    bool operator()(InIter first, InIter last) const {
      return operator()(std::inner_product(first, last, weight.begin(), std::size_t(0)));
    }

    size_array weight;
  }; // struct Even

  typedef TiledArray::PredShape<GlobalFixture::coordinate_system, Even> PredShapeT;

  PredShapeFixture() :
      p(r.weight()),
      ps(r, m, p)
  { }

  Even p;
  PredShapeT ps;
};

struct ShapeFixture : public DenseShapeFixture, public SparseShapeFixture, public PredShapeFixture {
  typedef ShapeT* ShapePtr;

  ShapeFixture() { }

};

#endif // TILEDARRAY_SHAPE_FIXTURES_H__INCLUDED
