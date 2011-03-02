#ifndef TILEDARRAY_SHAPE_FIXTURES_H__INCLUDED
#define TILEDARRAY_SHAPE_FIXTURES_H__INCLUDED

#include "TiledArray/range.h"
#include "range_fixture.h"
#include "TiledArray/sparse_shape.h"
#include "TiledArray/dense_shape.h"
#include "TiledArray/pred_shape.h"

struct DenseShapeFixture : public RangeFixture {
  typedef TiledArray::DenseShape<GlobalFixture::coordinate_system, std::size_t> DenseShapeT;

  DenseShapeFixture() : ds(r)
  { }

  DenseShapeT ds;
};


struct SparseShapeFixture : public RangeFixture {
  typedef TiledArray::SparseShape<GlobalFixture::coordinate_system, std::size_t> SparseShapeT;
  typedef madness::WorldDCDefaultPmap<std::size_t> PMapT;

  SparseShapeFixture() :
      world(* GlobalFixture::world),
      pmap(new SparseShapeFixture::PMapT(* GlobalFixture::world)),
      list(SparseShapeFixture::make_list(r.volume())),
      ss(world, r, pmap, list.begin(), list.end())
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

  madness::World& world;
  std::shared_ptr<PMapT> pmap;
  std::vector<std::size_t> list;
  SparseShapeT ss;
};

struct PredShapeFixture : public RangeFixture {
  struct Even {
    typedef GlobalFixture::coordinate_system::size_array size_array;

    Even(const size_array& w) : weight(w) { }

    bool operator()(std::size_t i) const {
      return (i % 2) == 0;
    }

    template<typename InIter>
    bool operator()(InIter first, InIter last) const {
      return operator()(std::inner_product(first, last, weight.begin(), std::size_t(0)));
    }

    size_array weight;
  }; // struct Even

  typedef TiledArray::PredShape<GlobalFixture::coordinate_system, std::size_t, Even> PredShapeT;

  PredShapeFixture() : p(r.weight()), ps(r, p)
  { }

  Even p;
  PredShapeT ps;
};

struct ShapeFixture : public PredShapeFixture, public SparseShapeFixture, public DenseShapeFixture {
  typedef TiledArray::Shape<GlobalFixture::coordinate_system, std::size_t>* ShapePtr;

};

#endif // TILEDARRAY_SHAPE_FIXTURES_H__INCLUDED
