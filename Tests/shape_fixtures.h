#ifndef TILEDARRAY_SHAPE_FIXTURES_H__INCLUDED
#define TILEDARRAY_SHAPE_FIXTURES_H__INCLUDED

#include "TiledArray/range.h"
#include "range_fixture.h"
#include "versioned_pmap_fixture.h"
#include "TiledArray/sparse_shape.h"
#include "TiledArray/dense_shape.h"
#include "TiledArray/pred_shape.h"

struct BaseShapeFixture :
    public virtual RangeFixture,
    public virtual VersionedPmapFixture
{ };

struct DenseShapeFixture : public BaseShapeFixture {
  typedef TiledArray::DenseShape<GlobalFixture::coordinate_system> DenseShapeT;

  DenseShapeFixture() :
      RangeFixture(),
      VersionedPmapFixture(),
      BaseShapeFixture(),
      ds(r, m)
  { }

  DenseShapeT ds;
};


struct SparseShapeFixture : public BaseShapeFixture {
  typedef TiledArray::SparseShape<GlobalFixture::coordinate_system> SparseShapeT;

  SparseShapeFixture() :
      RangeFixture(),
      VersionedPmapFixture(),
      BaseShapeFixture(),
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

struct PredShapeFixture : public BaseShapeFixture {
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

  typedef TiledArray::PredShape<GlobalFixture::coordinate_system, Even> PredShapeT;

  PredShapeFixture() :
      RangeFixture(),
      VersionedPmapFixture(),
      BaseShapeFixture(),
      p(r.weight()),
      ps(r, m, p)
  { }

  Even p;
  PredShapeT ps;
};

struct ShapeFixture : public virtual PredShapeFixture, public virtual SparseShapeFixture, public virtual DenseShapeFixture {
  typedef TiledArray::Shape<GlobalFixture::coordinate_system>* ShapePtr;

  ShapeFixture() : RangeFixture() { }

};

#endif // TILEDARRAY_SHAPE_FIXTURES_H__INCLUDED
