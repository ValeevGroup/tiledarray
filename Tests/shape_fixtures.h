#ifndef TILEDARRAY_SHAPE_FIXTURES_H__INCLUDED
#define TILEDARRAY_SHAPE_FIXTURES_H__INCLUDED

#include "TiledArray/range.h"
#include "range_fixture.h"
#include "TiledArray/sparse_shape.h"
#include "TiledArray/dense_shape.h"
#include "TiledArray/pred_shape.h"
/*
struct DenseShapeFixture : public RangeFixture {
  typedef TiledArray::DenseShape<std::size_t> DenseShapeT;

  DenseShapeFixture() : ds(r)
  { }

  DenseShapeT ds;
};


struct SparseShapeFixture : public RangeFixture {
  typedef TiledArray::SparseShape<std::size_t> SparseShapeT;



  SparseShapeFixture() : world(* GlobalFixture::world), ss(world, r)
  { }

  madness::World& world;
  SparseShapeT ss;
};

struct PredShapeFixture : public RangeFixture {
  struct Even {
    template<std::size_t N>
    Even(const boost::array<std::size_t, N>& w) : weight(w) { }

    bool operator()(std::size_t i) const {
      return (i % 2) == 0;
    }

    template<typename InIter>
    bool operator()(InIter first, InIter last) const {
      return operator()(std::inner_product(first, last, weight.begin(), std::size_t(0)));
    }

    TiledArray::detail::ArrayRef<const std::size_t> weight;
  }; // struct Even

  typedef TiledArray::PredShape<std::size_t> PredShapeT;

  PredShapeFixture() : p(r.weight()), ps(r, p)
  { }

  Even p;
  PredShapeT ps;
};

struct ShapeFixture : public PredShapeFixture, public SparseShapeFixture, public DenseShapeFixture {
  typedef TiledArray::Shape<std::size_t>* ShapePtr;

};
*/
#endif // TILEDARRAY_SHAPE_FIXTURES_H__INCLUDED
