#ifndef TILEDARRAY_TEST_ARRAY_FIXTURE_H__INCLUDED
#define TILEDARRAY_TEST_ARRAY_FIXTURE_H__INCLUDED

#include "TiledArray/array.h"
#include "TiledArray/annotated_tensor.h"
#include "range_fixture.h"
#include <vector>
#include "unit_test_config.h"

struct ArrayFixture : public TiledRangeFixture {
  typedef Array<int, GlobalFixture::dim> ArrayN;
  typedef ArrayN::index index;
  typedef ArrayN::size_type size_type;
  typedef ArrayN::value_type tile_type;

  ArrayFixture();

  ~ArrayFixture();


  std::vector<std::size_t> list;
  madness::World& world;
  ArrayN a;
}; // struct ArrayFixture


struct AnnotatedTensorFixture : public ArrayFixture {
  typedef expressions::TensorExpression<expressions::Tensor<ArrayN::element_type> > array_annotation;

  AnnotatedTensorFixture() : vars(make_var_list()), aa(expressions::make_annotatied_tensor(a, vars)), perm() {
    std::array<std::size_t, GlobalFixture::dim> p;
    p[0] = GlobalFixture::dim - 1;
    for(std::size_t i = 1; i < GlobalFixture::dim; ++i)
      p[i] = i - 1;
    perm = Permutation(p.begin(), p.end());
  }


  static std::string make_var_list(std::size_t first = 0,
      std::size_t last = GlobalFixture::dim);

  expressions::VariableList vars;
  array_annotation aa;
  Permutation perm;
}; // struct AnnotatedTensorFixture

#endif // TILEDARRAY_TEST_ARRAY_FIXTURE_H__INCLUDED
