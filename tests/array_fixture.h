#ifndef TILEDARRAY_TEST_ARRAY_FIXTURE_H__INCLUDED
#define TILEDARRAY_TEST_ARRAY_FIXTURE_H__INCLUDED

#include "TiledArray/array.h"
#include "TiledArray/annotated_array.h"
#include "range_fixture.h"
#include <vector>
#include "unit_test_config.h"

struct ArrayFixture : public TiledRangeFixture {
  typedef Array<int, GlobalFixture::coordinate_system> ArrayN;
  typedef ArrayN::index index;
  typedef ArrayN::ordinal_index ordinal_index;
  typedef ArrayN::value_type tile_type;

  ArrayFixture();

  ~ArrayFixture();


  std::vector<std::size_t> list;
  madness::World& world;
  ArrayN a;
}; // struct ArrayFixture


struct AnnotatedArrayFixture : public ArrayFixture {
  typedef expressions::AnnotatedArray<ArrayN> array_annotation;
  typedef expressions::AnnotatedArray<const ArrayN> const_array_annotation;

  AnnotatedArrayFixture() : vars(make_var_list()), aa(a, vars), perm() {
    std::array<std::size_t, GlobalFixture::coordinate_system::dim> p;
    p[0] = GlobalFixture::coordinate_system::dim - 1;
    for(std::size_t i = 1; i < GlobalFixture::coordinate_system::dim; ++i)
      p[i] = i - 1;
    perm = Permutation(p.begin(), p.end());
  }


  static std::string make_var_list(std::size_t first = 0,
      std::size_t last = GlobalFixture::element_coordinate_system::dim);

  expressions::VariableList vars;
  array_annotation aa;
  Permutation perm;
}; // struct AnnotatedArrayFixture

#endif // TILEDARRAY_TEST_ARRAY_FIXTURE_H__INCLUDED
