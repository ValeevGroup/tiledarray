#ifndef TILEDARRAY_TEST_MATH_FIXTURE_H__INCLUDED
#define TILEDARRAY_TEST_MATH_FIXTURE_H__INCLUDED

#include <cstddef>
#include "unit_test_config.h"
#include "TiledArray/range.h"
#include "TiledArray/tile.h"
#include "TiledArray/annotated_array.h"

struct MathFixture {
  typedef TiledArray::Tile<int, GlobalFixture::coordinate_system> array_type;
  typedef TiledArray::CoordinateSystem<2,
        GlobalFixture::coordinate_system::level,
        GlobalFixture::coordinate_system::order,
        GlobalFixture::coordinate_system::ordinal_index> coordinate_system2;
  typedef TiledArray::Tile<int, coordinate_system2> array2_type;
  typedef array_type::range_type range_type;
  typedef TiledArray::expressions::AnnotatedArray<array_type > array_annotation;
  typedef array_annotation::index index;
  typedef array_annotation::ordinal_index ordinal_index;

  MathFixture()
  { }

  static std::string make_var_list(std::size_t first = 0,
      std::size_t last = GlobalFixture::element_coordinate_system::dim);

  static const TiledArray::expressions::VariableList vars;
  static const range_type r;
  static const array_type f1;
  static const array_type f2;
  static const array_type f3;

  static const array_annotation a1;
  static const array_annotation a2;
  static const array_annotation a3;
};

#endif // TILEDARRAY_TEST_MATH_FIXTURE_H__INCLUDED
