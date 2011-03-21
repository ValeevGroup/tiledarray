#ifndef TILEDARRAY_TEST_MADNESS_FIXTURE_H__INCLUDED
#define TILEDARRAY_TEST_MADNESS_FIXTURE_H__INCLUDED

#include "TiledArray/coordinate_system.h"
#include <boost/array.hpp>

namespace madness {
  class World;
} // namespace madness

#ifndef TEST_DIM
#define TEST_DIM 3U
#endif
#if TEST_DIM > 20
#error "TEST_DIM cannot be greater than 20"
#endif
#if !defined(TEST_C_STYLE_CS) && !defined(TEST_FORTRAN_CS)
#define TEST_C_STYLE_CS
#endif

struct GlobalFixture {
  GlobalFixture();
  ~GlobalFixture();

  typedef TiledArray::CoordinateSystem<TEST_DIM, 1U, TiledArray::detail::decreasing_dimension_order, std::size_t> c_coordinate_system;
  typedef TiledArray::CoordinateSystem<TEST_DIM, 1U, TiledArray::detail::increasing_dimension_order, std::size_t> fortran_coordinate_system;
  typedef TiledArray::CoordinateSystem<TEST_DIM, 0U, TiledArray::detail::decreasing_dimension_order, std::size_t> c_element_coordinate_system;
  typedef TiledArray::CoordinateSystem<TEST_DIM, 0U, TiledArray::detail::increasing_dimension_order, std::size_t> fortran_element_coordinate_system;

#ifdef TEST_C_STYLE_CS
  typedef c_coordinate_system coordinate_system;
  typedef c_element_coordinate_system element_coordinate_system;
#else
  typedef fortran_coordinate_system coordinate_system;
  typedef c_element_coordinate_system element_coordinate_system;
#endif

  static madness::World* world;
  static unsigned int count;
  static const std::array<std::size_t, 20> primes;
};

#endif // TILEDARRAY_TEST_MADNESS_FIXTURE_H__INCLUDED
