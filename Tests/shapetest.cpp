#include <shapetest.h>
#include <shape.h>
#include <predicate.h>

using namespace TiledArray;

void ShapeTest() {

  std::cout << "Shape Tests:" << std::endl;

  // Create an orthotope that will be used for shape tests.
  // Test with C-style Range Array constructor.
  Orthotope<3>::index_t dim0[] = {0,4,6};
  Orthotope<3>::index_t tiles[3] = {2,2,2};
  size_t* dim_set[3] = {dim0,dim0,dim0};
  Orthotope<3> ortho4(const_cast<const size_t**>(dim_set), tiles);

  std::cout << "Orthotope used in shape tests." << std::endl;
  std::cout << "ortho3 = " << ortho4 << std::endl;
  // Default Constructor, not allowed.
  // Shape<4, OffTupleFilter<4> > shapeDefault;

  Shape<3, OffTupleFilter<3> > shp(&ortho4);
}