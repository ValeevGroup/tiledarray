#include <shapetest.h>
#include <shape.h>
#include <predicate.h>

using namespace TiledArray;

void ShapeTest() {

  std::cout << "Shape Tests:" << std::endl;

  // Default Constructor, not allowed.
  // Shape<3, OffTupleFilter<3> > shapeDefault;

  // Create an orthotope that will be used for shape tests.
  // Test with C-style Range Array constructor.
  Orthotope<3>::index_t dim0[] = {0,4,6};
  Orthotope<3>::index_t dim1[] = {0,4,6,9};
  Orthotope<3>::index_t dim2[] = {0,4,6,9,10};
  Orthotope<3>::index_t tiles[3] = {2,3,4};
  const size_t* dim_set[] = {dim0,dim1,dim2};
  Orthotope<3> ortho3(dim_set, tiles);

  std::cout << "Orthotope used in shape tests." << std::endl;
  std::cout << "ortho3 = " << ortho3 << std::endl;
  typedef Shape<3, OffTupleFilter<3> > shape_t;
  shape_t shp0(&ortho3);
  std::cout << "Constructed shp0" << std::endl << "shp0 = " << shp0 << std::endl;
  
  typedef shape_t::iterator shapeiter;
  shapeiter i0 = shp0.begin();
  
  const int pm0[] = {0,2,1};
  Tuple<3> perm0(pm0);
  shp0.permute(perm0);
  std::cout << "Permuted shp0 with " << perm0 << std::endl;
  std::cout << "shp0 = " << shp0 << std::endl;
  
  
}