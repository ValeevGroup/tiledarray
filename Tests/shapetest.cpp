#include <shapetest.h>
#include <range1.h>
#include <range.h>
#include <shape.h>
#include <predicate.h>

using namespace TiledArray;

void ShapeTest() {

  typedef Range<4>::element_index::index eindex;
  typedef Range<4>::tile_index::index tindex;

  // Create a range for use with ShapeIterator.

  // Test with C-style Range Array constructor.
  eindex dim0[] = {0, 2, 4, 6};
  eindex dim1[] = {0, 2, 4, 6};
  eindex dim2[] = {0, 2, 4, 6};
  tindex tiles[3] = {3, 3, 3};

  Range1 rng_set[3] = {Range1(dim0, tiles[0]),
                       Range1(dim1, tiles[1]),
                       Range1(dim2, tiles[2]) };

  Range<3> rng(rng_set);

  std::cout << "Start ShapeIterator tests: " << std::endl;

  typedef Shape< Range<3> > Shape3;
  typedef PredShape<Range<3>, DensePred<3> > DenseShape3;
  typedef PredShape<Range<3>, LowerTrianglePred<3> > LowerTriShape3;
  Shape3* shp1 = new DenseShape3(rng);
  Shape3* shp2 = new LowerTriShape3(rng);
  DenseShape3 dshp(rng);
  LowerTriShape3 tshp(rng);

  std::cout << "Dense Predicate Iterator" << std::endl << "iterate over tiles:" << std::endl;

  DenseShape3::Iterator tile_it = dshp.begin();
  for(; !(tile_it == dshp.end()); ++tile_it)
    std::cout << *tile_it << std::endl;

  std::cout << "LowerTriange Predicate Iterator" << std::endl << "iterator over tiles" << std::endl;

  LowerTriShape3::Iterator tri_it = tshp.begin();
  for(; tri_it != tshp.end(); ++tri_it)
    std::cout << *tri_it << std::endl;

  std::cout << "Dense Abstract Predicate Iterator" << std::endl << "iterate over tiles:" << std::endl;
  for(Shape3::Iterator it = shp1->begin(); it != shp1->end(); ++it)
    std::cout << *it << std::endl;

  std::cout << "LowerTriangle Abstract Predicate Iterator" << std::endl << "iterate over tiles:" << std::endl;
  for(Shape3::Iterator it = shp2->begin(); it != shp2->end(); ++it)
    std::cout << *it << std::endl;

  Shape3::Iterator::value_type x;

}
