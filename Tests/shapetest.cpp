#include <shape.h>
#include <range.h>
#include <range1.h>
#include <predicate.h>
#include <iostream>
#include <boost/make_shared.hpp>
#include "shapetest.h"

using namespace TiledArray;

// Forward declaration of TiledArray Permutation.
template <unsigned int DIM>
class Permutation;

void ShapeTest() {

  typedef Range<std::size_t, 4>::tile_index_type::index eindex;
  typedef Range<std::size_t, 4>::index_type::index tindex;

  // Create a range for use with ShapeIterator.

  // Test with C-style Range Array constructor.
  eindex dim0[] = {0, 2, 4, 6};
  eindex dim1[] = {0, 2, 4, 6};
  eindex dim2[] = {0, 2, 4, 6};
  tindex tiles[3] = {4, 4, 4};

  Range1 rng_set[3] = {Range1(dim0, dim0 + tiles[0]),
                       Range1(dim1, dim1 + tiles[1]),
                       Range1(dim2, dim2 + tiles[2]) };

  boost::shared_ptr<Range<std::size_t, 3> > rng = boost::make_shared<Range<std::size_t, 3> >(& rng_set[0], & rng_set[0] + 3);

  std::cout << "Start ShapeIterator tests: " << std::endl;

  typedef Shape<std::size_t, 3> Shape3;
  typedef DensePred<3> DPred;
  typedef PredShape<std::size_t, 3, DPred> DenseShape3;
  typedef PredShape<std::size_t, 3, LowerTrianglePred<3> > LowerTriShape3;
  Shape3* shp1 = new DenseShape3(rng);
  Shape3* shp2 = new LowerTriShape3(rng);
  DenseShape3 dshp(rng);
  LowerTriShape3 tshp(rng);

  std::cout << "Dense Predicate Iterator" << std::endl << "iterate over tiles:" << std::endl;

  DenseShape3::const_iterator tile_it = dshp.begin();
  for(; !(tile_it == dshp.end()); ++tile_it)
    std::cout << *tile_it << std::endl;

  std::cout << "LowerTriange Predicate Iterator" << std::endl << "iterator over tiles" << std::endl;

  LowerTriShape3::const_iterator tri_it = tshp.begin();
  for(; tri_it != tshp.end(); ++tri_it)
    std::cout << *tri_it << std::endl;

  std::cout << "Dense Abstract Predicate Iterator" << std::endl << "iterate over tiles:" << std::endl;
  for(Shape3::const_iterator it = shp1->begin(); it != shp1->end(); ++it)
    std::cout << *it << std::endl;

  std::cout << "LowerTriangle Abstract Predicate Iterator" << std::endl << "iterate over tiles:" << std::endl;
  for(Shape3::const_iterator it = shp2->begin(); it != shp2->end(); ++it)
    std::cout << *it << std::endl;

  Shape3::const_iterator::value_type x;

}
