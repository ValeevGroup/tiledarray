#include <iostream>
#include <range.h>
#include <shape.h>
#include <boost/smart_ptr.hpp>
#include <array.h>
#include <local_array.h>

using namespace TiledArray;

void ArrayTest() {

  std::cout << "Array Tests:" << std::endl;

  typedef Range<4>::element_index::index eindex;
  typedef Range<4>::tile_index::index tindex;
  typedef Shape<3> Shape3;
  typedef PredShape<3> DenseShape3;

  // Create a Range for use with Array.

  // Test with C-style Range Array constructor.
  eindex dim0[] = {0, 2, 4, 6};
  eindex dim1[] = {0, 2, 4, 6};
  eindex dim2[] = {0, 2, 4, 6};
  tindex tiles[3] = {3, 3, 3};

  Range1 rng_set[3] = {Range1(dim0, tiles[0]),
                       Range1(dim1, tiles[1]),
                       Range1(dim2, tiles[2]) };

  Range<3> rng(rng_set);
  boost::shared_ptr<DenseShape3> shp( new DenseShape3(&rng));


  typedef LocalArray<double, 3> LArray3;
  boost::shared_ptr<LArray3> a1(new LArray3(shp));
  a1->assign(1.0);

  std::cout << "End Array Tests" << std::endl << std::endl;
}
