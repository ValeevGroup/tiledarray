#include <iostream>
#include <range.h>
#include <shape.h>
#include <boost/smart_ptr.hpp>
#include <array.h>

using namespace TiledArray;

void ArrayTest() {

  typedef Range<4>::element_index::index eindex;
  typedef Range<4>::tile_index::index tindex;
  typedef Shape< Range<3> > Shape3;
  typedef PredShape<Range<3>, DensePred<3> > DenseShape3;

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
  DenseShape3 shp(rng);

  Array<double, 3> a1(boost::shared_ptr<Shape3>(&shp));
}
