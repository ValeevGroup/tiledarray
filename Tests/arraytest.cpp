#include <iostream>
#include <range.h>
#include <shape.h>
#include <boost/smart_ptr.hpp>
#include <boost/make_shared.hpp>
#include <array.h>
#include <local_array.h>
#include <distributed_array.h>

#include "arraytest.h"

using namespace TiledArray;

void ArrayTest(DistributedWorld& world) {

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

  boost::shared_ptr<Range<3> > rng = boost::make_shared< Range<3> >(rng_set);
  boost::shared_ptr<DenseShape3> shp = boost::make_shared<DenseShape3 >(rng);


  typedef LocalArray<double, 3> LArray3;
  boost::shared_ptr<LArray3> a1 = boost::make_shared<LArray3>(shp);
  a1->assign(1.0);
  std::cout << *(a1->begin()) << std::endl;

  // make an initialized Future<Tile>
  typedef LArray3::tile Tile;
  Future<Tile> tfut0( *(a1->begin()) );
  assert(tfut0.probe() == true);  // OK since the Future is assigned, hence result is available immediately
  std::cout << tfut0.get() << std::endl;   // Future::get() returns the value

  typedef DistributedArray<double, 3> DArray3;
  boost::shared_ptr<DArray3> a2(new DArray3(world,shp));
  a2->assign(2.0);
  std::cout << *(a2->begin()) << std::endl;

  std::cout << "End Array Tests" << std::endl << std::endl;
}
