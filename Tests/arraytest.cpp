#include "array.h"
#include "shape.h"
#include "tiled_range1.h"
#include "predicate.h"
#include "coordinate_system.h"
#include <boost/smart_ptr.hpp>
#include <boost/make_shared.hpp>
#include <boost/iterator/filter_iterator.hpp>
#include <world/world.h>
#include <iostream>

//using namespace madness;
using namespace TiledArray;


void ArrayTest(madness::World& world) {

  std::cout << "Array Tests:" << std::endl;

  typedef TiledRange<std::size_t, 3> TRange3;
  typedef TiledRange<std::size_t, 4> TRange4;
  typedef TRange4::tile_index_type::index eindex;
  typedef TRange4::index_type::index tindex;
  typedef Shape<std::size_t, 3> Shape3;
  typedef DensePred<3> DPred;
  typedef PredShape<std::size_t, 3,DPred> DenseShape3;

  // Create a Range for use with Array.

  // Test with C-style Range Array constructor.
  eindex dim0[] = {0, 2, 4, 6};
  eindex dim1[] = {0, 2, 4, 6};
  eindex dim2[] = {0, 2, 4, 6};
  tindex tiles[3] = {3, 3, 3};

  TRange3::tiled_range1_type rng_set[3] = {
      TRange3::tiled_range1_type(dim0, dim0 + tiles[0]),
      TRange3::tiled_range1_type(dim1, dim1 + tiles[1]),
      TRange3::tiled_range1_type(dim2, dim2 + tiles[2]) };

  boost::shared_ptr<TRange3> rng = boost::make_shared<TRange3>(& rng_set[0], & rng_set[0] + 3);
  boost::shared_ptr<DenseShape3> shp = boost::make_shared<DenseShape3>(rng);


  typedef Array<double, 3> LArray3;
  LArray3 a1(world, shp);
  LArray3 a2(world, shp);
  a1.assign(1.0);
  a2.assign(2.0);
  std::cout << a1.begin()->second << std::endl;

  // make an initialized Future<Tile>
  typedef LArray3::tile Tile;
  madness::Future<Tile> tfut0( a1.begin()->second );
  assert(tfut0.probe() == true);  // OK since the Future is assigned, hence result is available immediately
  std::cout << tfut0.get() << std::endl;   // Future::get() returns the value

  std::cout << "End Array Tests" << std::endl << std::endl;
}
