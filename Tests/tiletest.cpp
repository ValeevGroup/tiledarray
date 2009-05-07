#include <iostream>
#include <tile.h>

using namespace TiledArray;

template<unsigned int Level>
struct LevelTag { };

void TileTest() {

  typedef Tile<double, 3, ArrayCoordinate<size_t, 3, LevelTag<0> > > Tile3;

  Tile3::size_array sizes = { { 5, 5, 5 } };
  Tile3 t(sizes);

  std::cout << t << std::endl;

}
