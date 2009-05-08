#include <iostream>
#include <boost/array.hpp>
#include <tile.h>

#include "tiletest.h"

template<typename T, typename Index>
class gen {
public:
  gen() : c(0) {}

  const T& operator ()(const Index& i) {
    ++c;
    return c;
  }

  T c;
};

using namespace TiledArray;

void TileTest() {

  std::cout << "Tile Tests:" << std::endl;

  typedef Tile<double, 3 > Tile3;

  Tile3::size_array sizes = { { 3, 3, 3 } };
  Tile3 t(sizes);

  std::cout << t << std::endl;

  std::fill(t.begin(), t.end(), 1.0);

  std::cout << t << std::endl;

  gen<double, Tile3::index_type> g;

  t.assign(g);

  std::cout << t << std::endl;

  Permutation<3>::Array pa = { { 2, 0, 1 } };
  Permutation<3> p(pa);

  Tile3 t2(sizes);

  t2 = p ^ t;

  std::cout << t2 << std::endl;

  std::cout << "End Tile Tests" << std::endl << std::endl;
}
