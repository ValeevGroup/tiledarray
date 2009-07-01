#include <tile.h>
#include <array_storage.h>
#include <coordinates.h>
#include <permutation.h>
#include <iostream>
#include <math.h>
#include <utility>
#define WORLD_INSTANTIATE_STATIC_TEMPLATES
#include <world/world.h>

// Element Generation object test.
template<typename T, typename Index>
class gen {
public:
  const T operator ()(const Index& i) {
    typedef typename Index::index index_t;
	index_t result = 0;
    index_t e = 0;
    for(unsigned int d = 0; d < Index::dim(); ++d) {
      e = i[d] * static_cast<index_t>(std::pow(10.0, static_cast<int>(Index::dim()-d-1)));
      result += e;
    }

    return result;
  }
};

using namespace TiledArray;
/*
void TileTest() {

  std::cout << "Tile Tests:" << std::endl;

  typedef Tile<double, 3 > Tile3;

  Tile3 defTile;

  Tile3::size_array sizes = { { 3, 3, 3 } };
  Tile3 t(sizes, Tile3::index_type(1));

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
  t ^= p;
  std::cout << t2 << std::endl;
  std::cout << t << std::endl;

  std::cout << "End Tile Tests" << std::endl << std::endl;
}
*/
