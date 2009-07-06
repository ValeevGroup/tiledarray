#include <tile.h>
#include <array_storage.h>
#include <coordinates.h>
#include <permutation.h>
#include <iostream>
#include <math.h>
#include <utility>
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

using namespace TiledArray;

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

struct TileFixture {
  typedef Tile<double, 3> Tile3;
  typedef Tile3::size_array size_array;
  TileFixture() {
    s[0] = s[1] = s[2] = 10;
    t.resize(s, 1);

  }

  ~TileFixture() { }

  Tile3 t;
  size_array s;
};

Tile<double, 3> add(Tile<double, 3>& t, int s) {
  Tile<double, 3> result(t);
  for(Tile<double, 3>::iterator it = t.begin(); it != t.end(); ++it)
    *it += s;

  return result;
}

BOOST_FIXTURE_TEST_SUITE( tile_suite , TileFixture )

BOOST_AUTO_TEST_CASE( math_time_1 )
{
  Tile3 t1(s, 0);
  int s = 1;
  for(unsigned int i = 0; i < 1000; ++i) {
    t1 = add(t, s);
  }
}

BOOST_AUTO_TEST_CASE( math_time_2 )
{
  Tile3 t1(s, 0);
  int s = 1;
  for(unsigned int i = 0; i < 1000; ++i) {
    t1 = t + s;
  }
}


BOOST_AUTO_TEST_SUITE_END()


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
