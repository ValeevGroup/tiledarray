#include <shapetest.h>
#include <shape.h>

using namespace TiledArray;

class DummyPredicate {
  public:
  DummyPredicate() {}
  bool includes(const Range<4>::tile_index& t) const { return true; }
};

void ShapeTest() {

  // Construct a Range
  typedef Range<4>::element_index::index eindex;
  typedef Range<4>::tile_index::index tindex;
  eindex dim0[] = {0,10,20,30};
  eindex dim1[] = {0,5,10,15,20};
  eindex dim2[] = {0,3,6,9,12,15};
  eindex dim3[] = {0,2,4,6,8,10,12};
  tindex tiles[4] = {3, 4, 5, 6};
  Range1 rng_set[4] = {Range1(dim0, tiles[0]),
                       Range1(dim1, tiles[1]),
                       Range1(dim2, tiles[2]),
                       Range1(dim3, tiles[3])};
  std::vector<Range1> rng_vector(rng_set, rng_set + 4);
  Range<4> rng0(rng_vector.begin(),rng_vector.end());

  std::cout << "ShapeIterator Tests:" << std::endl;

  typedef Range<4>::tile_iterator RangeIterator;

  PredShapeIterator<RangeIterator,DummyPredicate> shp0(rng0.begin_tile());

  std::cout << "*shp0 = " << *shp0 << std::endl;
  ++shp0;
  std::cout << "*(++shp0) = " << *shp0 << std::endl;
  // TODO don't know yet how to finish iteration

}
