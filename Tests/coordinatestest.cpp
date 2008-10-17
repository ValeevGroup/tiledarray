#include <iostream>
#include <coordinatestest.h>
#include <coordinates.h>

using namespace TiledArray;

struct ElementTag{};
struct TileTag{};  

void CoordinatesTest() {
  std::cout << "Coordinates Tests:" << std::endl;

  typedef LatticeCoordinate<int, 1, ElementTag> Point;

  // Default constuctor
  Point p1(3);
  std::cout << "Point in 1-d space: " << p1 << std::endl;
  
}
