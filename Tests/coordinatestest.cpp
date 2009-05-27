#include <coordinates.h>
#include <permutation.h>
#include <iostream>
#include "coordinatestest.h"

using namespace TiledArray;

struct ElementTag{};
struct TileTag{};

void CoordinatesTest() {
  std::cout << "Coordinates Tests:" << std::endl;

  typedef ArrayCoordinate<int, 1, ElementTag> Point1;
  typedef ArrayCoordinate<int, 2, ElementTag> Point2;
  typedef ArrayCoordinate<int, 3, ElementTag> Point3;
  typedef ArrayCoordinate<int, 4, ElementTag> Point4;
  typedef ArrayCoordinate<int, 4, ElementTag, CoordinateSystem<4,TiledArray::detail::increasing_dimension_order> > FPoint4;

  // Default constructor
  Point1 p1D;
  std::cout << "1D Point with default constructor: " << p1D << std::endl;

  // Constructor with specified value for all elements
  Point2 p2D(1);
  std::cout << "2D Point with single value specified: " << p2D << std::endl;

  // Copy constructor test
  Point3 p3D(2);
  Point3 p3Dcopy(p3D);
  std::cout << "3D Point created with copy constructor: Original: " << p3D
      << " Copy: " << p3Dcopy << std::endl;

  // Multiple values specified
  int values4[4] = {1, 2, 3, 5};
  Point4 comp1(1,2,3,4);
  Point4 comp2 = Point4::make(2, 3, 4, 5);
  Point4 comp3 = make_coord<Point4>(2, 3, 4, 5);
  Point4 comp4(values4, values4 + 4);
  Point4 comp5(comp1);
  std::cout << "comp1(1,2,3,4) = " << comp1 << std::endl;
  std::cout << "comp2 = Point4::make(2, 3, 4, 5): " << comp2 << std::endl;
  std::cout << "comp3 = make_coord<Point4>(2, 3, 4, 5): " << comp2 << std::endl;
  std::cout << "comp4(values4, values4 + 4): " << comp4 << std::endl;
  std::cout << "comp5(comp1): " << comp5 << std::endl;

  // Element accessor
  comp5[3] = 3;
  std::cout << "Point element accessor, comp5[3] = 3: "
      << comp5 << std::endl;

  // Point iterator test
  std::cout << "Iteration tests comp1: ";
  for (Point4::iterator it = comp1.begin(); it != comp1.end(); ++it) {
    std::cout << *it << ", ";
  }
  std::cout << std::endl;

  // Arithmetic tests
  std::cout << "Arithmetic Tests:" << std::endl;
  std::cout << "(2,2,2) + (1,1,1) = " << Point3(2) + Point3(1) << std::endl;
  std::cout << "(3,3,3) - (1,1,1) = " << Point3(3) - Point3(1) << std::endl;
  std::cout << "(1,1,1) - (3,3,3) = " << Point3(1) - Point3(3) << std::endl;
  std::cout << "-(4,4,4) = " << -(Point3(4)) << std::endl;
  std::cout << "(2,2,2) += (1,1,1) = " << (Point3(2) += Point3(1)) << std::endl;
  std::cout << "(3,3,3) -= (1,1,1) = " << (Point3(3) -= Point3(1)) << std::endl;

  std::cout << "Comparision Tests:" << std::endl;
  std::cout << "(1,2,3,4) < (2,3,4,5) = " << (comp1 < comp2) << std::endl;
  std::cout << "(1,2,3,4) > (2,3,4,5) = " << (comp1 > comp2) << std::endl;
  std::cout << "(1,2,3,4) <= (2,3,4,5) = " << (comp1 <= comp2) << std::endl;
  std::cout << "(1,2,3,4) >= (2,3,4,5) = " << (comp1 >= comp2) << std::endl;
  std::cout << "(1,2,3,4) == (2,3,4,5) = " << (comp1 == comp2) << std::endl;
  std::cout << "(1,2,3,4) != (2,3,4,5) = " << (comp1 != comp2) << std::endl;

  std::cout << "(1,2,3,4) < (4,3,2,1) = " << (comp1 < comp3) << std::endl;
  std::cout << "(1,2,3,4) > (4,3,2,1) = " << (comp1 > comp3) << std::endl;
  std::cout << "(1,2,3,4) <= (4,3,2,1) = " << (comp1 <= comp3) << std::endl;
  std::cout << "(1,2,3,4) >= (4,3,2,1) = " << (comp1 >= comp3) << std::endl;
  std::cout << "(1,2,3,4) == (4,3,2,1) = " << (comp1 == comp3) << std::endl;
  std::cout << "(1,2,3,4) != (4,3,2,1) = " << (comp1 != comp3) << std::endl;

  std::cout << "(1,2,3,4) == (1,2,3,4) = " << (comp1 == comp1) << std::endl;
  std::cout << "(1,2,3,4) == (1,2,3,5) = " << (comp1 == comp4) << std::endl;
  std::cout << "(1,2,3,4) == (1,2,3,3) = " << (comp1 == comp5) << std::endl;

  std::cout << "Iteration Tests:" << std::endl;
  std::cout << "++" << comp1 << " = "; std::cout << ++comp1 << std::endl;
  std::cout << "--" << comp2 << " = "; std::cout << --comp2 << std::endl;

  std::cout << "Testing Fortran-style Point:" << std::endl;
  FPoint4 fpt1(1,2,3,4);
  FPoint4 fpt2(2,3,4,5);
  std::cout << "++" << fpt1 << " = "; std::cout << ++fpt1 << std::endl;
  std::cout << "--" << fpt2 << " = "; std::cout << --fpt2 << std::endl;

//  std::cout << "volume(" << comp2 << ") = " << volume(comp2) << std::endl;

  // permutation
  Permutation<3> perm3 = Permutation<3>::unit();
  std::cout << "Unit Permutation: " << perm3 << std::endl;
  Permutation<4>::Index _perm4[] = {0,2,1,3};
  Permutation<4> perm4(_perm4, _perm4 + 4);
  std::cout << "Permutation: " << perm4 << std::endl;
  std::cout << "Applying Permutation " << perm4 << " to Point " << comp3 << " = " << (perm4^comp3) << std::endl;
  {
    Permutation<4> p1(2,1,0,3);
    Permutation<4> p2(1,3,2,0);
    std::cout << "Product of " << p1 << " and " << p2 << " = " << (p1^p2) << std::endl;
  }

  std::cout << "End Point Test" << std::endl << std::endl;
}
