#include <iostream>
#include <coordinatestest.h>
#include <coordinates.h>
#include <permutation.h>

using namespace TiledArray;

struct ElementTag{};
struct TileTag{};  

void CoordinatesTest() {
  std::cout << "Coordinates Tests:" << std::endl;

  typedef ArrayCoordinate<int, 1, ElementTag> Point1;
  typedef ArrayCoordinate<int, 2, ElementTag> Point2;
  typedef ArrayCoordinate<int, 3, ElementTag> Point3;
  typedef ArrayCoordinate<int, 4, ElementTag> Point4;

  // Default constuctor
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
  int values1[] = { 1, 2, 3, 4 };
  Point4 p4D(values1);
  std::cout << "4D Point with values specified for each element: " << p4D
      << std::endl;
  
  // Element accessor
  p4D[0] = 5;
  p4D[1] = 6;
  p4D[2] = 7;
  p4D[3] = 8;
  std::cout << "Point element accessor, set 4D Point to (5, 6, 7, 8): "
      << p4D << std::endl;
  
  // Point iterator test
  std::cout << "Iteration tests: ";
  for (Point4::iterator it = p4D.begin(); it != p4D.end(); ++it) {
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

  // Comparison tests
  int values2[4] = {2, 3, 4, 5};
  int values3[4] = {4, 3, 2, 1};
  int values4[4] = {1, 2, 3, 5};
  int values5[4] = {1, 2, 3, 3};
  Point4 comp1(values1);
  Point4 comp2(values2);
  Point4 comp3(values3);
  Point4 comp4(values4);
  Point4 comp5(values5);

  std::cout << "Comparision Tests:" << std::endl;
  std::cout << "(1,2,3,4) < (2,3,4,5) = " << (comp1 < comp2) << std::endl;
  std::cout << "(1,2,3,4) > (2,3,4,5) = " << (comp1> comp2) << std::endl;
  std::cout << "(1,2,3,4) <= (2,3,4,5) = " << (comp1 <= comp2) << std::endl;
  std::cout << "(1,2,3,4) >= (2,3,4,5) = " << (comp1 >= comp2) << std::endl;
  std::cout << "(1,2,3,4) == (2,3,4,5) = " << (comp1 == comp2) << std::endl;
  std::cout << "(1,2,3,4) != (2,3,4,5) = " << (comp1 != comp2) << std::endl;

  std::cout << "(1,2,3,4) < (4,3,2,1) = " << (comp1 < comp3) << std::endl;
  std::cout << "(1,2,3,4) > (4,3,2,1) = " << (comp1> comp3) << std::endl;
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

  std::cout << "End Point Test" << std::endl << std::endl;
  
  // permutation
  Permutation<3> perm3 = Permutation<3>::unit();
  std::cout << "Unit Permutation: " << perm3 << std::endl;
  Permutation<4>::Index _perm4[] = {0,2,1,3};
  Permutation<4> perm4(_perm4);
  std::cout << "Permutation: " << perm4 << std::endl;
  std::cout << "Applying Permutation " << perm4 << " to Point " << comp3 << " = " << (perm4^comp3) << std::endl;
  {
    Permutation<4>::Index _p1[] = {2,1,0,3};
    Permutation<4> p1(_p1);
    Permutation<4>::Index _p2[] = {1,3,2,0};
    Permutation<4> p2(_p2);
    std::cout << "Product of " << p1 << " and " << p2 << " = " << (p1^p2) << std::endl;
  }
}
