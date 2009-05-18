#include <iostream>
#include <boost/array.hpp>

#include <permutation.h>
#include <coordinates.h>

#include "permutationtest.h"

using namespace TiledArray;

template <typename T, unsigned long int D>
std::ostream& operator <<(std::ostream& out, boost::array<T,D>& a) {
  out << "{{ ";
  for(unsigned long int i = 0; i < D - 1; ++i) {
    out << a[i] << ", ";
  }
  out << a[D - 1] << " }}";

  return out;
}

void PermutationTest() {
  std::cout << "Start Permutation Tests:" << std::endl;

  // Test constructors, all of them should be equal.
  Permutation<3> unit(0,1,2);
  Permutation<3> p0;
  Permutation<3> p1(0,2,1);

  Permutation<3>::Array a2 = { {0, 2, 1} };
  Permutation<3> p2(a2);

  typedef ArrayCoordinate<size_t, 3, LevelTag<0> > Index3;

  std::cout << "unit = " << unit << std::endl << "p0 = " << p0 << std::endl;
  std::cout << "p1 = " << p1 << std::endl << "p2 = " << p2 << std::endl;

  std::cout << "Comparison: "
      << (p0 == Permutation<3>::unit() && !(p0 == p1) ? "Pass" : "Fail") << std::endl;
  std::cout << "Default constructor: " << (p0 == unit ? "Pass" : "Fail") << std::endl;
  std::cout << "Constructors: " << (p1 == p2 ? "Pass" : "Fail") << std::endl;

  boost::array<int, 3> atest = {{4,5,6}};
  boost::array<int, 3> aresult;
  boost::array<int, 3> aperm = {{4,6,5}};
  aresult = p1 ^ atest;
  atest ^= p1;
  std::cout << "Permute Array: " << (atest == aperm && atest == aperm ? "Pass" : "Fail") << std::endl;

  Index3 ctest(4,5,6);
  Index3 cresult = p1 ^ ctest;
  std::cout << "Permute ArrayCoordinate: "
      << (cresult == Index3(4,6,5) ? "Pass" : "Fail") << std::endl;

  std::cout << "Reverse permutation: "
      << ((p1 ^ -p1) == Permutation<3>::unit() && (-p1 ^ cresult) == ctest ? "Pass" : "Fail") << std::endl;


  std::cout << "End Permutation Tests" << std::endl;
}
