#include <iostream>
#include <range1.h>
#include <range1test.h>

using namespace TiledArray;

void Range1Test() {

  std::cout << "Range1 Tests:" << std::endl;

  // Test Default constructor
  std::cout << "Default Constructor:" << std::endl;
  Range1 rngDefault;
  std::cout << "rngDefault = " << rngDefault << std::endl;

  // Test c-style array constructor
  std::cout << std::endl << "C-style array constructor:" << std::endl;
  size_t ranges[] = { 0, 3, 7, 10, 20, 100 };
  size_t tiles = 5;
  Range1 rng1(ranges, tiles);
  std::cout << "rng1 = " << rng1 << std::endl;

  // Test vector array constructor
  std::cout << std::endl << "Vector array constructor:" << std::endl;
  ranges[5] = 50;
  const std::vector<size_t> ranges_vector(ranges, ranges + tiles + 1);
  Range1 rng2(ranges_vector);
  std::cout << "rng2 = " << rng2 << std::endl;

  // Test copy constructor
  std::cout << std::endl << "Copy constructor:" << std::endl;
  Range1 rng3(rng2);
  std::cout << "rng3 = " << rng3 << std::endl;

  // Accessor functions
  std::cout << std::endl << "Accessor fuctions:" << std::endl;
  std::cout << "rngDefault.empty() = " << rngDefault.empty() << std::endl;
  std::cout << "rng1.empty() = " << rng1.empty() << std::endl;
  std::cout << "rng1.size() = " << rng1.size() << std::endl;
  std::cout << "rng1.size(1) = " << rng1.size(1) << std::endl;
  std::cout << "rng1.includes_tile(1) = " << rng1.includes_tile(1) << std::endl;
  std::cout << "rng1.includes_tile(5) = " << rng1.includes_tile(5) << std::endl;
  std::cout << "rng1.includes_element(1) = " << rng1.includes_element(1) << std::endl;
  std::cout << "rng1.includes_element(100) = " << rng1.includes_element(100) << std::endl;

  // Test assignment operator
  std::cout << std::endl << "Assignment operator:" << std::endl;
  rngDefault = rng3;
  std::cout << "rngDefault = rng3 = " << rngDefault << std::endl;

  // Test comparison operators
  std::cout << std::endl << "Comparison operator:" << std::endl;
  std::cout << "(rngDefault == rng3) = " << (rngDefault == rng3) << std::endl;
  std::cout << "(rngDefault != rng3) = " << (rngDefault != rng3) << std::endl;
  std::cout << "(rng1 == rng3) = " << (rng1 == rng3) << std::endl;
  std::cout << "(rng1 != rng3) = " << (rng1 != rng3) << std::endl;

  //Test iterator
  std::cout << std::endl << "Iterator:" << std::endl << "Tiling for rng3"
      << std::endl;
  for(Range1::tile_iterator it = rng3.begin_tile(); it != rng3.end_tile(); ++it) {
    std::cout << "[" << rng3.start_element(*it) << ","<< rng3.finish_element(*it) << ")" << std::endl;
  }

  // Test tile index map
  std::cout << std::endl << "Tile index map for rng2:" << std::endl;
  for(Range1::element_iterator it = rng2.begin_element(); it != rng2.end_element(); ++it) {
    std::cout << "element_index= " << *it << " tile_index= "
        << *(rng2.find(*it)) << std::endl;
  }


  std::cout << "End Range1 Tests" << std::endl << std::endl;
}
