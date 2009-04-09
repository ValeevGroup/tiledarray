#include <iostream>
#include <range.h>
#include "rangetest.h"

using namespace TiledArray;

void
RangeTest()
{
  typedef Range<4>::element_index::index eindex;
  typedef Range<4>::tile_index::index tindex;

	std::cout << "Range Tests:" << std::endl;

	std::cout << "Constructor tests: ";
	// Test default constructor.
	Range<4> ortho1;

	// Test with C-style Range Array constructor.
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

	Range<4> ortho2(rng_set);
	Range<4> ortho3(rng_vector.begin(),rng_vector.end());

	std::cout << "PASSED" << std::endl;

	std::cout << "Ranges: ";
	std::cout << "ortho1 = " << ortho1 << std::endl;
    std::cout << "ortho2 = " << ortho2 << std::endl;
    std::cout << "ortho3 = " << ortho3 << std::endl;

	std::cout << "comparison tests: " << std::endl;
	std::cout << "ortho1 == ortho1: " << (ortho1==ortho1 ? "true" : "false") << std::endl;
    std::cout << "ortho1 == ortho3: " << (ortho1==ortho3 ? "true" : "false") << std::endl;
    std::cout << "ortho2 == ortho3: " << (ortho2==ortho3 ? "true" : "false") << std::endl;
    std::cout << "ortho1 != ortho1: " << (ortho1!=ortho1 ? "true" : "false") << std::endl;
    std::cout << "ortho1 != ortho3: " << (ortho1!=ortho3 ? "true" : "false") << std::endl;
    std::cout << "ortho2 != ortho3: " << (ortho2!=ortho3 ? "true" : "false") << std::endl;

    std::cout << "ordinal tests: " << std::endl;
    typedef Range<4>::element_index element_index;
    typedef Range<4>::tile_index tile_index;
    tile_index::index _t1[] = {0, 2, 4, 1};
    tile_index t1(_t1);
    element_index::index _e1[] = {1, 3, 3, 2};
    element_index e1(_e1);
    element_index::index _e2[] = {29, 19, 14, 12};
    element_index e2(_e2);
    std::cout << "tile_index: t1=" << t1 << std::endl;
    std::cout << "ortho2.includes(t1) = " << ortho2.includes(t1) << std::endl;
    std::cout << "element_index: e1=" << e1 << std::endl;
    std::cout << "ortho2.includes(e1) = " << ortho2.includes(e1) << std::endl;
    std::cout << "ortho2.find(e1) = " << *(ortho2.find(e1)) << std::endl;
    std::cout << "element_index: e2=" << e2 << std::endl;
    std::cout << "ortho2.includes(e2) = " << ortho2.includes(e2) << std::endl;
    std::cout << "ortho2.find(e2) = " << *(ortho2.find(e2)) << std::endl;

    std::cout << "tile iterator tests: " << std::endl;
    for(Range<4>::tile_iterator t=ortho2.begin(); t!=ortho2.end(); ++t) {
      std::cout << "t = " << *t << std::endl;
    }

    std::cout << "End Range Tests" << std::endl << std::endl;
}
