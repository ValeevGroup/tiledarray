#include "tiledarray.h"
#include "orthotopetest.h"

using namespace TiledArray;

void
OrthotopeTest()
{
	std::cout << "Orthotope Tests:" << std::endl << std::endl;

	std::cout << "Constructor tests: ";
	// Test default constructor.
	Orthotope<4> ortho1;

	// Test with C-style Range Array constructor.
	size_t dim0[] = {0,10,20,30};
	size_t dim1[] = {0,5,10,15,20};
	size_t dim2[] = {0,3,6,9,12,15};
	size_t dim3[] = {0,2,4,6,8,10,12};
	size_t tiles[4] = {3, 4, 5, 6};

	Range rng_set[4] = {Range(dim0, tiles[0]),
						Range(dim1, tiles[1]),
						Range(dim2, tiles[2]),
						Range(dim3, tiles[3])};

	std::vector<Range> rng_vector(rng_set, rng_set + 4);

	size_t* dim_set[4] = {dim0,dim1,dim2,dim3};

	Orthotope<4> ortho2(rng_set);
	Orthotope<4> ortho3(rng_vector);
	Orthotope<4> ortho4(const_cast<const size_t**>(dim_set), tiles);

	if(ortho2 == ortho3 && ortho3 == ortho4)
		std::cout << "PASSED" << std::endl;

	
	
}