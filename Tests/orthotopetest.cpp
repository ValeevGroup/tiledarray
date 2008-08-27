#include "tiledarray.h"
#include "orthotopetest.h"

using namespace TiledArray;

void
OrthotopeTest()
{
	std::cout << "Orthotope Tests:" << std::endl;

	std::cout << "Constructor tests: ";
	// Test default constructor.
	Orthotope<4> ortho1;

	// Test with C-style Range Array constructor.
	Orthotope<4>::index_t dim0[] = {0,10,20,30};
	Orthotope<4>::index_t dim1[] = {0,5,10,15,20};
	Orthotope<4>::index_t dim2[] = {0,3,6,9,12,15};
	Orthotope<4>::index_t dim3[] = {0,2,4,6,8,10,12};
	Orthotope<4>::index_t tiles[4] = {3, 4, 5, 6};

	Range rng_set[4] = {Range(dim0, tiles[0]),
						Range(dim1, tiles[1]),
						Range(dim2, tiles[2]),
						Range(dim3, tiles[3])};

	std::vector<Range> rng_vector(rng_set, rng_set + 4);

	const size_t* dim_set[] = {dim0,dim1,dim2,dim3};

	Orthotope<4> ortho2(rng_set);
	Orthotope<4> ortho3(rng_vector);
	Orthotope<4> ortho4(dim_set, tiles);

	if(ortho2 == ortho3 && ortho3 == ortho4 && ortho1.count() == 1)
		std::cout << "PASSED" << std::endl;

	std::cout << "Tile accessor tests:" << std::endl;
	Tuple<4>::value_t tile_data[] = {0, 2, 4, 1};
	Tuple<4> tile(tile_data);
	std::cout << "ortho2 = " << ortho2 << std::endl;
	std::cout << "Tile data for tile = " << tile << std::endl
		<< " low(tile)= " << ortho2.low(tile)
		<< " high(tile)= " << ortho2.high(tile)
		<< " size(tile)= " << ortho2.size(tile)
		<< " range(tile)= [ " << ortho2.range(tile).first << "," << ortho2.range(tile).second << " ]"
		<< " count(tile)= " << ortho2.nelements(tile)
		<< std::endl;
	
	Tuple<4>::value_t element_data[] = {0, 5, 7, 7};
	Tuple<4> element(element_data);
	std::cout << "element = " << element << std::endl;
	std::cout << "ortho2.tile(element) = " << ortho2.tile(element) << std::endl;
	
  std::cout << "Permutation Test:" << std::endl;

  std::cout << "End Orthotope Test" << std::endl << std::endl;
}