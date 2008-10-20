//============================================================================
// Name        : TiledArrayTest.cpp
// Author      : Justus Calvin
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

//#define TEST_TUPLE
#define TEST_COORDINATES
#define TEST_RANGE
//#define TEST_ORTHOTOPE
//#define TEST_SHAPE
//#define TEST_TILEMAP
//#define TEST_ARRAY

#include "tupletest.h"
#include "coordinatestest.h"
#include "rangetest.h"
#include "orthotopetest.h"
#include "shapetest.h"
#include "tilemaptest.h"
#include "arraytest.h"

namespace TiledArray { };
using namespace TiledArray;

int main(int argc, char* argv[]) {
	
	RUN_TUPLE_TEST
	RUN_COORDINATES_TEST
	RUN_RANGE_TEST
	RUN_ORTHOTOPE_TEST
	RUN_SHAPE_TEST
    RUN_TILEMAP_TEST
    RUN_ARRAY_TEST
	
	return 0;
}
