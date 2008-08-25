//============================================================================
// Name        : TiledArrayTest.cpp
// Author      : Justus Calvin
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

//#define TEST_TUPLE
#define TEST_RANGE
#define TEST_ORTHOTOPE
//#define TEST_SHAPE
#define TEST_TILEMAP

#include "tiledarray.h"
#include "tupletest.h"
#include "rangetest.h"
#include "orthotopetest.h"
#include "shapetest.h"
#include "tilemaptest.h"

using namespace TiledArray;

int main() {
	
	RUN_TUPLE_TEST
	RUN_RANGE_TEST
	RUN_ORTHOTOPE_TEST
	RUN_SHAPE_TEST
    RUN_TILEMAP_TEST
	
	return 0;
}
