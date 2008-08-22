//============================================================================
// Name        : TiledArrayTest.cpp
// Author      : Justus Calvin
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>

//#define TEST_TUPLE
#define TEST_RANGE
//#define TEST_ORTHOTOPE
//#define TEST_SHAPE

#include "tiledarray.h"
#include "tupletest.h"
#include "rangetest.h"
#include "orthotopetest.h"
#include "shapetest.h"



int main() {
	
	RUN_TUPLE_TEST
	RUN_RANGE_TEST
	RUN_ORTHOTOPE_TEST
	RUN_SHAPE_TEST
	
	return 0;
}
