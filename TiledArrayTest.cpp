//============================================================================
// Name        : TiledArrayTest.cpp
// Author      : Justus Calvin
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C, Ansi-style
//============================================================================

#include <stdio.h>
#include <stdlib.h>

#define TEST_TUPLE
#define TEST_TRIPLET
#define TEST_SHAPE

#include "tiledarray.h"
#include "tupletest.h"
#include "triplettest.h"
#include "shapetest.h"


int main() {
	
	RUN_TUPLE_TEST
	RUN_TRIPLET_TEST
	RUN_SHAPE_TEST
	
	return 0;
}
