#ifndef TESTCOORDINATES_H_
#define TESTCOORDINATES_H_

#ifdef TEST_COORDINATES
#define RUN_COORDINATES_TEST { CoordinatesTest(); }
#else
#define RUN_COORDINATES_TEST { ; }
#endif

void
CoordinatesTest();

#endif
