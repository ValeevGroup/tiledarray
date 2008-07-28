#ifndef SHAPETEST_H_
#define SHAPETEST_H_

#ifdef TEST_SHAPE
#define RUN_SHAPE_TEST	{ ShapeTest(); }
#else
#define RUN_SHAPE_TEST	{ ; }
#endif

void
ShapeTest();

#endif /*SHAPETEST_H_*/
