#ifndef RANGE1TEST_H__INCLUDED
#define RANGE1TEST_H__INCLUDED

#ifdef TEST_RANGE1
#define RUN_RANGE1_TEST	{ Range1Test(); }
#else
#define RUN_RANGE1_TEST	{ ; }
#endif

void Range1Test();

#endif // RANGE1TEST_H__INCLUDED
