#ifndef RANGETEST_H_
#define RANGETEST_H_

#ifdef TEST_RANGE
#define RUN_RANGE_TEST	{ RangeTest(); }
#else
#define RUN_RANGE_TEST	{ ; }
#endif

void
RangeTest();

#endif /*RANGETEST_H_*/
