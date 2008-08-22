#ifndef RANGETEST_H__INCLUDED
#define RANGETEST_H__INCLUDED

#ifdef TEST_RANGE
#define RUN_RANGE_TEST	{ RangeTest(); }
#else
#define RUN_RANGE_TEST	{ ; }
#endif

void RangeTest();

#endif // RANGETEST_H__INCLUDED
