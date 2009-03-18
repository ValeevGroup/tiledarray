#ifndef ARRAYTEST_H__INCLUDED
#define ARRAYTEST_H__INCLUDED

#ifdef TEST_ARRAY
#define RUN_ARRAY_TEST	{ ArrayTest(); }
#else
#define RUN_ARRAY_TEST	{ ; }
#endif

void ArrayTest();

#endif // ARRAYTEST_H__INCLUDED
