#ifndef ARRAYTEST_H__INCLUDED
#define ARRAYTEST_H__INCLUDED

#include <madness_runtime.h>

#ifdef TEST_ARRAY
#define RUN_ARRAY_TEST	{ ArrayTest(world); }
#else
#define RUN_ARRAY_TEST	{ ; }
#endif

extern void ArrayTest(madness::World& world);

#endif // ARRAYTEST_H__INCLUDED
