#ifndef ARRAYTEST_H__INCLUDED
#define ARRAYTEST_H__INCLUDED

#ifdef TEST_ARRAY
#define RUN_ARRAY_TEST	{ ArrayTest(argc,argv); }
#else
#define RUN_ARRAY_TEST	{ ; }
#endif

void ArrayTest(int argc, char* argv[]);

#endif // ARRAYTEST_H__INCLUDED
