#ifndef ORTHOTOPETEST_H_
#define ORTHOTOPETEST_H_

#ifdef TEST_ORTHOTOPE
#define RUN_ORTHOTOPE_TEST	{ OrthotopeTest(); }
#else
#define RUN_ORTHOTOPE_TEST	{ ; }
#endif

void
OrthotopeTest();

#endif /*ORTHOTOPETEST_H_*/
