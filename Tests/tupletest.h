#ifndef TESTTUPLE_H_
#define TESTTUPLE_H_

#ifdef TEST_TUPLE
#define RUN_TUPLE_TEST { TupleTest(); }
#else
#define RUN_TUPLE_TEST { ; }
#endif

void
TupleTest();

#endif /*TESTTUPLE_H_*/
