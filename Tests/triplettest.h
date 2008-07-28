#ifndef TRIPLETEST_H_
#define TRIPLETEST_H_

#ifdef TEST_TRIPLET
#define RUN_TRIPLET_TEST { TripletTest(); }
#else
#define RUN_TRIPLET_TEST { ; }
#endif

void
TripletTest();

#endif /*TRIPLETEST_H_*/
