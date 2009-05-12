#ifndef PERMUTATIONTEST_H__INCLUDED
#define PERMUTATIONTEST_H__INCLUDED

#ifdef TEST_PERMUTATION
#define RUN_PERMUTATION_TEST { PermutationTest(); }
#else
#define RUN_PERMUTATION_TEST { ; }
#endif

void
PermutationTest();

#endif // PERMUTATIONTEST_H__INCLUDED
