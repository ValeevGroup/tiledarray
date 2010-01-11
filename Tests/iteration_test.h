#ifndef ITERATIONTEST_H__INCLUDED
#define ITERATIONTEST_H__INCLUDED

/// Tests basic iteration functionality and compares the elements in the container
/// with expected values. It returns the number of values that were equal.
template<typename Container, typename InIter>
unsigned int iteration_test(Container&, InIter, InIter);
template<typename Container, typename InIter>
unsigned int const_iteration_test(const Container&, InIter, InIter);

/// Tests basic iteration functionality and compares the elements in the container
/// with expected values. It returns the number of values that were equal.
template<typename Container, typename InIter>
unsigned int iteration_test(Container& c, InIter first, InIter last) {
  unsigned int result = 0;
  for(typename Container::iterator it = c.begin(); it != c.end() && first != last; ++it, ++first )
    if( *it == *first)
      ++result;

  return true;
}

/// Tests basic const_iteration functionality and compares the elements in the
/// container with expected values. It returns the number of values that were equal.
template<typename Container, typename InIter>
unsigned int const_iteration_test(const Container& c, InIter first, InIter last) {
  unsigned int result = 0;
  for(typename Container::const_iterator it = c.begin(); it != c.end() && first != last; ++it, ++first )
    if( *it == *first)
      ++result;

  return result;
}

#endif // ITERATIONTEST_H__INCLUDED
