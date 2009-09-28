#include "permutation.h"
#include "coordinates.h" // for boost array output
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

using namespace TiledArray;

struct AlgorithmFixture {
  AlgorithmFixture() : n(119), v(n, 1) {}
  ~AlgorithmFixture() {}

  template <typename RandIter> struct Counter {
    typedef typename std::iterator_traits<RandIter>::value_type value_t;
    Counter(value_t& i) : i_(i) {}
    value_t& i_;
    void operator()(const RandIter& it) { i_ += *it; }
  };

  typedef std::vector<int>::const_iterator iter;
  typedef detail::ForLoop<Counter<iter>, iter> ForLoop1;

  std::size_t n;
  std::vector<int> v;
};

BOOST_FIXTURE_TEST_SUITE( algorithm_suite, AlgorithmFixture )

BOOST_AUTO_TEST_CASE( constructor )
{
  int c = 0;
  Counter<iter> functor(c);
  BOOST_REQUIRE_NO_THROW( ForLoop1 l0 (functor, (ForLoop1::diff_t)n) );
}

BOOST_AUTO_TEST_CASE( evaluate )
{
  int c = 0;
  Counter<iter> functor(c);
  ForLoop1 l0 (functor, (ForLoop1::diff_t)n); l0(v.begin());
  BOOST_CHECK_EQUAL(c, n);
}

BOOST_AUTO_TEST_SUITE_END()
