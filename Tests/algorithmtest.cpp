#include "permutation.h"
#include "coordinates.h" // for boost array output
#include <boost/test/unit_test.hpp>
#include <boost/test/output_test_stream.hpp>

using namespace TiledArray;

struct AlgorithmFixture {
  AlgorithmFixture() : zero(ndim, 0), unit(ndim, 1) {
    std::size_t nn[ndim] = {19, 23, 29, 31, 37};
    n.resize(ndim);
    std::copy(nn, nn+ndim, n.begin());

    std::size_t size = 1;
    for(std::size_t d=0; d<ndim; ++d) size *= n[d];
    v.resize(size);
    std::fill(v.begin(), v.end(), 1);
  }
  ~AlgorithmFixture() {}

  template <typename RandIter> struct Counter {
    typedef typename std::iterator_traits<RandIter>::value_type value_t;
    Counter(value_t& i) : i_(i) {}
    value_t& i_;
    void operator()(const RandIter& it) { i_ += *it; }
  };

  const static std::size_t ndim = 5;

  typedef std::vector<int>::const_iterator iter;
  typedef detail::ForLoop<Counter<iter>, iter> ForLoop1;
  typedef detail::NestedForLoop<2, Counter<iter>, iter> ForLoop2;
  typedef detail::NestedForLoop<2, Counter<int*>, int*> ForLoop2_ptr;
  typedef detail::NestedForLoop<ndim, Counter<iter>, iter> ForLoopN;
  typedef detail::NestedForLoop<ndim, Counter<int*>, int*> ForLoopN_ptr;

  std::vector<std::size_t> zero;
  std::vector<std::size_t> unit;
  std::vector<std::size_t> n;
  std::vector<int> v;
};

BOOST_FIXTURE_TEST_SUITE( algorithm_suite, AlgorithmFixture )

BOOST_AUTO_TEST_CASE( constructor_loop1 )
{
  int c = 0;
  Counter<iter> functor(c);
  BOOST_REQUIRE_NO_THROW( ForLoop1 l0 (functor, (ForLoop1::diff_t)n[0]) );
}

BOOST_AUTO_TEST_CASE( constructor_loop2 )
{
  int c = 0;
  Counter<iter> functor(c);
  BOOST_REQUIRE_NO_THROW( ForLoop2 l0(functor, n.begin(), n.begin() + 2, unit.begin(), unit.begin() + 2) );
}

BOOST_AUTO_TEST_CASE( constructor_loopN )
{
  int c = 0;
  Counter<iter> functor(c);
  BOOST_REQUIRE_NO_THROW( ForLoopN l0(functor, n.begin(), n.end(), unit.begin(), unit.end()) );
}

BOOST_AUTO_TEST_CASE( evaluate_loop1 )
{
  int c = 0;
  Counter<iter> functor(c);
  ForLoop1 l0 (functor, (ForLoop1::diff_t)n[0]); l0(v.begin());
  BOOST_CHECK_EQUAL((std::size_t)c, n[0]);
}

BOOST_AUTO_TEST_CASE( evaluate_loop2 )
{
  { // outer dim = 0, inner dim = 1
    int c = 0;
    Counter<iter> functor(c);
    ForLoop2 l0(functor, n.begin(), n.begin() + 2, unit.begin(), unit.begin() + 2); l0(v.begin());
    BOOST_CHECK_EQUAL((std::size_t)c, n[0]*n[1]);
  }
  { // outer dim = 4, inner dim = 3
    int c = 0;
    Counter<iter> functor(c);
    ForLoop2 l0(functor, n.rbegin(), n.rbegin() + 2, unit.rbegin(), unit.rbegin() + 2); l0(v.begin());
    BOOST_CHECK_EQUAL((std::size_t)c, n[4]*n[3]);
  }
  { // outer dim = 0, inner dim = 1
    int c = 0;
    Counter<int*> functor(c);
    ForLoop2_ptr l0(functor, n.begin(), n.begin() + 2, unit.begin(), unit.begin() + 2); l0(&(v[0]));
    BOOST_CHECK_EQUAL((std::size_t)c, n[0]*n[1]);
  }
}

BOOST_AUTO_TEST_CASE( evaluate_loopN )
{
  { // outer dim = 0, inner dim = 4
    int c = 0;
    Counter<iter> functor(c);
    ForLoopN l0(functor, n.begin(), n.end(), unit.begin(), unit.end()); l0(v.begin());
    BOOST_CHECK_EQUAL((std::size_t)c, v.size());
  }
  { // outer dim = 4, inner dim = 0
    int c = 0;
    Counter<iter> functor(c);
    ForLoopN l0(functor, n.rbegin(), n.rend(), unit.rbegin(), unit.rend()); l0(v.begin());
    BOOST_CHECK_EQUAL((std::size_t)c, v.size());
  }
  { // outer dim = 0, inner dim = 4
    int c = 0;
    Counter<int*> functor(c);
    ForLoopN_ptr l0(functor, n.begin(), n.end(), unit.begin(), unit.end()); l0(&(v[0]));
    BOOST_CHECK_EQUAL((std::size_t)c, v.size());
  }
}

BOOST_AUTO_TEST_SUITE_END()
