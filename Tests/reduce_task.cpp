#include "TiledArray/reduce_task.h"
#include "unit_test_config.h"
#include <functional>

using namespace TiledArray;
using namespace TiledArray::detail;

struct ReduceTaskFixture {

  ReduceTaskFixture() : world(*GlobalFixture::world), rt(world, std::plus<int>()) {

  }

  madness::World& world;
  ReduceTask<int, std::plus<int> > rt;

};

BOOST_FIXTURE_TEST_SUITE( reduce_task_suite, ReduceTaskFixture )

BOOST_AUTO_TEST_CASE( constructors )
{
  BOOST_REQUIRE_NO_THROW( (ReduceTaskImpl<int, std::plus<int> >(world, std::plus<int>())) );
  BOOST_REQUIRE_NO_THROW( (ReduceTask<int, std::plus<int> >(world, std::plus<int>())) );
  BOOST_REQUIRE_NO_THROW( (ReduceTask<int, std::plus<int> >(rt)) );
}

BOOST_AUTO_TEST_CASE( impl_reduce_op )
{
  ReduceTaskImpl<int, std::plus<int> > impl(world, std::plus<int>());

  // Check that reduce operation works correctly
  madness::Future<int> v1(5);
  madness::Future<int> v2(7);

  BOOST_CHECK_EQUAL(impl.reduce(v1,v2), 12);
}

BOOST_AUTO_TEST_CASE( reduce_op )
{
  // Check that reduce operation works correctly
  BOOST_CHECK_EQUAL(rt(2,3), 5);

  // Check that reduce operation works correctly for futures
  madness::Future<int> v1(5);
  madness::Future<int> v2(7);

  BOOST_CHECK_EQUAL(rt(v1,v2), 12);
}

BOOST_AUTO_TEST_CASE( impl_add_future )
{
  ReduceTaskImpl<int, std::plus<int> > impl(world, std::plus<int>());

  BOOST_CHECK_EQUAL(impl.size(), 0);

  for(int i = 1; i < 10; ++i) {
    madness::Future<int> f;
    impl.add(f);

    BOOST_CHECK_EQUAL(impl.size(), i);
  }

  // Check the range
  madness::Range<ReduceTaskImpl<int, std::plus<int> >::container_type::iterator>
    range = impl.range(8);

  // Check the range size
  BOOST_CHECK_EQUAL(range.size(), 9);
  BOOST_CHECK_EQUAL(std::distance(range.begin(), range.end()), 9);

  // Check that none of the elements in the range have been set.
  for(madness::Range<ReduceTaskImpl<int, std::plus<int> >::container_type::iterator>::iterator it = range.begin(); it != range.end(); ++it)
    BOOST_CHECK(!(it->probe()));
}

BOOST_AUTO_TEST_CASE( add_value )
{
  BOOST_CHECK_EQUAL(rt.size(), 0);

  for(int i = 1; i < 10; ++i) {
    rt.add(i);

    BOOST_CHECK_EQUAL(rt.size(), i);
  }
}

BOOST_AUTO_TEST_CASE( add_future )
{
  BOOST_CHECK_EQUAL(rt.size(), 0);

  for(int i = 1; i < 10; ++i) {
    madness::Future<int> f;
    rt.add(f);

    BOOST_CHECK_EQUAL(rt.size(), i);
  }
}


BOOST_AUTO_TEST_CASE( reduce_value )
{
  int sum = 0;
  for(int i = 0; i < 100; ++i) {
    sum += i;
    rt.add(i);
  }

  madness::Future<int> result = rt();

  BOOST_CHECK_EQUAL(result.get(), sum);
}

BOOST_AUTO_TEST_CASE( reduce_future )
{
  std::vector<madness::Future<int> > fut_vec;

  for(int i = 0; i < 100; ++i) {
    madness::Future<int> f;
    fut_vec.push_back(f);
    rt.add(f);
  }

  madness::Future<int> result = rt();

  BOOST_CHECK(!(result.probe()));

  int sum = 0;
  for(int i = 0; i < 99; ++i) {
    sum += i;
    fut_vec[i].set(i);
    BOOST_CHECK(!(result.probe()));
  }

  sum += 99;
  fut_vec[99].set(99);

  world.gop.fence();

  BOOST_CHECK(result.probe());

  BOOST_CHECK_EQUAL(result.get(), sum);

}


BOOST_AUTO_TEST_SUITE_END()
