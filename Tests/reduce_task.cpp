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

BOOST_AUTO_TEST_CASE( reduce_value )
{
  int sum = 0;
  for(int i = 0; i < 100; ++i) {
    sum += i;
    rt.add(i);
  }

  madness::Future<int> result = rt.submit();

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

  madness::Future<int> result = rt.submit();

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
