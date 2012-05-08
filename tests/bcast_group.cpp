#include "TiledArray/bcast_group.h"
#include "unit_test_config.h"

struct BcastGroupFixture {
  BcastGroupFixture() {
    count = 0;
    for(ProcessID p = 0; p < GlobalFixture::world->size(); ++p)
      group.push_back(p);
  }

  madness::AtomicInt count; // Counter for remote tasks to increment
  std::vector<ProcessID> group; // The processes that are in the group
}; // BcastGroupFixture

// Task object that will be broadcast and run on each node
struct BcastTask {
  // Default constructor
  BcastTask() : world_(NULL), wptr_() { }

  // Task object constructor
  BcastTask(madness::World* world, const madness::AtomicInt& count) :
      world_(world), wptr_(*world, const_cast<madness::AtomicInt*>(&count))
  { }

  // Run task
  void operator()() const {
    world_->taskq.add(wptr_.owner(), & return_task, wptr_);
  }

  // Serialize the task object
  template <typename Archive>
  void serialize(Archive& ar) {
    ar & world_ & wptr_;
  }

private:

  // The task that is run on the calling node to verify the task was run on the remote node
  static madness::Void return_task(const madness::detail::WorldPtr<madness::AtomicInt>& wptr) {
    ++(*wptr);
    return madness::None;
  }

  madness::World* world_; // A pointer to the world
  madness::detail::WorldPtr<madness::AtomicInt> wptr_; // A world pointer to the root counter
}; // BcastTask

BOOST_FIXTURE_TEST_SUITE( bcast_group_suite , BcastGroupFixture )

BOOST_AUTO_TEST_CASE( bcast )
{
  // Do test where each process is the root
  for(ProcessID p = 0; p < GlobalFixture::world->size(); ++p) {
    if(p == GlobalFixture::world->rank()) {
      // Construct the broadcast objects
      TiledArray::detail::BcastGroup bg(* GlobalFixture::world, group);

      // Broadcast the task
      bg(BcastTask(GlobalFixture::world, count));
      GlobalFixture::world->gop.fence();

      // Check that everyone ran the task.
      BOOST_CHECK_EQUAL(count, GlobalFixture::world->size());
    }
  }
}

BOOST_AUTO_TEST_SUITE_END()
