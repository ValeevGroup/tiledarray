
#include <stdexcept>
#include <env.h>

using namespace TiledArray;

///////////

RuntimeEnvironment::RuntimeEnvironment() {}

RuntimeEnvironment::~RuntimeEnvironment() {}

RuntimeEnvironment* RuntimeEnvironment::instance_;

RuntimeEnvironment&
RuntimeEnvironment::Instance() {
  if (instance_)
    return *instance_;
  throw std::runtime_error("RuntimeEnvironment::Instance() -- instance has not been created yet");
}

///////////

TestRuntimeEnvironment::TestRuntimeEnvironment(unsigned int nproc, unsigned int me) :
  nproc_(nproc),
  me_(me)
  {
  }

TestRuntimeEnvironment::~TestRuntimeEnvironment() {}

void
TestRuntimeEnvironment::CreateInstance(unsigned int nproc, unsigned int me) {
  if (instance_ != 0)
    throw std::runtime_error("TestRuntimeEnvironment::CreateInstance() -- instance has already been created");
  instance_ = new TestRuntimeEnvironment(nproc, me);
}

void
TestRuntimeEnvironment::DestroyInstance() {
  delete instance_;
  instance_ = 0;
}

unsigned int
TestRuntimeEnvironment::nproc() const {
  return nproc_;
}

unsigned int
TestRuntimeEnvironment::me() const {
  return me_;
}
