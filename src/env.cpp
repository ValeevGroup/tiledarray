#include <env.h>

///////////////

RuntimeEnvironment::RuntimeEnvironment() {}

RuntimeEnvironment::~RuntimeEnvironment() {}

///////////////

TestRuntimeEnvironment::TestRuntimeEnvironment(unsigned int nproc) : nproc_(nproc) {}

TestRuntimeEnvironment::~TestRuntimeEnvironment() {}

unsigned int
TestRuntimeEnvironment::nproc() const {
  return nproc_;
}

///////////////

SerialRuntimeEnvironment::SerialRuntimeEnvironment() {}

SerialRuntimeEnvironment::~SerialRuntimeEnvironment() {}

unsigned int
SerialRuntimeEnvironment::nproc() const {
  return 1;
}

