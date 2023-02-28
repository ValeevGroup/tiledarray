#include <tiledarray.h>

int main(int argc, char* argv[]) {
  assert(!TA::initialized());
  assert(!TA::finalized());

  try {
    // Initializes TiledArray
    auto& world = TA_SCOPED_INITIALIZE(argc, argv);

    // Do some work here.

    assert(TA::initialized());
    assert(!TA::finalized());
    // if (argc > 1) throw "";
  }  // TA::finalize() called when leaving this scope
  // exceptional return
  catch (...) {
    assert(!TA::initialized());
    assert(TA::finalized());
    std::cerr << "oops!\n";
    return 1;
  }

  // normal return
  assert(!TA::initialized());
  assert(TA::finalized());
  return 0;
}
