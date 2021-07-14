#include <TiledArray/config.h>
#include "TiledArray/util/threads.h"

#ifdef TILEDARRAY_HAS_INTEL_MKL
#include <mkl_service.h>
#endif

int TiledArray::max_threads = 1;

int TiledArray::get_num_threads() {
#ifdef TILEDARRAY_HAS_INTEL_MKL
  return mkl_get_max_threads();
#endif
  return 1;
}

void TiledArray::set_num_threads(int n) {
#ifdef TILEDARRAY_HAS_INTEL_MKL
  mkl_set_num_threads(n);
#endif
}
