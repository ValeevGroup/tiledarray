#include <TiledArray/config.h>
#include <TiledArray/initialize.h>
#include <TiledArray/util/threads.h>

#ifdef TILEDARRAY_HAS_CUDA
#include <TiledArray/cuda/cublas.h>
#include <TiledArray/external/cuda.h>
#include <cutt.h>
#endif

namespace TiledArray {
namespace {

#ifdef TILEDARRAY_HAS_CUDA
/// initialize cuda environment
inline void cuda_initialize() {
  /// initialize cudaGlobal
  cudaEnv::instance();
  //
  cuBLASHandlePool::handle();
  // initialize cuTT
  cuttInitialize();
}

/// finalize cuda environment
inline void cuda_finalize() {
  CudaSafeCall(cudaDeviceSynchronize());
  cuttFinalize();
  cublasDestroy(cuBLASHandlePool::handle());
  delete &cuBLASHandlePool::handle();
  cudaEnv::instance().reset(nullptr);
}
#endif

inline bool& initialized_madworld_accessor() {
  static bool flag = false;
  return flag;
}
inline bool initialized_madworld() { return initialized_madworld_accessor(); }
inline bool& initialized_accessor() {
  static bool flag = false;
  return flag;
}
inline bool& finalized_accessor() {
  static bool flag = false;
  return flag;
}

}  // namespace
}  // namespace TiledArray

/// @return true if TiledArray (and, necessarily, MADWorld runtime) is in an
/// initialized state
bool TiledArray::initialized() { return initialized_accessor(); }

/// @return true if TiledArray has been finalized at least once
bool TiledArray::finalized() { return finalized_accessor(); }

/// @name TiledArray initialization.
///       These functions initialize TiledArray and (if needed) MADWorld
///       runtime.
/// @note the default World object is set to the object returned by these.
/// @warning MADWorld can only be initialized/finalized once, hence if
/// TiledArray initializes MADWorld
///          it can also be initialized/finalized only once.

/// @{

/// @throw TiledArray::Exception if TiledArray initialized MADWorld and
/// TiledArray::finalize() had been called
TiledArray::World& TiledArray::initialize(int& argc, char**& argv,
                                          const SafeMPI::Intracomm& comm,
                                          bool quiet) {
  if (initialized_madworld() && finalized())
    throw Exception(
        "TiledArray finalized MADWorld already, cannot re-initialize MADWorld "
        "again");
  if (!initialized()) {
    if (!madness::initialized())
      initialized_madworld_accessor() = true;
    else {  // if MADWorld initialized, we must assume that comm is its default
            // World.
      if (madness::World::is_default(comm))
        throw Exception(
            "MADWorld initialized before TiledArray::initialize(argc, argv, "
            "comm), but not initialized with comm");
    }
    auto& default_world = initialized_madworld()
                              ? madness::initialize(argc, argv, comm, quiet)
                              : *madness::World::find_instance(comm);
    TiledArray::set_default_world(default_world);
#ifdef TILEDARRAY_HAS_CUDA
    TiledArray::cuda_initialize();
#endif
    TiledArray::max_threads = TiledArray::get_num_threads();
    TiledArray::set_num_threads(1);
    madness::print_meminfo_disable();
    initialized_accessor() = true;
    return default_world;
  } else
    throw Exception("TiledArray already initialized");
}

/// Finalizes TiledArray (and MADWorld runtime, if it had not been initialized
/// when TiledArray::initialize was called).
void TiledArray::finalize() {
  // reset number of threads
  TiledArray::set_num_threads(TiledArray::max_threads);
#ifdef TILEDARRAY_HAS_CUDA
  TiledArray::cuda_finalize();
#endif
  TiledArray::get_default_world()
      .gop.fence();  // TODO remove when madness::finalize() fences
  if (initialized_madworld()) {
    madness::finalize();
  }
  TiledArray::reset_default_world();
  initialized_accessor() = false;
  finalized_accessor() = true;
}

void TiledArray::ta_abort() {
  std::abort();
}

void TiledArray::ta_abort(const std::string &m) {
  std::cerr << m << std::endl;
  ta_abort();
}
