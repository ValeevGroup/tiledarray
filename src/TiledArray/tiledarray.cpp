#include <TiledArray/config.h>
#include <TiledArray/initialize.h>
#include <TiledArray/util/threads.h>

#include <TiledArray/math/linalg/basic.h>

#include <madness/world/safempi.h>

#ifdef TILEDARRAY_HAS_CUDA
#include <TiledArray/cuda/cublas.h>
#include <TiledArray/external/cuda.h>
#include <librett.h>
#endif

#if TILEDARRAY_HAS_TTG
#include <ttg.h>
#endif

#include <cerrno>
#include <cstdlib>

namespace TiledArray {
namespace {

#ifdef TILEDARRAY_HAS_CUDA
/// initialize cuda environment
inline void cuda_initialize() {
  /// initialize cudaGlobal
  cudaEnv::instance();
  //
  cuBLASHandlePool::handle();
  // initialize LibreTT
  librettInitialize();
}

/// finalize cuda environment
inline void cuda_finalize() {
  CudaSafeCall(cudaDeviceSynchronize());
  librettFinalize();
  cublasDestroy(cuBLASHandlePool::handle());
  delete &cuBLASHandlePool::handle();
  // although TA::cudaEnv is a singleton, must explicitly delete it so
  // that CUDA runtime is not finalized before the cudaEnv dtor is called
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
    if (!madness::initialized()) {
      initialized_madworld_accessor() = true;
    } else {  // if MADWorld initialized, we must assume that comm is its
              // default World.
      if (!madness::World::is_default(comm))
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

    // if have TTG initialize it also
#if TILEDARRAY_HAS_TTG
    ttg::initialize(argc, argv, -1, madness::ParsecRuntime::context());
#endif

    // check if user specified linear algebra backend + params
    auto* linalg_backend_cstr = std::getenv("TA_LINALG_BACKEND");
    if (linalg_backend_cstr) {
      using namespace std::literals::string_literals;
      if ("scalapack"s == linalg_backend_cstr) {
        TiledArray::set_linalg_backend(
            TiledArray::LinearAlgebraBackend::ScaLAPACK);
      } else if ("ttg"s == linalg_backend_cstr) {
        TiledArray::set_linalg_backend(TiledArray::LinearAlgebraBackend::TTG);
      } else if ("lapack"s == linalg_backend_cstr) {
        TiledArray::set_linalg_backend(
            TiledArray::LinearAlgebraBackend::LAPACK);
      } else
        TA_EXCEPTION(
            "TiledArray::initialize: invalid value of environment variable "
            "TA_LINALG_BACKEND, valid values are \"scalapack\", \"ttg\", and "
            "\"lapack\"");
    }
    const char* linalg_distributed_minsize_cstr =
        std::getenv("TA_LINALG_DISTRIBUTED_MINSIZE");
    if (linalg_distributed_minsize_cstr) {
      char* end;
      const auto linalg_distributed_minsize =
          std::strtoul(linalg_distributed_minsize_cstr, &end, 10);
      if (errno == ERANGE)
        TA_EXCEPTION(
            "TiledArray::initialize: invalid value of environment variable "
            "TA_LINALG_DISTRIBUTED_MINSIZE");
      TiledArray::set_linalg_crossover_to_distributed(
          linalg_distributed_minsize);
    }

    return default_world;
  } else
    throw Exception("TiledArray already initialized");
}

/// Finalizes TiledArray (and MADWorld runtime, if it had not been initialized
/// when TiledArray::initialize was called).
void TiledArray::finalize() {
  // finalize in the reverse order of initialize
#if TILEDARRAY_HAS_TTG
  ttg::finalize();
#endif

  // reset number of threads
  TiledArray::set_num_threads(TiledArray::max_threads);
  TiledArray::get_default_world().gop.fence();  // this should ensure no pending
                                                // tasks using cuda allocators
#ifdef TILEDARRAY_HAS_CUDA
  TiledArray::cuda_finalize();
#endif
  if (initialized_madworld()) {
    madness::finalize();
  }
  TiledArray::reset_default_world();
  initialized_accessor() = false;
  finalized_accessor() = true;
}

TiledArray::detail::Finalizer::~Finalizer() noexcept {
  static std::mutex mtx;
  std::scoped_lock lock(mtx);
  if (TiledArray::initialized()) {
    TiledArray::finalize();
  }
}

TiledArray::detail::Finalizer TiledArray::scoped_finalizer() { return {}; }

void TiledArray::ta_abort() { SafeMPI::COMM_WORLD.Abort(); }

void TiledArray::ta_abort(const std::string& m) {
  std::cerr << m << std::endl;
  ta_abort();
}

void TiledArray::taskq_wait_busy() {
  madness::threadpool_wait_policy(madness::WaitPolicy::Busy);
}

void TiledArray::taskq_wait_yield() {
  madness::threadpool_wait_policy(madness::WaitPolicy::Yield);
}

void TiledArray::taskq_wait_usleep(int us) {
  madness::threadpool_wait_policy(madness::WaitPolicy::Sleep, us);
}
