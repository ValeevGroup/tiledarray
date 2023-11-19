#include <TiledArray/config.h>
#include <TiledArray/initialize.h>
#include <TiledArray/util/threads.h>

#include <TiledArray/math/linalg/basic.h>

#include <madness/world/safempi.h>

#ifdef TILEDARRAY_HAS_DEVICE
#include <TiledArray/device/blas.h>
#include <TiledArray/external/device.h>
#include <librett.h>
#endif

#if TILEDARRAY_HAS_TTG
#include <ttg.h>
#endif

#include <cerrno>
#include <cstdlib>

namespace TiledArray {
namespace {

#ifdef TILEDARRAY_HAS_DEVICE
/// initialize cuda environment
inline void device_initialize() {
  /// initialize deviceEnv
  deviceEnv::instance();
#if defined(TILEDARRAY_HAS_DEVICE)
  BLASQueuePool::initialize();
#endif
  // initialize LibreTT
  librettInitialize();
}

/// finalize cuda environment
inline void device_finalize() {
  DeviceSafeCall(device::deviceSynchronize());
  librettFinalize();
#if defined(TILEDARRAY_HAS_DEVICE)
  BLASQueuePool::finalize();
#endif
  // although TA::deviceEnv is a singleton, must explicitly delete it so
  // that the device runtime is not finalized before the deviceEnv dtor is
  // called
  deviceEnv::instance().reset(nullptr);
}
#endif  // TILEDARRAY_HAS_DEVICE

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

inline bool& initialized_mpi_accessor() {
  static bool flag = false;
  return flag;
}

#if defined(TILEDARRAY_HAS_TTG) || defined(HAVE_PARSEC)
inline parsec_context_t*& initialized_parsec_ctx_accessor() {
  static parsec_context_t* ctx = nullptr;
  return ctx;
}
#endif

inline bool& quiet_accessor() {
  static bool quiet = false;
  return quiet;
}

}  // namespace
}  // namespace TiledArray

bool TiledArray::initialized() { return initialized_accessor(); }

bool TiledArray::finalized() { return finalized_accessor(); }

bool TiledArray::initialized_to_be_quiet() { return quiet_accessor(); }

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

    // if need to use TTG, and MADNESS uses PaRSEC *also*
    // must initialize PaRSEC here so that both MADNESS and TTG use
    // the same context
#if defined(TILEDARRAY_HAS_TTG) && defined(HAVE_PARSEC)
    if (!ttg::initialized()) {
      // preconditions:
      // - MADWorld is NOT initialized, or initialized and used PaRSEC context
      // using same communicator as comm
      if (madness::initialized()) {
        auto parsec_comm = reinterpret_cast<MPI_Comm>(
            madness::ParsecRuntime::context()->comm_ctx);
        int parsec_comm_equiv_comm = 0;
        MADNESS_MPI_TEST(MPI_Comm_compare(comm.Get_mpi_comm(), parsec_comm,
                                          &parsec_comm_equiv_comm));
        if (!parsec_comm_equiv_comm)
          throw std::runtime_error(
              "TiledArray::initialize(): existing MADWorld's PaRSEC context "
              "must use same communicator as given to TA::initialize()");
      } else {
        // if MPI is not initialized yet, initialize it here, and remember we
        // did that
        if (!SafeMPI::Is_initialized()) {
          SafeMPI::Init_thread(argc, argv, MPI_THREAD_MULTIPLE);
          initialized_mpi_accessor() = true;
        }

        // create a PaRSEC context using the given communicator
        const auto nthreads = madness::ThreadPool::default_nthread();
        const int num_parsec_threads = nthreads + 1;
        auto* ctx = parsec_init(num_parsec_threads, &argc, &argv);
        parsec_remote_dep_set_ctx(ctx, (intptr_t)comm.Get_mpi_comm());

        // tell MADWorld to use this context
        madness::ParsecRuntime::initialize_with_existing_context(ctx);

        // remember the context and free it in finalize()
        initialized_parsec_ctx_accessor() = ctx;
      }
    }
#endif

    // initialize MADWorld, if not yet initialized
    auto& default_world = initialized_madworld()
                              ? madness::initialize(argc, argv, comm, quiet)
                              : *madness::World::find_instance(comm);
    TiledArray::set_default_world(default_world);
#ifdef TILEDARRAY_HAS_DEVICE
    TiledArray::device_initialize();
#endif
    TiledArray::max_threads = TiledArray::get_num_threads();
    TiledArray::set_num_threads(1);
    madness::print_meminfo_disable();
    initialized_accessor() = true;
    quiet_accessor() = quiet;

    // if have TTG, initialize it also
#if defined(TILEDARRAY_HAS_TTG)
    // use same PaRSEC context as used by MADWorld if it uses PaRSEC
#if defined(HAVE_PARSEC)
    ttg::initialize(argc, argv, -1, madness::ParsecRuntime::context());
#else   // defined(HAVE_PARSEC)
    ttg::initialize(argc, argv, -1);
#endif  // defined(HAVE_PARSEC)
#endif  // defined(TILEDARRAY_HAS_TTG)

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

void TiledArray::finalize() {
  // finalize in the reverse order of initialize
#if TILEDARRAY_HAS_TTG
  ttg::finalize();
#endif

  // reset number of threads
  TiledArray::set_num_threads(TiledArray::max_threads);
  TiledArray::get_default_world().gop.fence();  // this should ensure no pending
                                                // tasks using cuda allocators
#ifdef TILEDARRAY_HAS_DEVICE
  TiledArray::device_finalize();
#endif
  if (initialized_madworld()) {
    madness::finalize();
  }
  TiledArray::reset_default_world();
#if defined(TILEDARRAY_HAS_TTG) && defined(HAVE_PARSEC)
  if (initialized_parsec_ctx_accessor()) {
    parsec_context_wait(initialized_parsec_ctx_accessor());
    parsec_fini(&initialized_parsec_ctx_accessor());
    initialized_parsec_ctx_accessor() = nullptr;
  }
  if (initialized_mpi_accessor()) {
    SafeMPI::Finalize();
  }
#endif  // defined(TILEDARRAY_HAS_TTG) && defined(HAVE_PARSEC)
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
