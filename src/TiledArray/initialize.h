//
// Created by Chong Peng on 7/27/18.
//

#ifndef TILEDARRAY_INITIALIZE_H__INCLUDED
#define TILEDARRAY_INITIALIZE_H__INCLUDED

#include <TiledArray/config.h>
#include <TiledArray/external/madness.h>

namespace TiledArray {

/// @return true if TiledArray (and, necessarily, MADWorld runtime) is in an
/// initialized state
bool initialized();

/// @return true if TiledArray has been finalized at least once
bool finalized();

/// @return true if TiledArray (and, necessarily, MADWorld runtime) was
/// initialized to be quiet
bool initialized_to_be_quiet();

// clang-format off
/// @name TiledArray initialization.
///       These functions initialize TiledArray and (if needed) MADWorld
///       runtime.
// clang-format on

/// @{

// clang-format off
/// @param[in] argc the number of non-null strings pointed to by @p argv
/// @param[in] argv array of `argc+1` pointers to strings (the last of which is null)
///            specifying arguments passed to madness::initialize, MPI_Init_threads, and other similar initializers;
/// @param[in] comm the MADNESS communicator (an madness::SafeMPI::Intracomm object) to use for TiledArray computation
/// @param[in] quiet if true, will prevent initializers from writing to standard streams, if possible; the default is true
/// @throw TiledArray::Exception if TiledArray initialized MADWorld and
/// TiledArray::finalize() had been called
/// @note - `argc` and `argv` are typically the values received by the main() function of the application
/// @note - variants of initialize that do not take `comm` will construct default communicator
/// @note - the default World object is set to the object returned by these.
/// @note - The following environment variables can be used to control TiledArray
///       initialization:
///       | Environment Variable | Default| Description |
///       |----------------------|--------|-------------|
///       | `TA_LINALG_BACKEND`  | none   | If set, chooses the linear algebra backend to use; valid values are `scalapack` (distributed library ScaLAPACK, only available if configured with `ENABLE_SCALAPACK=ON`), `lapack` (non-distributed library LAPACK, always available), and `ttg` (experimental [TTG](https://github.com/TESSEorg/TTG) backend, only implements Cholesky); the default is to choose best available backend automatically (recommended) |
///       | `TA_LINALG_DISTRIBUTED_MINSIZE`  | 4194304 | Unless `TA_LINALG_BACKEND` is set, this controls the minimum matrix size (#rows times #columns) for which the distributed backend if chosen when selecting the best available backend |
/// @warning MADWorld can only be initialized/finalized once, hence if
///          TiledArray initializes MADWorld
///          it can also be initialized/finalized only once.
// clang-format on
World& initialize(int& argc, char**& argv, const SafeMPI::Intracomm& comm,
                  bool quiet = true);

inline World& initialize(int& argc, char**& argv, bool quiet = true) {
  return TiledArray::initialize(argc, argv, SafeMPI::COMM_WORLD, quiet);
}

inline World& initialize(int& argc, char**& argv, const MPI_Comm& comm,
                         bool quiet = true) {
  return TiledArray::initialize(argc, argv, SafeMPI::Intracomm(comm), quiet);
}

/// @}

#ifndef TA_SCOPED_INITIALIZE
/// calling this will initialize TA and then finalize it when leaving this scope
#define TA_SCOPED_INITIALIZE(args...) \
  TiledArray::initialize(args);       \
  auto finalizer = TiledArray::scoped_finalizer();
#endif

/// Finalizes TiledArray (and MADWorld runtime, if it had not been initialized
/// when TiledArray::initialize was called).
void finalize();

namespace detail {
struct Finalizer {
  ~Finalizer() noexcept;
};
}  // namespace detail

/// creates an object whose destruction upon leaving this scope will cause
/// TiledArray::finalize to be called
detail::Finalizer scoped_finalizer();

#ifndef TA_FINALIZE_AFTER_LEAVING_THIS_SCOPE
/// calling this will cause TiledArray::finalize() to be called (if needed)
/// upon leaving this scope
#define TA_FINALIZE_AFTER_LEAVING_THIS_SCOPE() \
  auto finalizer = TiledArray::scoped_finalizer();
#endif

void taskq_wait_busy();
void taskq_wait_yield();
void taskq_wait_usleep(int);

}  // namespace TiledArray

#endif  // TILEDARRAY_INITIALIZE_H__INCLUDED
