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
World& initialize(
  int& argc, char**& argv,
  const SafeMPI::Intracomm& comm,
  bool quiet = true
);

inline World& initialize(int& argc, char**& argv, bool quiet = true) {
  return TiledArray::initialize(argc, argv, SafeMPI::COMM_WORLD, quiet);
}

inline World& initialize(int& argc, char**& argv, const MPI_Comm& comm,
                         bool quiet = true) {
  return TiledArray::initialize(argc, argv, SafeMPI::Intracomm(comm), quiet);
}

/// @}

/// Finalizes TiledArray (and MADWorld runtime, if it had not been initialized
/// when TiledArray::initialize was called).
void finalize();

void taskq_wait_busy();
void taskq_wait_yield();
void taskq_wait_usleep(int);

}  // namespace TiledArray

#endif  // TILEDARRAY_INITIALIZE_H__INCLUDED
