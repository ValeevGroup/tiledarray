//
// Created by Chong Peng on 7/27/18.
//

#ifndef TILEDARRAY_INITIALIZE_H__INCLUDED
#define TILEDARRAY_INITIALIZE_H__INCLUDED


#include <TiledArray/config.h>

#include <TiledArray/external/madness.h>
#ifdef TILEDARRAY_HAS_CUDA
#include <TiledArray/external/cuda.h>
#endif

namespace TiledArray{

/// @name TiledArray initialization.
///       These functions initialize TiledArray AND MADWorld runtime components.
///       @note the default World object is set to the object returned by these.

/// @{
inline World& initialize(int& argc, char**& argv, const SafeMPI::Intracomm& comm) {
  auto& default_world = madness::initialize(argc, argv, comm);
  TiledArray::set_default_world(default_world);
#ifdef TILEDARRAY_HAS_CUDA
    TiledArray::cuda_initialize();
#endif
  return default_world;
}

inline World& initialize(int& argc, char**& argv) {
  return TiledArray::initialize(argc, argv, SafeMPI::COMM_WORLD);
}

inline World& initialize(int& argc, char**& argv, const MPI_Comm& comm) {
  return TiledArray::initialize(argc, argv, SafeMPI::Intracomm(comm));
}

inline void finalize() {
  madness::finalize();
  TiledArray::reset_default_world();
#ifdef TILEDARRAY_HAS_CUDA
    TiledArray::cuda_finalize();
#endif
}

/// @}

}

#endif //TILEDARRAY_INITIALIZE_H__INCLUDED
