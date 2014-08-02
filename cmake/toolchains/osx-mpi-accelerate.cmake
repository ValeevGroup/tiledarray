set(CMAKE_SYSTEM_NAME Darwin)

# Set compilers (assumes the compilers are in the PATH)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(MPI_C_COMPILER mpicc)
set(MPI_CXX_COMPILER mpicxx)

set(BLAS_FLAGS "-framework Accelerate")
set(LAPACK_FLAGS "-framework Accelerate")
set(INTEGER4 TRUE)

if(NOT ${CMAKE_SYSTEM_VERSION} VERSION_LESS 11.0)
  # Building on OS X 10.7 or later, so add "-Wl,-no_pie" linker flags.
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no_pie")
  string(STRIP "${CMAKE_EXE_LINKER_FLAGS}" CMAKE_EXE_LINKER_FLAGS)
  set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-no_pie")
  string(STRIP "${CMAKE_SHARED_LINKER_FLAGS}" CMAKE_SHARED_LINKER_FLAGS)
  set(CMAKE_STATIC_LINKER_FLAGS "${CMAKE_STATIC_LINKER_FLAGS} -Wl,-no_pie")
  string(STRIP "${CMAKE_STATIC_LINKER_FLAGS}" CMAKE_STATIC_LINKER_FLAGS)
endif()