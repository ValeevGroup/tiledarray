set(CMAKE_SYSTEM_NAME Darwin)

# Set compilers (assumes the compilers are in the PATH)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(MPI_C_COMPILER mpicc)
set(MPI_CXX_COMPILER mpicxx)

# Set compile flags
if(CMAKE_CXX_FLAGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -stdlib=libc++" CACHE STRING "C++ compile flags")
else()
  set(CMAKE_CXX_FLAGS "-std=c++11 -stdlib=libc++" CACHE STRING "C++ compile flags")
endif()

if(NOT ${CMAKE_SYSTEM_VERSION} VERSION_LESS 11.0)
  # Building on OS X 10.7 or later, so add "-Wl,-no_pie" linker flags.
  if(CMAKE_EXE_LINKER_FLAGS)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no_pie" CACHE STRING "Executable linker flags")
  else()
    set(CMAKE_EXE_LINKER_FLAGS "-Wl,-no_pie" CACHE STRING "Executable linker flags")
  endif()
endif()

# Set BLAS/LAPACK flags
set(BLAS_LINKER_FLAGS "-framework Accelerate" CACHE STRING "BLAS linker flags")
set(LAPACK_LINKER_FLAGS "-framework Accelerate" CACHE STRING "LAPACK linker flags")
set(INTEGER4 TRUE CACHE BOOL "Set Fortran integer size to 4 bytes")