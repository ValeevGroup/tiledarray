#
# Generic Toolchain for GCC + Intel IMPI + MKL + TBB
#
# REQUIREMENTS:
# - in PATH:
#   gcc, g++, mpiicc, and mpiicpc
# - environment variables:
#   * INTEL_DIR: the Intel compiler directory (includes MKL and TBB), e.g. /opt/intel
#   * GCC_DIR: ${GCC_DIR}/bin/gcc points to the gcc compiler to use
#   * EIGEN3_DIR or (deprecated) EIGEN_DIR: the Eigen3 directory
#   * BOOST_DIR: the Boost root directory
#

# Set paths (add a cache entry for the paths used by the configure script)
set(GCC_ROOT_DIR "$ENV{GCC_DIR}")
set(MKL_ROOT_DIR "$ENV{INTEL_DIR}/mkl")
set(TBB_ROOT_DIR "$ENV{INTEL_DIR}/tbb" CACHE PATH "TBB root directory")
# query EIGEN3_DIR and (deprecated) EIGEN_DIR envvars
if (DEFINED ENV{EIGEN3_DIR})
  set(ENV_EIGEN3_DIR "$ENV{EIGEN3_DIR}")
else()
  set(ENV_EIGEN3_DIR "$ENV{EIGEN_DIR}")
endif()
set(EIGEN3_INCLUDE_DIR "${ENV_EIGEN3_DIR}" CACHE PATH "Eigen3 library directory")
set(BOOST_ROOT "$ENV{BOOST_DIR}" CACHE PATH "Boost root directory")

# Set compilers (assumes the compilers are in the PATH)
if ($ENV{GCC_VERSION})
  set(CMAKE_C_COMPILER gcc-$ENV{GCC_VERSION})
  set(CMAKE_CXX_COMPILER g++-$ENV{GCC_VERSION})
else ()
  set(CMAKE_C_COMPILER gcc)
  set(CMAKE_CXX_COMPILER g++)
endif ()
set(MPI_C_COMPILER mpiicc)
set(MPI_CXX_COMPILER mpiicpc)
set(MPI_CXX_SKIP_MPICXX ON)

# Set compile flags
set(CMAKE_C_FLAGS_INIT             "-std=c99" CACHE STRING "Initial C compile flags")
set(CMAKE_C_FLAGS_DEBUG            "-g -Wall" CACHE STRING "Initial C debug compile flags")
set(CMAKE_C_FLAGS_MINSIZEREL       "-Os -march=native -DNDEBUG" CACHE STRING "Initial C minimum size release compile flags")
set(CMAKE_C_FLAGS_RELEASE          "-O3 -march=native -DNDEBUG" CACHE STRING "Initial C release compile flags")
set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-Og -g -march=native -Wall" CACHE STRING "Initial C release with debug info compile flags")
set(CMAKE_CXX_FLAGS_INIT           "" CACHE STRING "Initial C++ compile flags")
set(CMAKE_CXX_FLAGS_DEBUG          "-g -Wall -ftemplate-backtrace-limit=0" CACHE STRING "Initial C++ debug compile flags")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -march=native -DNDEBUG" CACHE STRING "Initial C++ minimum size release compile flags")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -march=native -DNDEBUG" CACHE STRING "Initial C++ release compile flags")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Og -g -march=native -Wall -ftemplate-backtrace-limit=0" CACHE STRING "Initial C++ release with debug info compile flags")

set(LAPACK_LIBRARIES "-Wl,--start-group" "${MKL_ROOT_DIR}/lib/intel64/libmkl_intel_lp64.a" 
    "${MKL_ROOT_DIR}/lib/intel64/libmkl_core.a" "${MKL_ROOT_DIR}/lib/intel64/libmkl_sequential.a" "-Wl,--end-group"
    "-lpthread" "-lm" "-ldl" CACHE STRING "BLAS linker flags")
set(LAPACK_INCLUDE_DIRS ${MKL_ROOT_DIR}/include CACHE STRING "LAPACK include directories")
set(LAPACK_COMPILE_DEFINITIONS BTAS_HAS_CBLAS=1;BTAS_HAS_LAPACKE=1;BTAS_HAS_INTEL_MKL=1;MADNESS_LINALG_USE_LAPACKE CACHE STRING "LAPACK preprocessor definitions")
set(INTEGER4 TRUE CACHE BOOL "Set Fortran integer size to 4 bytes")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
