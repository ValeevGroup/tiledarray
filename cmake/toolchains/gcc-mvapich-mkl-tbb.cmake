#
# Generic Toolchain for GCC + MVAPICH + MKL + TBB
#
# REQUIREMENTS:
# - in PATH:
#   gcc, g++, mpicc, and mpicxx
# - environment variables:
#   * INTEL_DIR: the Intel compiler directory (includes MKL and TBB), e.g. /opt/intel
#   * EIGEN3_DIR or (deprecated) EIGEN_DIR: the Eigen3 directory
#   * BOOST_DIR: the Boost root directory
#

# Set paths (add a cache entry for the paths used by the configure script)
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
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(MPI_C_COMPILER mpicc)
set(MPI_CXX_COMPILER mpicxx)

# Set compile flags
set(CMAKE_C_FLAGS_INIT             "-std=c99" CACHE STRING "Inital C compile flags")
set(CMAKE_C_FLAGS_DEBUG            "-g -Wall" CACHE STRING "Inital C debug compile flags")
set(CMAKE_C_FLAGS_MINSIZEREL       "-Os -march=native -DNDEBUG" CACHE STRING "Inital C minimum size release compile flags")
set(CMAKE_C_FLAGS_RELEASE          "-O3 -march=native -DNDEBUG" CACHE STRING "Inital C release compile flags")
set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g -Wall" CACHE STRING "Inital C release with debug info compile flags")
set(CMAKE_CXX_FLAGS_INIT           "" CACHE STRING "Inital C++ compile flags")
set(CMAKE_CXX_FLAGS_DEBUG          "-g -Wall" CACHE STRING "Inital C++ debug compile flags")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -march=native -DNDEBUG" CACHE STRING "Inital C++ minimum size release compile flags")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -march=native -DNDEBUG" CACHE STRING "Inital C++ release compile flags")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -Wall" CACHE STRING "Inital C++ release with debug info compile flags")

set(LAPACK_LIBRARIES "-Wl,--start-group" "${MKL_ROOT_DIR}/lib/intel64/libmkl_intel_lp64.a" "${MKL_ROOT_DIR}/lib/intel64/libmkl_core.a" "${MKL_ROOT_DIR}/lib/intel64/libmkl_sequential.a" "-Wl,--end-group" "-lm" "-ldl" CACHE STRING "BLAS linker flags")
set(LAPACK_INCLUDE_DIRS ${MKL_ROOT_DIR}/include CACHE STRING "LAPACK include directories")
set(LAPACK_COMPILE_DEFINITIONS BTAS_HAS_CBLAS=1;MADNESS_LINALG_USE_LAPACKE CACHE STRING "LAPACK preprocessor definitions")
set(INTEGER4 TRUE CACHE BOOL "Set Fortran integer size to 4 bytes")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
