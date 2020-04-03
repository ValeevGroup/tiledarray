#
# Generic Toolchain using MKL + TBB on ALCF Theta
#
# REQUIREMENTS:
# - load PrgEnv-gnu or PrgEnv-intel to place Cray compiler wrappers (cc, CC) in PATH
# - load modules:
#   - module add boost
#   - module add cmake
# - set environment variables:
#   - export CRAYPE_LINK_TYPE=dynamic
#   - export INTEL_DIR=/theta-archive/intel/compilers_and_libraries_2019.5.281/linux
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
set(CMAKE_C_COMPILER cc)
set(CMAKE_CXX_COMPILER CC)
# setting these breaks FindMPI in cmake 3.9 and later
#set(MPI_C_COMPILER cc)
#set(MPI_CXX_COMPILER CC)
set(MPI_CXX_SKIP_MPICXX ON)

# Set compile flags
set(CMAKE_C_FLAGS_INIT             "-std=c99" CACHE STRING "Initial C compile flags")
set(CMAKE_C_FLAGS_DEBUG            "-g -Wall" CACHE STRING "Initial C debug compile flags")
set(CMAKE_C_FLAGS_MINSIZEREL       "-Os -march=native -DNDEBUG" CACHE STRING "Initial C minimum size release compile flags")
set(CMAKE_C_FLAGS_RELEASE          "-O3 -march=native -DNDEBUG" CACHE STRING "Initial C release compile flags")
set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g -Wall" CACHE STRING "Initial C release with debug info compile flags")
set(CMAKE_CXX_FLAGS_INIT           "-fno-omit-frame-pointer -fno-optimize-sibling-calls" CACHE STRING "Initial C++ compile flags")
set(CMAKE_CXX_FLAGS_DEBUG          "-g -Wall -ftemplate-backtrace-limit=0" CACHE STRING "Initial C++ debug compile flags")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -march=native -DNDEBUG" CACHE STRING "Initial C++ minimum size release compile flags")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -march=native -DNDEBUG" CACHE STRING "Initial C++ release compile flags")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-Og -g3 -Wall -ftemplate-backtrace-limit=0" CACHE STRING "Initial C++ release with debug info compile flags")

set(LAPACK_LIBRARIES "-Wl,--start-group" "${MKL_ROOT_DIR}/lib/intel64/libmkl_intel_lp64.a" "${MKL_ROOT_DIR}/lib/intel64/libmkl_core.a" "${MKL_ROOT_DIR}/lib/intel64/libmkl_sequential.a" "-Wl,--end-group" "-lpthread" "-lm" "-ldl" CACHE STRING "BLAS linker flags")
set(LAPACK_INCLUDE_DIRS ${MKL_ROOT_DIR}/include CACHE STRING "LAPACK include directories")
set(LAPACK_COMPILE_DEFINITIONS BTAS_HAS_CBLAS=1;MADNESS_LINALG_USE_LAPACKE CACHE STRING "LAPACK preprocessor definitions")
set(INTEGER4 TRUE CACHE BOOL "Set Fortran integer size to 4 bytes")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
