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

# Set compilers (assumes the compilers are in the PATH)
set(CMAKE_C_COMPILER cc)
set(CMAKE_CXX_COMPILER CC)
# setting these breaks FindMPI in cmake 3.9 and later
#set(MPI_C_COMPILER cc)
#set(MPI_CXX_COMPILER CC)
set(MPI_CXX_SKIP_MPICXX ON)

####### Compile flags
include(${CMAKE_CURRENT_LIST_DIR}/_std_c_flags.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/_std_cxx_flags.cmake)

####### Eigen
include(${CMAKE_CURRENT_LIST_DIR}/_eigen.cmake)

####### Boost
include(${CMAKE_CURRENT_LIST_DIR}/_boost.cmake)

####### BLAS/LAPACK Libraries
set(INTEGER4 TRUE CACHE BOOL "Set Fortran integer size to 4 bytes")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
include(${CMAKE_CURRENT_LIST_DIR}/_tbb.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/_mkl.cmake)
