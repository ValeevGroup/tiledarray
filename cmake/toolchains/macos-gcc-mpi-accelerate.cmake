####### Compilers
include(${CMAKE_CURRENT_LIST_DIR}/_gcc.cmake)
set(MPI_C_COMPILER mpicc)
set(MPI_CXX_COMPILER mpicxx)
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
include(${CMAKE_CURRENT_LIST_DIR}/_accelerate.cmake)

####### Platform-specific bits
