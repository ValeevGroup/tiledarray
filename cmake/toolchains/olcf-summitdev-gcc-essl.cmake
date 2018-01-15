# load the following modules: gcc, cmake, mercurial, essl, boost, netlib-lapack

# Set environment paths
# N.B. since loading gcc purges xl must specify XLF root manually
#set(OLCF_XLF_ROOT $ENV{OLCF_XLF_ROOT})
set(OLCF_XLF_ROOT /sw/summitdev/xl/20170914-beta/xlf/15.1.6)
set(OLCF_ESSL_ROOT $ENV{OLCF_ESSL_ROOT})

# Set compilers
set(CMAKE_C_COMPILER       "gcc")
set(CMAKE_CXX_COMPILER     "g++")
set(CMAKE_Fortran_COMPILER "gfortran")
set(MPI_C_COMPILER         "mpicc")
set(MPI_CXX_COMPILER       "mpicxx")

# Set compile flags
set(CMAKE_C_FLAGS_INIT             "-std=c99" CACHE STRING "Inital C compile flags")
set(CMAKE_C_FLAGS_DEBUG            "-g -Wall" CACHE STRING "Inital C debug compile flags")
set(CMAKE_C_FLAGS_MINSIZEREL       "-Os -DNDEBUG" CACHE STRING "Inital C minimum size release compile flags")
set(CMAKE_C_FLAGS_RELEASE          "-O3 -DNDEBUG" CACHE STRING "Inital C release compile flags")
set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g -Wall" CACHE STRING "Inital C release with debug info compile flags")
set(CMAKE_CXX_FLAGS_INIT           "" CACHE STRING "Inital C++ compile flags")
set(CMAKE_CXX_FLAGS_DEBUG          "-g -Wall" CACHE STRING "Inital C++ debug compile flags")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG" CACHE STRING "Inital C++ minimum size release compile flags")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG" CACHE STRING "Inital C++ release compile flags")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -Wall" CACHE STRING "Inital C++ release with debug info compile flags")


# Set BLAS/LAPACK libraries
set(XLF_LIBRARIES ${OLCF_XLF_ROOT}/lib/libxlf90_r.a;${OLCF_XLF_ROOT}/lib/libxlfmath.a;-ldl;-lm)
set(BLAS_LIBRARIES ${OLCF_ESSL_ROOT}/lib64/libessl.so;${XLF_LIBRARIES})
set(LAPACK_LIBRARIES $ENV{OLCF_NETLIB_LAPACK_ROOT}/lib64/liblapack.a;${BLAS_LIBRARIES})
set(INTEGER4 TRUE CACHE BOOL "Set Fortran integer size to 4 bytes")

# Set build params
set(BUILD_SHARED_LIBS OFF)
