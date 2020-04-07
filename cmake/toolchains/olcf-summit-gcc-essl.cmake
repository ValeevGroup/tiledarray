#
# N.B. load the following modules: gcc, cmake, essl, boost, netlib-lapack
#

set(CMAKE_SYSTEM_NAME Linux)

# Set environment paths
# N.B. since loading gcc purges xl must specify XLF root manually
#set(OLCF_XLF_ROOT $ENV{OLCF_XLF_ROOT})
set(OLCF_XLF_ROOT /sw/summit/xl/16.1.1-5/xlf/16.1.1)
set(OLCF_ESSL_ROOT $ENV{OLCF_ESSL_ROOT})

# Set compilers
set(CMAKE_C_COMPILER       "gcc")
set(CMAKE_CXX_COMPILER     "g++")
set(CMAKE_Fortran_COMPILER "gfortran")
set(MPI_C_COMPILER         "mpicc")
set(MPI_CXX_COMPILER       "mpicxx")

# Compile flags
include(${CMAKE_CURRENT_LIST_DIR}/_std_c_flags.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/_std_cxx_flags.cmake)

# Set BLAS/LAPACK libraries
set(XLF_LIBRARIES ${OLCF_XLF_ROOT}/lib/libxlf90_r.a;${OLCF_XLF_ROOT}/lib/libxlfmath.a;-ldl;-lm)
set(BLAS_LIBRARIES ${OLCF_ESSL_ROOT}/lib64/libessl.so;${XLF_LIBRARIES})
set(LAPACK_LIBRARIES $ENV{OLCF_NETLIB_LAPACK_ROOT}/lib64/liblapack.a;-lgfortran;${BLAS_LIBRARIES})
set(INTEGER4 TRUE CACHE BOOL "Set Fortran integer size to 4 bytes")

# Set build params
set(BUILD_SHARED_LIBS OFF)

# for some reason the NSF filesystem cannot handle RESTRICT checks , e.g.
# CMake Error: Remove failed on file: /autofs/nccs-svm1_home1/.....: System Error: Device or resource busy
set(MADNESS_CMAKE_EXTRA_ARGS "-DRESTRICT_KEYWORD=__restrict")
# same for MADNESS_HAS_MKL ... seems like CHECK_CXX_SOURCE_COMPILES is the culprit
set(MADNESS_HAS_MKL OFF)
