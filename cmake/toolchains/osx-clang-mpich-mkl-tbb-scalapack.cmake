#set(CMAKE_SYSTEM_NAME Darwin)

# Set compilers (assumes the compilers are in the PATH)
set(CMAKE_C_COMPILER clang)
set(CMAKE_CXX_COMPILER clang++)
set(MPI_C_COMPILER mpicc)
set(MPI_CXX_COMPILER mpicxx)
set(MPI_CXX_SKIP_MPICXX ON)

# Set compile flags
set(CMAKE_C_FLAGS_INIT             "-std=c99  -m64 -I${MKLROOT}/include" CACHE STRING "Initial C compile flags")
set(CMAKE_C_FLAGS_DEBUG            "-g -Wall" CACHE STRING "Initial C debug compile flags")
set(CMAKE_C_FLAGS_MINSIZEREL       "-Os -march=native -DNDEBUG" CACHE STRING "Initial C minimum size release compile flags")
set(CMAKE_C_FLAGS_RELEASE          "-O3 -march=native -DNDEBUG" CACHE STRING "Initial C release compile flags")
set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g -Wall" CACHE STRING "Initial C release with debug info compile flags")
set(CMAKE_CXX_FLAGS_INIT           "" CACHE STRING "Initial C++ compile flags")
set(CMAKE_CXX_FLAGS_DEBUG          "-g -Wall" CACHE STRING "Initial C++ debug compile flags")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -march=native -DNDEBUG" CACHE STRING "Initial C++ minimum size release compile flags")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -march=native -DNDEBUG" CACHE STRING "Initial C++ release compile flags")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -Wall" CACHE STRING "Initial C++ release with debug info compile flags")

# Libraries
if(EXISTS $ENV{MKLROOT})
  set(MKL_ROOT_DIR "$ENV{MKLROOT}" CACHE PATH "MKL root directory")
else()
  set(MKL_ROOT_DIR "/opt/intel/mkl" CACHE PATH "MKL root directory")
endif()
if(EXISTS $ENV{TBBROOT})
  set(TBB_ROOT_DIR "$ENV{TBBROOT}" CACHE PATH "TBB root directory")
else()
  set(TBB_ROOT_DIR "/opt/intel/tbb" CACHE PATH "TBB root directory")
endif()

# configure BLAS+LAPACK for MADNESS
set(BLAS_LINKER_FLAGS "${MKL_ROOT_DIR}/lib/libmkl_intel_lp64.a" "${MKL_ROOT_DIR}/lib/libmkl_sequential.a" "${MKL_ROOT_DIR}/lib/libmkl_core.a" "-lpthread" "-lm" "-ldl" CACHE STRING "BLAS linker flags")
set(LAPACK_LIBRARIES ${BLAS_LINKER_FLAGS} CACHE STRING "LAPACK linker flags")
set(LAPACK_INCLUDE_DIRS ${MKL_ROOT_DIR}/include CACHE STRING "LAPACK include directories")
set(LAPACK_COMPILE_DEFINITIONS BTAS_HAS_CBLAS=1;BTAS_HAS_LAPACKE=1;BTAS_HAS_INTEL_MKL=1;MADNESS_LINALG_USE_LAPACKE CACHE STRING "LAPACK preprocessor definitions")
set(INTEGER4 TRUE CACHE BOOL "Set Fortran integer size to 4 bytes")

# configure BLACS
set(blacs_LIBRARIES "${MKL_ROOT_DIR}/lib/libmkl_blacs_mpich_lp64.a" ${BLAS_LINKER_FLAGS})

# configure ScaLAPACK
set(ENABLE_SCALAPACK ON)
set(IntelMKL_LIBRARIES ${blacs_LIBRARIES})
set(scalapack_LIBRARIES "${MKL_ROOT_DIR}/lib/libmkl_scalapack_lp64.a" ${blacs_LIBRARIES})

set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
