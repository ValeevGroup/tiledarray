# Set compilers (assumes the compilers are in the PATH)
set(MPICHROOT "/usr/lib64/mpich")
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(MPI_C_COMPILER ${MPICHROOT}/bin/mpicc)
set(MPI_CXX_COMPILER ${MPICHROOT}/bin/mpicxx)

set(MKLROOT "/opt/intel/mkl")

# Set compile flags
set(CMAKE_C_FLAGS_INIT             "-std=c99" CACHE STRING "Inital C compile flags")
set(CMAKE_C_FLAGS_DEBUG            "-g -Wall" CACHE STRING "Inital C debug compile flags")
set(CMAKE_C_FLAGS_MINSIZEREL       "-Os -march=native -DNDEBUG" CACHE STRING "Inital C minimum size release compile flags")
set(CMAKE_C_FLAGS_RELEASE          "-O3 -march=native -DNDEBUG" CACHE STRING "Inital C release compile flags")
set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g -Wall" CACHE STRING "Inital C release with debug info compile flags")
set(CMAKE_CXX_FLAGS_INIT           "-m64 -I${MKLROOT}/include -I/opt/intel/tbb/include" CACHE STRING "Inital C++ compile flags")
set(CMAKE_CXX_FLAGS_DEBUG          "-g -Wall" CACHE STRING "Inital C++ debug compile flags")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -march=native -DNDEBUG" CACHE STRING "Inital C++ minimum size release compile flags")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -march=native -DNDEBUG" CACHE STRING "Inital C++ release compile flags")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -Wall" CACHE STRING "Inital C++ release with debug info compile flags")


set(BLAS_LIBRARIES "-Wl,--start-group ${MKLROOT}/lib/intel64/libmkl_intel_lp64.a ${MKLROOT}/lib/intel64/libmkl_core.a ${MKLROOT}/lib/intel64/libmkl_sequential.a -Wl,--end-group -lpthread -lm -ldl" CACHE STRING "BLAS linker flags")
set(LAPACK_LIBRARIES "${BLAS_LIBRARIES};-ldl" CACHE STRING "LAPACK linker flags")
set(LAPACK_INCLUDE_DIRS ${MKL_ROOT_DIR}/include CACHE STRING "LAPACK include directories")
set(LAPACK_COMPILE_DEFINITIONS HAVE_INTEL_MKL=1;_HAS_INTEL_MKL=1;BTAS_HAS_CBLAS=1 CACHE STRING "LAPACK preprocessor definitions")
set(INTEGER4 TRUE CACHE BOOL "Set Fortran integer size to 4 bytes")
set(TBB_INCLUDE_DIR "/opt/intel/tbb/include")
set(TBB_LIBRARY "/opt/intel/tbb/lib/intel64/gcc4.4")

