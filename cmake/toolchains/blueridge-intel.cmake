set(CMAKE_SYSTEM_NAME Linux)

# Set paths (add a cache entry for the paths used by the configure script)
set(GCC_ROOT_DIR "/opt/apps/gcc/4.9.2")
set(MKL_ROOT_DIR "$ENV{INTEL_DIR}/mkl")
set(TBB_ROOT_DIR "$ENV{INTEL_DIR}/tbb" CACHE PATH "TBB root directory")
set(BOOST_ROOT "$ENV{BOOST_DIR}" CACHE PATH "Boost root directory")

# Set compilers (assumes the compilers are in the PATH)
set(CMAKE_C_COMPILER icc)
set(CMAKE_CXX_COMPILER icpc)
set(MPI_C_COMPILER mpiicc)
set(MPI_CXX_COMPILER mpiicpc)

# Set compile flags
set(CMAKE_C_FLAGS_INIT             "-std=c99 -gcc-name=${GCC_ROOT_DIR}/bin/gcc" CACHE STRING "Inital C compile flags")
set(CMAKE_C_FLAGS_DEBUG            "-g -Wall" CACHE STRING "Inital C debug compile flags")
set(CMAKE_C_FLAGS_MINSIZEREL       "-Os -DNDEBUG" CACHE STRING "Inital C minimum size release compile flags")
set(CMAKE_C_FLAGS_RELEASE          "-O3 -DNDEBUG" CACHE STRING "Inital C release compile flags")
set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g -Wall" CACHE STRING "Inital C release with debug info compile flags")
set(CMAKE_CXX_FLAGS_INIT           "-std=c++11 -gxx-name=${GCC_ROOT_DIR}/bin/g++" CACHE STRING "Inital C++ compile flags")
set(CMAKE_CXX_FLAGS_DEBUG          "-g -Wall" CACHE STRING "Inital C++ debug compile flags")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -DNDEBUG" CACHE STRING "Inital C++ minimum size release compile flags")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -DNDEBUG" CACHE STRING "Inital C++ release compile flags")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -Wall" CACHE STRING "Inital C++ release with debug info compile flags")

set(LAPACK_LIBRARIES "-Wl,--start-group" "${MKL_ROOT_DIR}/lib/intel64/libmkl_intel_lp64.a" 
    "${MKL_ROOT_DIR}/lib/intel64/libmkl_core.a" "${MKL_ROOT_DIR}/lib/intel64/libmkl_sequential.a" 
    "-Wl,--end-group" "-lm" "-ldl" CACHE STRING "BLAS linker flags")
set(INTEGER4 TRUE CACHE BOOL "Set Fortran integer size to 4 bytes")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
