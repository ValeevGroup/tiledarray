set(CMAKE_SYSTEM_NAME Linux)

# Set compilers (assumes the compilers are in the PATH)
set(CMAKE_C_COMPILER gcc)
set(CMAKE_CXX_COMPILER g++)
set(MPI_C_COMPILER mpicc)
set(MPI_CXX_COMPILER mpicxx)

# Set compile flags
set(CMAKE_C_FLAGS_INIT             "-std=c99 -m64" CACHE STRING "Inital C compile flags")
set(CMAKE_C_FLAGS_DEBUG            "-g -Wall" CACHE STRING "Inital C debug compile flags")
set(CMAKE_C_FLAGS_MINSIZEREL       "-Os -march=native -DNDEBUG" CACHE STRING "Inital C minimum size release compile flags")
set(CMAKE_C_FLAGS_RELEASE          "-O3 -march=native -DNDEBUG" CACHE STRING "Inital C release compile flags")
set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g -Wall" CACHE STRING "Inital C release with debug info compile flags")
set(CMAKE_CXX_FLAGS_INIT           "-std=c++11 -m64" CACHE STRING "Inital C++ compile flags")
set(CMAKE_CXX_FLAGS_DEBUG          "-g -Wall" CACHE STRING "Inital C++ debug compile flags")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -march=native -DNDEBUG" CACHE STRING "Inital C++ minimum size release compile flags")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -march=native -DNDEBUG" CACHE STRING "Inital C++ release compile flags")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -Wall" CACHE STRING "Inital C++ release with debug info compile flags")

# Set BLAS/LAPACK flags
set(BLAS_LIBRARIES "-Wl,--start-group;$ENV{MKLROOT}/lib/intel64/libmkl_intel_ilp64.a;$ENV{MKLROOT}/lib/intel64/libmkl_core.a;$ENV{MKLROOT}/lib/intel64/libmkl_sequential.a;-Wl,--end-group;-lpthread;-lm" CACHE STRING "BLAS linker flags")
set(LAPACK_LIBRARIES "-Wl,--start-group;$ENV{MKLROOT}/lib/intel64/libmkl_intel_ilp64.a;$ENV{MKLROOT}/lib/intel64/libmkl_core.a;$ENV{MKLROOT}/lib/intel64/libmkl_sequential.a;-Wl,--end-group;-lpthread;-lm" CACHE STRING "LAPACK linker flags")
set(INTEGER4 TRUE CACHE BOOL "Set Fortran integer size to 4 bytes")
