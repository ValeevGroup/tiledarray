set(CMAKE_SYSTEM_NAME Darwin)

# Set compilers (assumes the compilers are in the PATH)
set(CMAKE_C_COMPILER icc)
set(CMAKE_CXX_COMPILER icpc)
set(MPI_C_COMPILER mpicc)
set(MPI_CXX_COMPILER mpicxx)

# Set compile flags
set(CMAKE_C_FLAGS_INIT             "-std=c99 -use-clang-env" CACHE STRING "Inital C compile flags")
set(CMAKE_C_FLAGS_DEBUG            "-g -Wall -diag-disable remark,279,654,1125" CACHE STRING "Inital C debug compile flags")
set(CMAKE_C_FLAGS_MINSIZEREL       "-Os -march=native -DNDEBUG" CACHE STRING "Inital C minimum size release compile flags")
set(CMAKE_C_FLAGS_RELEASE          "-O3 -march=native -DNDEBUG" CACHE STRING "Inital C release compile flags")
set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g -Wall -diag-disable remark,279,654,1125" CACHE STRING "Inital C release with debug info compile flags")
set(CMAKE_CXX_FLAGS_INIT           "-std=c++11 -use-clang-env" CACHE STRING "Inital C++ compile flags")
set(CMAKE_CXX_FLAGS_DEBUG          "-g -Wall -diag-disable remark,279,654,1125" CACHE STRING "Inital C++ debug compile flags")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -march=native -DNDEBUG" CACHE STRING "Inital C++ minimum size release compile flags")
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -march=native -DNDEBUG" CACHE STRING "Inital C++ release compile flags")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -Wall -diag-disable remark,279,654,1125" CACHE STRING "Inital C++ release with debug info compile flags")

if(NOT ${CMAKE_SYSTEM_VERSION} VERSION_LESS 11.0)
  # Building on OS X 10.7 or later, so add "-Wl,-no_pie" linker flags.
  set(CMAKE_EXE_LINKER_FLAGS_INIT "-Wl,-no_pie" CACHE STRING "Initial executable linker flags")
endif()

# Set BLAS/LAPACK flags
set(BLAS_LINKER_FLAGS "-mkl" CACHE STRING "BLAS linker flags")
set(LAPACK_LINKER_FLAGS "-mkl" CACHE STRING "LAPACK linker flags")
set(INTEGER4 TRUE CACHE BOOL "Set Fortran integer size to 4 bytes")
