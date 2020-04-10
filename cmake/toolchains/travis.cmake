#set(CMAKE_SYSTEM_NAME Linux)
# Set compile flags
set(CMAKE_C_FLAGS_INIT             "-std=c99  -m64 -I/usr/include" CACHE STRING "Inital C compile flags")
set(CMAKE_C_FLAGS_DEBUG            "-g -Wall" CACHE STRING "Inital C debug compile flags")
set(CMAKE_C_FLAGS_MINSIZEREL       "-Os -march=native -DNDEBUG" CACHE STRING "Inital C minimum size release compile flags")
set(CMAKE_C_FLAGS_RELEASE          "-O3 -march=native -DNDEBUG" CACHE STRING "Inital C release compile flags")
set(CMAKE_C_FLAGS_RELWITHDEBINFO   "-O2 -g -Wall" CACHE STRING "Inital C release with debug info compile flags")
set(CMAKE_CXX_FLAGS_INIT           "" CACHE STRING "Inital C++ compile flags")
set(CMAKE_CXX_FLAGS_DEBUG          "-g -Wall" CACHE STRING "Inital C++ debug compile flags")
set(CMAKE_CXX_FLAGS_MINSIZEREL     "-Os -march=native -DNDEBUG" CACHE STRING "Inital C++ minimum size release compile flags")
# clang issue with mismatched alloc/free in Eigen goes away if NDEBUG is not defined ... just a workaround
set(CMAKE_CXX_FLAGS_RELEASE        "-O3 -march=native" CACHE STRING "Inital C++ release compile flags")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -Wall" CACHE STRING "Inital C++ release with debug info compile flags")

# Libraries

set(BLAS_LINKER_FLAGS "-L/usr/lib/libblas" "-lblas" "-L/usr/lib/lapack" "-llapack" "-L/usr/lib" "-llapacke" CACHE STRING "BLAS linker flags")
set(LAPACK_LIBRARIES ${BLAS_LINKER_FLAGS} CACHE STRING "LAPACK linker flags")
set(LAPACK_INCLUDE_DIRS "/usr/include" CACHE STRING "LAPACK include directories")
set(LAPACK_COMPILE_DEFINITIONS MADNESS_LINALG_USE_LAPACKE CACHE STRING "LAPACK preprocessor definitions")
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Build shared libraries")
