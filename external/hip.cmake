# cmake 3.21 introduced HIP language support
cmake_minimum_required(VERSION 3.21.0)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)
enable_language(HIP)

set(HIP_FOUND TRUE)
set(TILEDARRAY_HAS_HIP 1 CACHE BOOL "Whether TiledArray has HIP support")
set(TILEDARRAY_CHECK_HIP_ERROR 1 CACHE BOOL "Whether TiledArray will check HIP errors")

# find HIP components
find_package(hipblas REQUIRED)
find_package(rocprim REQUIRED)  # for rocthrust, per https://github.com/ROCmSoftwarePlatform/rocThrust#using-rocthrust-in-a-project
find_package(rocthrust REQUIRED)

foreach (library hipblas;rocthrust)
  if (NOT TARGET roc::${library})
    message(FATAL_ERROR "roc::${library} not found")
  endif()
endforeach()

##
## Umpire
##
include(external/umpire.cmake)

##
## LibreTT
##
include(external/librett.cmake)
