include(SanitizeCUDAImplicitDirectories)

set(CUDA_FOUND TRUE)

set(TILEDARRAY_HAS_CUDA 1 CACHE BOOL "Whether TiledArray has CUDA support")

if(ENABLE_CUDA_ERROR_CHECK)
  set (TILEDARRAY_CHECK_CUDA_ERROR 1)
endif(ENABLE_CUDA_ERROR_CHECK)

# TODO uncomment, and remove workaround, when 3.17.0 is released
cmake_minimum_required(VERSION 3.17.0) # decouples C++ and CUDA standards, see https://gitlab.kitware.com/cmake/cmake/issues/19123
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# despite enable_language(CUDA), this is still needed to find cuBLAS
find_package(CUDA REQUIRED)

message(STATUS "CUDA Host Compiler: ${CMAKE_CUDA_HOST_COMPILER}")
message(STATUS "CUDA NVCC FLAGS: ${CUDA_NVCC_FLAGS}")
message(STATUS "CUDA Include Dirs: ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}")
message(STATUS "CUDA Libraries: ${CUDA_LIBRARIES}")
message(STATUS "cuBLAS Libraries: : ${CUDA_CUBLAS_LIBRARIES}")

# CUDA_nvToolsExt_LIBRARY found by CMake 3.16 and later
if (CMAKE_VERSION VERSION_LESS 3.16.0)
  get_filename_component(CUDA_LIBRARY_DIR ${CUDA_cudart_static_LIBRARY} DIRECTORY)
  find_library(CUDA_nvToolsExt_LIBRARY nvToolsExt HINTS ${CUDA_LIBRARY_DIR})
endif()
message(STATUS "NVTX library : ${CUDA_nvToolsExt_LIBRARY}")

# sanitize implicit dirs if CUDA host compiler != C++ compiler
message(STATUS "CMAKE Implicit Include Directories: ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES}")
message(STATUS "CMAKE Implicit Link Directories: ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")
sanitize_cuda_implicit_directories()
message(STATUS "CMAKE Implicit Include Directories: ${CMAKE_CUDA_IMPLICIT_INCLUDE_DIRECTORIES}")
message(STATUS "CMAKE Implicit Link Directories: ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES}")

##
## Umpire
##
include(external/umpire.cmake)

##
## cuTT
##
include(external/cutt.cmake)
