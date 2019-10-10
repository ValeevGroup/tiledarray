include(SanitizeCUDAImplicitDirectories)

set(CUDA_FOUND TRUE)

set(TILEDARRAY_HAS_CUDA 1)

if(ENABLE_CUDA_ERROR_CHECK)
  set (TILEDARRAY_CHECK_CUDA_ERROR 1)
endif(ENABLE_CUDA_ERROR_CHECK)

enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

# despite enable_language(CUDA), this is still needed to find cuBLAS
find_package(CUDA REQUIRED)

message(STATUS "CUDA Include Dirs: " ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
message(STATUS "CUDA Host Compiler: " ${CMAKE_CUDA_HOST_COMPILER})
message(STATUS "CUDA NVCC FLAGS: ${CUDA_NVCC_FLAGS}")
message(STATUS "cuBLAS Libraries: : ${CUDA_CUBLAS_LIBRARIES}")

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


