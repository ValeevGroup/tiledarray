find_package(CUDA)

message(STATUS "CUDA version:      ${CUDA_VERSION_STRING}")
message(STATUS "CUDA Include Path: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries:    ${CUDA_LIBRARIES}")
message(STATUS "cuBLAS Libraries:    ${CUDA_CUBLAS_LIBRARIES}")

if(CUDA_FOUND)
  set (TILEDARRAY_HAS_CUDA 1)
else(CUDA_FOUND)
  set (TILEDARRAY_HAS_CUDA 0)
endif(CUDA_FOUND)


if(CUDA_FOUND)

  # TODO test CUDA
  # make cuda interface library
  add_library(TiledArray_CUDA INTERFACE)

  set_target_properties(TiledArray_CUDA
          PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES
          ${CUDA_INCLUDE_DIRS}
          INTERFACE_LINK_LIBRARIES
          "${CUDA_LIBRARIES};${CUDA_CUBLAS_LIBRARIES}"
          )

  set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
  set(CUDA_CUBLAS_LIBRARIES PUBLIC ${CUDA_CUBLAS_LIBRARIES})

  install(TARGETS TiledArray_CUDA EXPORT tiledarray COMPONENT tiledarray)
#include_directories(${CUDA_INCLUDE_DIRS})

##
## cuTT
##
include(external/cutt.cmake)

##
## Umpire
##
include(external/umpire.cmake)

endif(CUDA_FOUND)