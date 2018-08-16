find_package(CUDA)


if(CUDA_FOUND)
  set (TILEDARRAY_HAS_CUDA 1)
else(CUDA_FOUND)
  set (TILEDARRAY_HAS_CUDA 0)
endif(CUDA_FOUND)


if(CUDA_FOUND)

  # TODO test CUDA
  set(CUDA_LIBRARIES PUBLIC ${CUDA_LIBRARIES})
  set(CUDA_CUBLAS_LIBRARIES PUBLIC ${CUDA_CUBLAS_LIBRARIES})
  # make cuda interface library
  add_library(TiledArray_CUDA INTERFACE)

  set_target_properties(TiledArray_CUDA
          PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES
          ${CUDA_INCLUDE_DIRS}
          INTERFACE_LINK_LIBRARIES
          "${CUDA_LIBRARIES} ${CUDA_CUBLAS_LIBRARIES}"
          )
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