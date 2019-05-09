option(CUDA_PROPAGATE_HOST_FLAGS OFF)

find_package(CUDA REQUIRED)

message(STATUS "CUDA version:      ${CUDA_VERSION_STRING}")
message(STATUS "CUDA Include Path: ${CUDA_INCLUDE_DIRS}")
message(STATUS "CUDA Libraries:    ${CUDA_LIBRARIES}")
message(STATUS "cuBLAS Libraries:    ${CUDA_CUBLAS_LIBRARIES}")
set(CUDA_HOST_COMPILER ${CMAKE_CUDA_HOST_COMPILER})

if(CUDA_FOUND)
  set (TILEDARRAY_HAS_CUDA 1)
  if(ENABLE_CUDA_ERROR_CHECK)
    set (TILEDARRAY_CHECK_CUDA_ERROR 1)
  else(ENABLE_CUDA_ERROR_CHECK)
    set (TILEDARRAY_CHECK_CUDA_ERROR 0)
  endif(ENABLE_CUDA_ERROR_CHECK)
else(CUDA_FOUND)
  set (TILEDARRAY_HAS_CUDA 0)
  set (TILEDARRAY_CHECK_CUDA_ERROR 0)
endif(CUDA_FOUND)


if(CUDA_FOUND)

  set(CUDA_NVCC_FLAGS ${CMAKE_CUDA_FLAGS})

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

  ##
  ## Umpire
  ##
  include(external/umpire.cmake)

  ##
  ## cuTT
  ##
  include(external/cutt.cmake)


endif(CUDA_FOUND)
