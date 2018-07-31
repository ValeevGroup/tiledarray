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
include_directories(${CUDA_INCLUDE_DIRS})



##
## find cuTT
##

find_path(_CUTT_INSTALL_DIR NAMES include/cutt.h lib/libcutt.a HINTS ${CUTT_INSTALL_DIR})

if( _CUTT_INSTALL_DIR )

  # peform compile check of CUTT

  message(STATUS "cuTT found at ${_CUTT_INSTALL_DIR}")

  add_library(TiledArray_CUTT INTERFACE)

  set_target_properties(TiledArray_CUTT
          PROPERTIES
          INTERFACE_INCLUDE_DIRECTORIES
          ${_CUTT_INSTALL_DIR}/include
          INTERFACE_LINK_LIBRARIES
          ${_CUTT_INSTALL_DIR}/lib/libcutt.a
          )

  install(TARGETS TiledArray_CUTT EXPORT tiledarray COMPONENT tiledarray)

elseif(TA_EXPERT)

  message("** cuTT was not found")
  message(STATUS "** Downloading and building cuTT is explicitly disabled in EXPERT mode")

else()


  include(ExternalProject)

  # set source and build path for cuTT in the TiledArray project
  set(EXTERNAL_SOURCE_DIR   ${PROJECT_BINARY_DIR}/external/source/cutt)
  # cutt only supports in source build
  set(EXTERNAL_BUILD_DIR  ${PROJECT_BINARY_DIR}/external/source/cutt)
  if (NOT CUTT_URL)
    set(CUTT_URL https://github.com/ap-hynninen/cutt)
  endif (NOT CUTT_URL)
  set(CUTT_TAG master)

  message("** Will clone cuTT from ${CUTT_URL}")

  ExternalProject_Add(cutt
        PREFIX ${CMAKE_INSTALL_PREFIX}
        STAMP_DIR ${PROJECT_BINARY_DIR}/external/cutt-stamp
        TMP_DIR ${PROJECT_BINARY_DIR}/external/tmp
        #--Download step--------------
        DOWNLOAD_DIR ${EXTERNAL_SOURCE_DIR}
        GIT_REPOSITORY ${CUTT_URL}
        GIT_TAG ${CUTT_TAG}
        #--Configure step-------------
        SOURCE_DIR ${EXTERNAL_SOURCE_DIR}
        CONFIGURE_COMMAND ""
        #--Build step-----------------
        BINARY_DIR ${EXTERNAL_BUILD_DIR}
        BUILD_COMMAND make
        #--Install step---------------
        INSTALL_COMMAND ""
        #--Custom targets-------------
        STEP_TARGETS download
        )

  # Add cuTT dependency to External
  add_dependencies(External cutt)

  # create an exportable interface target for BTAS
  add_library(TiledArray_CUTT INTERFACE)

  set_property(TARGET TiledArray_CUTT PROPERTY
          INTERFACE_INCLUDE_DIRECTORIES
          $<BUILD_INTERFACE:${EXTERNAL_SOURCE_DIR}>
          $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}>
          )

  install(TARGETS TiledArray_CUTT EXPORT tiledarray COMPONENT tiledarray)

  # how to install BTAS
  install(
          DIRECTORY
          ${EXTERNAL_SOURCE_DIR}/cutt/include
          ${EXTERNAL_SOURCE_DIR}/cutt/lib
          ${EXTERNAL_SOURCE_DIR}/cutt/lib64
          DESTINATION
          ${CMAKE_INSTALL_INCLUDEDIR}
#          ${CMAKE_INSTALL_PREFIX}/lib
#          ${CMAKE_INSTALL_PREFIX}/lib
          COMPONENT cutt
  )

endif(_CUTT_INSTALL_DIR)


endif(CUDA_FOUND)