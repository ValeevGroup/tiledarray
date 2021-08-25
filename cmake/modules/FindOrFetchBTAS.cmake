# try find_package
if (NOT TARGET BTAS::BTAS)
  include (FindPackageRegimport)
  find_package_regimport(BTAS 1.0.0 QUIET CONFIG)
  if (TARGET BTAS::BTAS)
    message(STATUS "Found BTAS CONFIG at ${BTAS_CONFIG}")
  endif (TARGET BTAS::BTAS)
endif (NOT TARGET BTAS::BTAS)

# if not found, build via FetchContent
if (NOT TARGET BTAS::BTAS)

  if (NOT TILEDARRAY_HAS_CUDA)
    # tell BLAS++/LAPACK++ to ignore CUDA
    set(use_cuda FALSE CACHE BOOL "Whether to look for CUDA-enabled libs in BLAS++/LAPACK++")
  endif()

  include(FetchContent)
  FetchContent_Declare(
      BTAS
      GIT_REPOSITORY      https://github.com/BTAS/btas.git
      GIT_TAG             ${TA_TRACKED_BTAS_TAG}
  )
  FetchContent_MakeAvailable(BTAS)
  FetchContent_GetProperties(BTAS
      SOURCE_DIR BTAS_SOURCE_DIR
      BINARY_DIR BTAS_BINARY_DIR
      )

  # use subproject targets as if they were in exported namespace ...
  if (TARGET BTAS AND NOT TARGET BTAS::BTAS)
    add_library(BTAS::BTAS ALIAS BTAS)
  endif(TARGET BTAS AND NOT TARGET BTAS::BTAS)

  # set BTAS_CONFIG to the install location so that we know where to find it
  set(BTAS_CONFIG ${CMAKE_INSTALL_PREFIX}/${BTAS_INSTALL_CMAKEDIR}/btas-config.cmake)

endif(NOT TARGET BTAS::BTAS)

# postcond check
if (NOT TARGET BTAS::BTAS)
  message(FATAL_ERROR "FindOrFetchBTAS could not make BTAS::BTAS target available")
endif(NOT TARGET BTAS::BTAS)
