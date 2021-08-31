# try find_package
if (NOT TARGET umpire)
  include (FindPackageRegimport)
  find_package_regimport(umpire QUIET CONFIG)
  if (TARGET umpire)
    message(STATUS "Found umpire CONFIG at ${umpire_CONFIG}")
  endif (TARGET umpire)
endif (NOT TARGET umpire)

# if not found, build via FetchContent
if (NOT TARGET umpire)

  include(FetchContent)
  FetchContent_Declare(
      umpire
      GIT_REPOSITORY      https://github.com/LLNL/Umpire.git
      GIT_TAG             ${TA_TRACKED_UMPIRE_TAG}
  )
  FetchContent_MakeAvailable(umpire)
  FetchContent_GetProperties(umpire
      SOURCE_DIR UMPIRE_SOURCE_DIR
      BINARY_DIR UMPIRE_BINARY_DIR
      )

  # set BTAS_CONFIG to the install location so that we know where to find it
  set(umpire_CONFIG ${CMAKE_INSTALL_PREFIX}/${UMPIRE_CMAKE_DIR}/umpire-config.cmake)

  # install umpire
  install(TARGETS umpire umpire_alloc EXPORT tiledarray COMPONENT tiledarray )

endif(NOT TARGET umpire)

# postcond check
if (NOT TARGET umpire)
  message(FATAL_ERROR "FindOrFetchUmpire could not make umpire target available")
endif(NOT TARGET umpire)
