# try find_package
if (NOT TARGET umpire-cxx-allocator)
  include (FindPackageRegimport)
  find_package_regimport(umpire-cxx-allocator QUIET CONFIG)
  if (TARGET umpire-cxx-allocator)
    message(STATUS "Found umpire-cxx-allocator CONFIG at ${umpire-cxx-allocator_CONFIG}")
  endif (TARGET umpire-cxx-allocator)
endif (NOT TARGET umpire-cxx-allocator)

# if not found, build via FetchContent
if (NOT TARGET umpire-cxx-allocator)

  if (TA_CUDA)
    set(UMPIRE_ENABLE_CUDA ON CACHE BOOL "Enable CUDA support in Umpire")
  endif()
  if (TA_HIP)
    set(UMPIRE_ENABLE_HIP ON CACHE BOOL "Enable HIP support in Umpire")
  endif()

  include(FetchContent)
  FetchContent_Declare(
      umpire-cxx-allocator
      GIT_REPOSITORY      https://github.com/ValeevGroup/umpire-cxx-allocator.git
      GIT_TAG             ${TA_TRACKED_UMPIRE-CXX-ALLOCATOR_TAG}
  )
  FetchContent_MakeAvailable(umpire-cxx-allocator)
  FetchContent_GetProperties(umpire-cxx-allocator
      SOURCE_DIR UMPIRE-CXX-ALLOCATOR_SOURCE_DIR
      BINARY_DIR UMPIRE-CXX-ALLOCATOR_BINARY_DIR
      )

  # set umpire-cxx-allocator_CONFIG to the install location so that we know where to find it
  set(umpire-cxx-allocator_CONFIG ${CMAKE_INSTALL_PREFIX}/${UMPIRE-CXX-ALLOCATOR_CMAKE_DIR}/umpire-cxx-allocator-config.cmake)

endif(NOT TARGET umpire-cxx-allocator)

# postcond check
if (NOT TARGET umpire-cxx-allocator)
  message(FATAL_ERROR "FindOrFetchUmpireCXXAllocator could not make umpire-cxx-allocator target available")
endif(NOT TARGET umpire-cxx-allocator)
