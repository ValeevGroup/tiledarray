# first try find_package
if (NOT TARGET MADworld)
  include (FindPackageRegimport)
  find_package_regimport(MADNESS ${TA_TRACKED_MADNESS_VERSION} CONFIG QUIET COMPONENTS world HINTS "${MADNESS_ROOT_DIR}")
  if (TARGET MADworld)
    message(STATUS "Found MADNESS CONFIG at ${MADNESS_CONFIG}")
  endif (TARGET MADworld)
endif (NOT TARGET MADworld)

# if not found, build via FetchContent
if (NOT TARGET MADworld)

  # TA-specific configuration
  set(MADNESS_BUILD_MADWORLD_ONLY ON CACHE BOOL "Whether to build MADNESS runtime only")
  set(ENABLE_PARSEC OFF CACHE BOOL "Whether to use PaRSEC as the task backend of MADWorld")
  set(MPI_THREAD "multiple" CACHE INTERNAL "MADNESS requires MPI_THREAD_MULTIPLE")
  set(MADNESS_ASSUMES_ASLR_DISABLED ${TA_ASSUMES_ASLR_DISABLED} CACHE BOOL "Whether MADNESS assumes ASLR to be disabled")
  set(MPI_CXX_SKIP_MPICXX ON CACHE BOOL "Whether to disable search for C++ MPI-2 bindings")
  set(DISABLE_WORLD_GET_DEFAULT ON CACHE INTERNAL "Whether to disable madness::World::get_default()")
  set(ENABLE_MEM_PROFILE ON CACHE INTERNAL "Whether to enable instrumented memory profiling in MADNESS")
  set(ENABLE_TASK_DEBUG_TRACE ${TILEDARRAY_ENABLE_TASK_DEBUG_TRACE} CACHE INTERNAL "Whether to enable task profiling in MADNESS")

  # Set error handling method (for TA_ASSERT_POLICY allowed values see top-level CMakeLists.txt)
  if(TA_ASSERT_POLICY STREQUAL TA_ASSERT_IGNORE)
    set(_MAD_ASSERT_TYPE disable)
  elseif(TA_ASSERT_POLICY STREQUAL TA_ASSERT_THROW)
    set(_MAD_ASSERT_TYPE throw)
  elseif(TA_ASSERT_POLICY STREQUAL TA_ASSERT_ABORT)
    set(_MAD_ASSERT_TYPE abort)
  endif()
  set(MAD_ASSERT_TYPE ${_MAD_ASSERT_TYPE} CACHE INTERNAL "MADNESS assert type")
  set(ASSERTION_TYPE "${MAD_ASSERT_TYPE}" CACHE INTERNAL "MADNESS assert type")

  # look for C and MPI here to make troubleshooting easier and be able to override defaults for MADNESS
  enable_language(C)
  find_package(MPI REQUIRED COMPONENTS C CXX)

  include(FetchContent)
  FetchContent_Declare(
          MADNESS
              GIT_REPOSITORY https://github.com/m-a-d-n-e-s-s/madness.git
          GIT_TAG ${TA_TRACKED_MADNESS_TAG}
  )
  FetchContent_MakeAvailable(MADNESS)
  FetchContent_GetProperties(MADNESS
          SOURCE_DIR MADNESS_SOURCE_DIR
          BINARY_DIR MADNESS_BINARY_DIR
          )

  # set MADNESS_CONFIG to the install location so that we know where to find it
  set(MADNESS_CONFIG ${CMAKE_INSTALL_PREFIX}/${MADNESS_INSTALL_CMAKEDIR}/madness-config.cmake)

endif(NOT TARGET MADworld)

# postcond check
if (NOT TARGET MADworld)
  message(FATAL_ERROR "FindOrFetchMADNESS could not make MADworld target available")
endif(NOT TARGET MADworld)
