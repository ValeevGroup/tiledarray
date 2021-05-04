find_package(range-v3 QUIET CONFIG)

if (TARGET range-v3::range-v3)
    message(STATUS "Found Range-V3 CONFIG at ${range-v3_CONFIG}")
else (TARGET range-v3::range-v3)

  include(FetchContent)
  FetchContent_Declare(
      RangeV3
      GIT_REPOSITORY      https://github.com/ericniebler/range-v3.git
      GIT_TAG             ${TA_TRACKED_RANGEV3_TAG}
  )
  FetchContent_MakeAvailable(RangeV3)
  FetchContent_GetProperties(RangeV3
      SOURCE_DIR RANGEV3_SOURCE_DIR
      BINARY_DIR RANGEV3_BINARY_DIR
      )

  # set range-v3_CONFIG to the install location so that we know where to find it
  set(range-v3_CONFIG ${CMAKE_INSTALL_PREFIX}/lib/cmake/range-v3/range-v3-config.cmake)

endif(TARGET range-v3::range-v3)

# postcond check
if (NOT TARGET range-v3::range-v3)
  message(FATAL_ERROR "FindOrFetchRangeV3 could not make range-v3::range-v3 target available")
endif(NOT TARGET range-v3::range-v3)
