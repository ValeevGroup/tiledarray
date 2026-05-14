# try find_package
if (NOT TARGET range-v3::range-v3)
  include (FindPackageRegimport)
  find_package_regimport(range-v3 QUIET CONFIG)
  if (TARGET range-v3::range-v3)
    message(STATUS "Found Range-V3 CONFIG at ${range-v3_CONFIG}")
  endif (TARGET range-v3::range-v3)
endif (NOT TARGET range-v3::range-v3)

# if not found, build via FetchContent
if (NOT TARGET range-v3::range-v3)

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

endif(NOT TARGET range-v3::range-v3)

# postcond check
if (NOT TARGET range-v3::range-v3)
  message(FATAL_ERROR "FindOrFetchRangeV3 could not make range-v3::range-v3 target available")
endif(NOT TARGET range-v3::range-v3)

# Treat range-v3 headers as system: range-v3 is header-only with no
# ordering risk against TA's headers, and it self-triggers
# -Wdeprecated-declarations (e.g. ranges::compressed_tuple used inside
# compressed_pair.hpp). The blanket CMAKE_NO_SYSTEM_FROM_IMPORTED=TRUE
# at the top-level avoids -isystem for general imported targets due to
# include-dir ordering; carve out range-v3 specifically here.
# range-v3::range-v3 is an ALIAS — resolve to the underlying target before
# touching properties.
get_target_property(_rv3_aliased range-v3::range-v3 ALIASED_TARGET)
if (NOT _rv3_aliased)
  set(_rv3_aliased range-v3::range-v3)
endif()
get_target_property(_rv3_inc ${_rv3_aliased} INTERFACE_INCLUDE_DIRECTORIES)
if (_rv3_inc)
  set_target_properties(${_rv3_aliased} PROPERTIES
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_rv3_inc}")
endif()
unset(_rv3_inc)
unset(_rv3_aliased)
