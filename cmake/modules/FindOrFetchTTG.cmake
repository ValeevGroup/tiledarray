if (NOT TARGET ttg-parsec)
  find_package(ttg CONFIG)
endif(NOT TARGET ttg-parsec)

if (TARGET ttg-parsec)
    message(STATUS "Found ttg CONFIG at ${ttg_CONFIG}")
else (TARGET ttg-parsec)

  include(FetchContent)
  FetchContent_Declare(
      ttg
      GIT_REPOSITORY      ${TA_TRACKED_TTG_URL}
      GIT_TAG             ${TA_TRACKED_TTG_TAG}
  )
  FetchContent_MakeAvailable(ttg)
  FetchContent_GetProperties(ttg
      SOURCE_DIR TTG_SOURCE_DIR
      BINARY_DIR TTG_BINARY_DIR
      )

endif(TARGET ttg-parsec)

# postcond check
if (NOT TARGET ttg-parsec)
  message(FATAL_ERROR "FindOrFetchTTG could not make ttg-parsec target available")
else()
  set(TILEDARRAY_HAS_TTG 1)
endif()
