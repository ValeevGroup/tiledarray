find_package(BTAS 1.0.0 QUIET)

if (NOT TARGET BTAS::BTAS)

  FetchContent_Declare(
      BTAS
      GIT_REPOSITORY      https://github.com/BTAS/btas.git
      GIT_TAG             f75b770085c90588e6ed7c7f45b598a16663cf81
  )
  FetchContent_MakeAvailable(BTAS)
  FetchContent_GetProperties(BTAS
      SOURCE_DIR BTAS_SOURCE_DIR
      BINARY_DIR BTAS_BINARY_DIR
      )

  # use subproject targets as if they were in exported namespace ...
  if (TARGET BTAS)
    add_library(BTAS::BTAS ALIAS BTAS)
  endif()

endif(NOT TARGET BTAS::BTAS)

# postcond check
if (NOT TARGET BTAS::BTAS)
  message(FATAL_ERROR "FindOrFetchBTAS could not provide BTAS targets")
endif(NOT TARGET BTAS::BTAS)
