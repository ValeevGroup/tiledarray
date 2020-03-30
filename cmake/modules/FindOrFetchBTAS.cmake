find_package(BTAS 1.0.0 QUIET)

if (NOT TARGET BTAS::btas)

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

endif(NOT TARGET BTAS::btas)
