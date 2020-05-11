find_package(BTAS 1.0.0 QUIET CONFIG)

if (NOT TARGET BTAS::BTAS)

  set(BTAS_ENABLE_MKL ${ENABLE_MKL} CACHE BOOL "Whether BTAS should seek MKL")
  if (MADNESS_FORTRAN_DEFAULT_INTEGER4)
    set(mkl_pref_ilp64 OFF)
  else(MADNESS_FORTRAN_DEFAULT_INTEGER4)
    set(mkl_pref_ilp64 ON)
  endif(MADNESS_FORTRAN_DEFAULT_INTEGER4)
  set(MKL_PREFER_ILP64 ${mkl_pref_ilp64} CACHE BOOL "MKL preference: ILP64 (yes) or {LP64,LP32} (no)")
  if (MADNESS_HAS_TBB)
    set(MKL_THREADING "TBB" CACHE STRING "MKL flavor: SEQ, TBB or OMP (default)")
  endif()

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
