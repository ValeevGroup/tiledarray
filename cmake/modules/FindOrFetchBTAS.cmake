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

  # BTAS will load BLAS++/LAPACK++ ... if those use CMake's FindBLAS/FindLAPACK (as indicated by defined BLA_VENDOR)
  # will need to specify Fortran linkage convention ... manually for now, switching to NWX's linear algebra discovery
  # is necessary to handle all the corner cases for automatic discovery
  if (DEFINED BLA_VENDOR)
    set(_linalgpp_use_standard_linalg_kits TRUE)
  endif(DEFINED BLA_VENDOR)

  if (NOT TILEDARRAY_HAS_CUDA)
    # tell BLAS++/LAPACK++ to ignore CUDA
    set(gpu_backend none CACHE STRING "The device backend to use for Linalg++")
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

  # define macros specifying Fortran mangling convention, if necessary
  if (_linalgpp_use_standard_linalg_kits)
    if (NOT TARGET blaspp AND NOT TARGET lapackpp)
      message(FATAL_ERROR "blaspp or lapackpp targets missing")
    endif(NOT TARGET blaspp AND NOT TARGET lapackpp)
    if (LINALG_MANGLING STREQUAL lower)
      target_compile_definitions(blaspp PUBLIC -DBLAS_FORTRAN_LOWER=1)
      target_compile_definitions(lapackpp PUBLIC -DLAPACK_FORTRAN_LOWER=1)
    elseif(LINALG_MANGLING STREQUAL UPPER OR LINALG_MANGLING STREQUAL upper)
      target_compile_definitions(blaspp PUBLIC -DBLAS_FORTRAN_UPPER=1)
      target_compile_definitions(lapackpp PUBLIC -DLAPACK_FORTRAN_UPPER=1)
    else()
      if (NOT LINALG_MANGLING STREQUAL lower_)
        message(WARNING "Linear algebra libraries' mangling convention not specified; specify -DLINALG_MANGLING={lower,lower_,UPPER}, if needed; will assume lower_")
      endif(NOT LINALG_MANGLING STREQUAL lower_)
      target_compile_definitions(blaspp PUBLIC -DBLAS_FORTRAN_ADD_=1)
      target_compile_definitions(lapackpp PUBLIC -DLAPACK_FORTRAN_ADD_=1)
    endif()
  endif (_linalgpp_use_standard_linalg_kits)

endif(NOT TARGET BTAS::BTAS)

# postcond check
if (NOT TARGET BTAS::BTAS)
  message(FATAL_ERROR "FindOrFetchBTAS could not make BTAS::BTAS target available")
endif(NOT TARGET BTAS::BTAS)
