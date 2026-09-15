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

  if (TILEDARRAY_HAS_CUDA)
    # tell BLAS++/LAPACK++ to also look for CUDA
    set(gpu_backend cuda CACHE STRING "The device backend to use for Linalg++")
  elseif (TILEDARRAY_HAS_HIP)
    # tell BLAS++/LAPACK++ to also look for HIP
    set(gpu_backend hip CACHE STRING "The device backend to use for Linalg++")
  else ()
    # tell BLAS++/LAPACK++ to not look for device backends
    set(gpu_backend none CACHE STRING "The device backend to use for Linalg++")
  endif()

  # forward TA's assertion policy to BTAS, else BTAS picks its own default
  # (BTAS_ASSERT_THROW whenever BUILD_TESTING=ON, no matter what TA_ASSERT does).
  # BTAS_ASSERT_POLICY has the same three modes as TA_ASSERT_POLICY, and
  # like it is not affected by NDEBUG.
  # N.B. only if the user did not ask for a specific BTAS_ASSERT_POLICY, hence a
  #      parent project that sets BTAS_ASSERT_POLICY first keeps control;
  #      a value that this module derived on an earlier configure (recorded in
  #      TA_BTAS_ASSERT_POLICY_DERIVED) is re-derived, so that reconfiguring
  #      with a different TA_ASSERT_POLICY keeps BTAS in sync
  if (TA_ASSERT_POLICY STREQUAL TA_ASSERT_THROW)
    set(_ta_btas_assert_policy BTAS_ASSERT_THROW)
  elseif (TA_ASSERT_POLICY STREQUAL TA_ASSERT_ABORT)
    set(_ta_btas_assert_policy BTAS_ASSERT_ABORT)
  else ()
    set(_ta_btas_assert_policy BTAS_ASSERT_IGNORE)
  endif()
  if (NOT DEFINED BTAS_ASSERT_POLICY OR
      (DEFINED TA_BTAS_ASSERT_POLICY_DERIVED AND BTAS_ASSERT_POLICY STREQUAL TA_BTAS_ASSERT_POLICY_DERIVED))
    set(BTAS_ASSERT_POLICY ${_ta_btas_assert_policy} CACHE STRING "Controls the behavior of BTAS_ASSERT" FORCE)
    set(TA_BTAS_ASSERT_POLICY_DERIVED ${_ta_btas_assert_policy} CACHE INTERNAL "BTAS_ASSERT_POLICY derived from TA_ASSERT_POLICY by TiledArray")
  endif()
  unset(_ta_btas_assert_policy)

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
        message(WARNING "Linear algebra libraries' mangling convention not specified; specify -DLINALG_MANGLING={lower,lower_,UPPER}, if needed; BLASPP will try to autodetect")
      endif(NOT LINALG_MANGLING STREQUAL lower_)
      # these were needed for some configs at some point in the past? But in most cases they just produce compile noise
#      target_compile_definitions(blaspp PUBLIC -DBLAS_FORTRAN_ADD_=1)
#      target_compile_definitions(lapackpp PUBLIC -DLAPACK_FORTRAN_ADD_=1)
    endif()
  endif (_linalgpp_use_standard_linalg_kits)

endif(NOT TARGET BTAS::BTAS)

# postcond check
if (NOT TARGET BTAS::BTAS)
  message(FATAL_ERROR "FindOrFetchBTAS could not make BTAS::BTAS target available")
endif(NOT TARGET BTAS::BTAS)

# Treat BTAS headers as system: header-only library, no include-order
# risk against TA's headers, and BTAS upstream trips warnings TA itself
# can't fix (e.g. btas/generic/converge_class.h -Wreturn-type on gcc).
# Carved out specifically here despite the top-level
# CMAKE_NO_SYSTEM_FROM_IMPORTED=TRUE.
get_target_property(_btas_aliased BTAS::BTAS ALIASED_TARGET)
if (NOT _btas_aliased)
  set(_btas_aliased BTAS::BTAS)
endif()
get_target_property(_btas_inc ${_btas_aliased} INTERFACE_INCLUDE_DIRECTORIES)
if (_btas_inc)
  set_target_properties(${_btas_aliased} PROPERTIES
      INTERFACE_SYSTEM_INCLUDE_DIRECTORIES "${_btas_inc}")
endif()
unset(_btas_inc)
unset(_btas_aliased)
