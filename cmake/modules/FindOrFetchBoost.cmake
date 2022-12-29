# Limit scope of the search if BOOST_ROOT or BOOST_INCLUDEDIR is provided.
if (BOOST_ROOT OR BOOST_INCLUDEDIR)
  set(Boost_NO_SYSTEM_PATHS TRUE)
endif()

# try find_package
if (NOT TARGET Boost::boost)
  include(FindPackageRegimport)
  find_package_regimport(Boost ${TA_TRACKED_BOOST_VERSION} QUIET)
  if (TARGET Boost::boost)
    message(STATUS "Found Boost ${Boost_VERSION}: ${Boost_INCLUDE_DIRS}")
  endif(TARGET Boost::boost)

  # Boost::* targets by default are not GLOBAL, so to allow users of TA to safely use them we need to make them global
  # more discussion here: https://gitlab.kitware.com/cmake/cmake/-/issues/17256
  foreach(tgt boost;headers;${Boost_BTAS_DEPS_LIBRARIES})
    if (TARGET Boost::${tgt})
      get_target_property(_boost_tgt_${tgt}_is_imported_global Boost::${tgt} IMPORTED_GLOBAL)
      if (NOT _boost_tgt_${tgt}_is_imported_global)
        set_target_properties(Boost::${tgt} PROPERTIES IMPORTED_GLOBAL TRUE)
      endif()
      unset(_boost_tgt_${tgt}_is_imported_global)
    endif()
  endforeach()

endif (NOT TARGET Boost::boost)

# if not found, build via FetchContent
if (NOT TARGET Boost::boost)
  include (FetchContent)
  cmake_minimum_required (VERSION 3.14.0)  # for FetchContent_MakeAvailable

  FetchContent_Declare(
          CMAKEBOOST
          GIT_REPOSITORY      https://github.com/Orphis/boost-cmake
  )
  FetchContent_MakeAvailable(CMAKEBOOST)
  FetchContent_GetProperties(CMAKEBOOST
          SOURCE_DIR CMAKEBOOST_SOURCE_DIR
          BINARY_DIR CMAKEBOOST_BINARY_DIR
          )

  # current boost-cmake/master does not install boost correctly, so warn that installed TiledArray will not be usable
  # boost-cmake/install_rules https://github.com/Orphis/boost-cmake/pull/45 is supposed to fix it but is inactive
  message(WARNING "Building Boost from source makes TiledArray unusable from the install location! Install Boost using package manager or manually and reconfigure/reinstall TiledArray to fix this")
  if (NOT TARGET Boost::headers)
    add_library(Boost::headers ALIAS Boost::boost)
  endif()
  foreach(_lib serialization regex locale locale_deps thread chrono)  # these are non-header-only components used by MPQC
    if (TARGET Boost_${_lib})
      install(TARGETS Boost_${_lib} EXPORT btas COMPONENT boost-libs)
      if (NOT TARGET Boost::${_lib})
        add_library(Boost::${_lib} ALIAS Boost_${_lib})
      endif()
    endif()
  endforeach()
#  export(EXPORT tiledarray
#      FILE "${PROJECT_BINARY_DIR}/boost-targets.cmake")
#  install(EXPORT tiledarray
#      FILE "boost-targets.cmake"
#      DESTINATION "${TILEDARRAY_INSTALL_CMAKEDIR}"
#      COMPONENT boost-libs)

endif(NOT TARGET Boost::boost)

# postcond check
if (NOT TARGET Boost::boost)
  message(FATAL_ERROR "FindOrFetchBoost could not make Boost::boost target available")
endif(NOT TARGET Boost::boost)
