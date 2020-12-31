
# -*- mode: cmake -*-

###################
# Find dependencies related to ScaLAPACK
###################

if( NOT TARGET scalapackpp::scalapackpp )
  find_package( scalapackpp CONFIG QUIET HINTS ${SCALAPACKPP_ROOT_DIR} )
endif()

if( TARGET scalapackpp::scalapackpp )

  message( STATUS "Found scalapackpp!" )

else()

  message(STATUS "Could not find scalapackpp! Building..." )

  include( DownloadProject )
  download_project(
    PROJ                blacspp
    GIT_REPOSITORY      https://github.com/wavefunction91/blacspp.git
    GIT_TAG             ${TA_TRACKED_BLACSPP_TAG}
    PREFIX              ${PROJECT_BINARY_DIR}/external
    UPDATE_DISCONNECTED 1
  )
  download_project(
    PROJ                scalapackpp
    GIT_REPOSITORY      https://github.com/wavefunction91/scalapackpp.git
    GIT_TAG             ${TA_TRACKED_SCALAPACKPP_TAG}
    PREFIX              ${PROJECT_BINARY_DIR}/external
    UPDATE_DISCONNECTED 1
  )

  if( DEFINED SCALAPACK_LIBRARIES )
    set( scalapack_LIBRARIES ${SCALAPACK_LIBRARIES} )
    set( blacs_LIBRARIES     ${SCALAPACK_LIBRARIES} )
  endif()

  set( BLACSPP_ENABLE_TESTS OFF )
  set( SCALAPACKPP_ENABLE_TESTS OFF )
  add_subdirectory( ${blacspp_SOURCE_DIR} ${blacspp_BINARY_DIR} )
  add_subdirectory( ${scalapackpp_SOURCE_DIR} ${scalapackpp_BINARY_DIR} )

  # propagate MPI_CXX_SKIP_MPICXX=ON
  target_compile_definitions( blacspp PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS} )
  target_compile_definitions( scalapackpp PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS} )

  install( TARGETS blacspp scalapackpp EXPORT tiledarray COMPONENT tiledarray )
  # Add these dependencies to External
  add_dependencies(External-tiledarray scalapackpp blacspp)

  # set {blacspp,scalapackpp}_CONFIG to the install location so that we know where to find it
  set(blacspp_CONFIG ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/blacspp/blacspp-config.cmake)
  set(scalapackpp_CONFIG ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/scalapackpp/scalapackpp-config.cmake)

endif()

add_library( TiledArray_SCALAPACK INTERFACE )
target_link_libraries( TiledArray_SCALAPACK INTERFACE scalapackpp::scalapackpp blacspp::blacspp)

install( TARGETS TiledArray_SCALAPACK EXPORT tiledarray COMPONENT tiledarray )

set( TILEDARRAY_HAS_SCALAPACK 1 )
