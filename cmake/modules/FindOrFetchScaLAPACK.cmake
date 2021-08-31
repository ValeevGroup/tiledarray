
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
  include(FetchContent)

  FetchContent_Declare( scalapackpp
    GIT_REPOSITORY      https://github.com/wavefunction91/scalapackpp.git
    GIT_TAG             ${TA_TRACKED_SCALAPACKPP_TAG}
  )
  FetchContent_MakeAvailable( scalapackpp )

  # propagate MPI_CXX_SKIP_MPICXX=ON
  target_compile_definitions( blacspp     PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS} )
  target_compile_definitions( scalapackpp PRIVATE ${MPI_CXX_COMPILE_DEFINITIONS} )

  install( TARGETS blacspp scalapackpp EXPORT tiledarray COMPONENT tiledarray )
  # Add these dependencies to External
  add_dependencies(External-tiledarray scalapackpp )

  # set {blacspp,scalapackpp}_CONFIG to the install location so that we know where to find it
  set(blacspp_CONFIG ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/blacspp/blacspp-config.cmake)
  set(scalapackpp_CONFIG ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/cmake/scalapackpp/scalapackpp-config.cmake)

endif()

add_library( TiledArray_SCALAPACK INTERFACE )
target_link_libraries( TiledArray_SCALAPACK INTERFACE scalapackpp::scalapackpp blacspp::blacspp)

install( TARGETS TiledArray_SCALAPACK EXPORT tiledarray COMPONENT tiledarray )

set( TILEDARRAY_HAS_SCALAPACK 1 )
