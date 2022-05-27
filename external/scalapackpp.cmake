if (NOT TARGET scalapackpp::scalapackpp)
  set(VGCMAKEKIT_TRACKED_SCALAPACKPP_TAG ${TA_TRACKED_SCALAPACKPP_TAG})
  include(FindOrFetchScaLAPACKPP)
endif()

# built {blacs,scalapack}pp as a subproject?
if (TARGET blacspp AND TARGET scalapackpp)
  install( TARGETS blacspp scalapackpp EXPORT tiledarray COMPONENT tiledarray )
  # Add these dependencies to External
  add_dependencies(External-tiledarray scalapackpp )
endif()

if (TARGET blacspp::blacspp AND TARGET scalapackpp::scalapackpp)
  add_library( TiledArray_SCALAPACK INTERFACE )
  target_link_libraries( TiledArray_SCALAPACK INTERFACE scalapackpp::scalapackpp blacspp::blacspp)

  install( TARGETS TiledArray_SCALAPACK EXPORT tiledarray COMPONENT tiledarray )

  set( TILEDARRAY_HAS_SCALAPACK 1 )
endif()