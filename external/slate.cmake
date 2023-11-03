if (NOT TARGET slate)
  set(VGCMAKEKIT_TRACKED_SLATE_TAG ${TA_TRACKED_SLATE_TAG} CACHE STRING "slate tag")
  include(FindOrFetchSLATE)
endif()

# built {blacs,scalapack}pp as a subproject? install as part of tiledarray export as well
# to be able to use TiledArray_SLATE from the build tree
if (TARGET slate)
    install( TARGETS slate EXPORT tiledarray COMPONENT tiledarray )
    # Add these dependencies to External
    add_dependencies(External-tiledarray slate )
endif()

if (TARGET slate)
  add_library( TiledArray_SLATE INTERFACE )
  target_link_libraries( TiledArray_SLATE INTERFACE slate )

  install( TARGETS TiledArray_SLATE EXPORT tiledarray COMPONENT tiledarray )

  set( TILEDARRAY_HAS_SLATE 1 )
endif()
