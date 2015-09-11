if(ENABLE_MONDRIAAN)

  find_package(Mondriaan REQUIRED)
    
  # Set the  build variables
  set(TiledArray_LIBRARIES ${Mondriaan_LIBRARIES} ${TiledArray_LIBRARIES})
  set(TiledArray_CONFIG_LIBRARIES ${Mondriaan_LIBRARIES} ${TiledArray_CONFIG_LIBRARIES})
  set(TILEDARRAY_HAS_MONDRIAAN 1)
      
endif()