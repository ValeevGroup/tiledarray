if(ENABLE_LIBUNWIND)

  if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    # libunwind is not supported on OS X or Windows
    set(Libunwind_FOUND FALSE)
  else()
    find_package(Libunwind REQUIRED)
    
    # Set the  build variables
    set(TiledArray_LIBRARIES ${Libunwind_LIBRARIES} ${TiledArray_LIBRARIES})
    set(TiledArray_CONFIG_LIBRARIES ${Libunwind_LIBRARIES} ${TiledArray_CONFIG_LIBRARIES})
  endif()
      
endif()