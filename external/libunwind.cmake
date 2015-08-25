if(ENABLE_LIBUNWIND)

  if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
    # libunwind is not supported on OS X or Windows
    set(LIBUNWIND_FOUND FALSE)
  else()
    find_package(Libunwind REQUIRED)
    
    # Set the  build variables
    list(APPEND TiledArray_LIBRARIES ${LIBUNWIND_LIBRARIES})
    list(APPEND TiledArray_CONFIG_LIBRARIES ${LIBUNWIND_LIBRARIES})
  endif()
      
endif()