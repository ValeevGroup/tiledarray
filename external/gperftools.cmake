if(ENABLE_GPERFTOOLS OR ENABLE_TCMALLOC_MINIMAL)

  if(CMAKE_SYSTEM_NAME MATCHES "Linux" AND (NOT Libunwind_FOUND OR (Libunwind_FOUND AND Libunwind_VERION LESS 0.99)))
    message(FATAL_ERROR "Gperftools requires libunwind 0.99 or higher, but it was was not found") 
  endif()

  if(ENABLE_TCMALLOC_MINIMAL)
    find_package(Gperftools REQUIRED tcmalloc_minimal)
  else()
    find_package(Gperftools REQUIRED)
  endif()
  
  # Set the  build variables
  set(TiledArray_LIBRARIES ${Gperftools_LIBRARIES} ${TiledArray_LIBRARIES})
  set(TiledArray_CONFIG_LIBRARIES ${Gperftools_LIBRARIES} ${TiledArray_CONFIG_LIBRARIES})
      
endif()