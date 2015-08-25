# - Try to find Libxc
# Input variables:
#   GPERFTOOLS_ROOT_DIR    - The libxc install directory
#   GPERFTOOLS_INCLUDE_DIR - The libxc include directory
#   GPERFTOOLS_LIBRARY     - The libxc library directory
# Components: profiler, and tcmalloc or tcmalloc_minimal
# Output variables:
#   GPERFTOOLS_FOUND        - System has libxc
#   GPERFTOOLS_INCLUDE_DIRS - The libxc include directories
#   GPERFTOOLS_LIBRARIES    - The libraries needed to use libxc
#   GPERFTOOLS_VERSION      - The version string for libxc

include(FindPackageHandleStandardArgs)
  
if(NOT GPERFTOOLS_FOUND)

  if(";${Gperftools_FIND_COMPONENTS};" MATCHES ";tcmalloc;" AND ";${Gperftools_FIND_COMPONENTS};" MATCHES ";tcmalloc_minimal;")
    message("ERROR: Invalid component selection for Gperftools: ${Gperftools_FIND_COMPONENTS}")
    message("ERROR: Gperftools cannot link both tcmalloc and tcmalloc_minimual")
    message(FATAL_ERROR "Gperftools component list is invalid")
  endif() 

  # Set default sarch paths for libxc
  if(GPERFTOOLS_ROOT_DIR)
    set(GPERFTOOLS_INCLUDE_DIR ${GPERFTOOLS_ROOT_DIR}/include CACHE PATH "The include directory for libxc")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
      set(GPERFTOOLS_LIBRARY ${GPERFTOOLS_ROOT_DIR}/lib64;${GPERFTOOLS_ROOT_DIR}/lib CACHE PATH "The library directory for libxc")
    else()
      set(GPERFTOOLS_LIBRARY ${GPERFTOOLS_ROOT_DIR}/lib CACHE PATH "The library directory for libxc")
    endif()
  endif()
  
  find_path(GPERFTOOLS_INCLUDE_DIRS NAMES gperftools/malloc_extension.h
      HINTS ${GPERFTOOLS_INCLUDE_DIR})

  # Search for component libraries
  foreach(_comp profiler tcmalloc tcmalloc_minimal)
    find_library(GPERFTOOLS_${_comp}_LIBRARY ${_comp} 
        HINTS ${GPERFTOOLS_LIBRARY})
    if(GPERFTOOLS_${_comp}_LIBRARY)
      set(GPERFTOOLS_${_comp}_FOUND TRUE)
    else()
      set(GPERFTOOLS_${_comp}_FOUND FALSE)
    endif()
    
    # Set gperftools libraries
    if(GPERFTOOLS_${_comp}_FOUND)
      if(";${Gperftools_FIND_COMPONENTS};" MATCHES ";${_comp};")
        list(APPEND GPERFTOOLS_LIBRARIES ${GPERFTOOLS_${_comp}_LIBRARY})
      endif()
    endif()
  endforeach()
  
  # Set gperftools libraries if not set based on component list
  if(NOT GPERFTOOLS_LIBRARIES)
    if(GPERFTOOLS_profiler_FOUND)
      set(GPERFTOOLS_LIBRARIES ${GPERFTOOLS_profiler_LIBRARY})
    endif()
    if(GPERFTOOLS_tcmalloc_FOUND)
      list(APPEND GPERFTOOLS_LIBRARIES ${GPERFTOOLS_tcmalloc_LIBRARY})
    elseif(GPERFTOOLS_tcmalloc_minimal_FOUND)
      list(APPEND GPERFTOOLS_LIBRARIES ${GPERFTOOLS_tcmalloc_minimal_LIBRARY})
    endif()
  endif()

  # handle the QUIETLY and REQUIRED arguments and set LIBXML2_FOUND to TRUE
  # if all listed variables are TRUE
  find_package_handle_standard_args(Gperftools
      FOUND_VAR GPERFTOOLS_FOUND
      REQUIRED_VARS GPERFTOOLS_LIBRARIES GPERFTOOLS_INCLUDE_DIRS
      HANDLE_COMPONENTS)

  mark_as_advanced(GPERFTOOLS_INCLUDE_DIR GPERFTOOLS_LIBRARY 
      GPERFTOOLS_INCLUDE_DIRS GPERFTOOLS_LIBRARIES)

endif(NOT GPERFTOOLS_FOUND)
