# - Try to find Libxc
# Input variables:
#   GPERFTOOLS_ROOT_DIR    - The libxc install directory
#   GPERFTOOLS_INCLUDE_DIR - The libxc include directory
#   GPERFTOOLS_LIBRARY     - The libxc library directory
# Components: profiler, and tcmalloc or tcmalloc_minimal
# Output variables:
#   Gperftools_FOUND        - System has libxc
#   Gperftools_INCLUDE_DIRS - The libxc include directories
#   Gperftools_LIBRARIES    - The libraries needed to use libxc
#   Gperftools_VERSION      - The version string for libxc

include(FindPackageHandleStandardArgs)
  
if(NOT Gperftools_FOUND)

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
  
  find_path(Gperftools_INCLUDE_DIRS NAMES gperftools/malloc_extension.h
      HINTS ${GPERFTOOLS_INCLUDE_DIR})

  # Search for component libraries
  foreach(_comp profiler tcmalloc tcmalloc_minimal)
    find_library(Gperftools_${_comp}_LIBRARY ${_comp} 
        HINTS ${GPERFTOOLS_LIBRARY})
    if(Gperftools_${_comp}_LIBRARY)
      set(Gperftools_${_comp}_FOUND TRUE)
    else()
      set(Gperftools_${_comp}_FOUND FALSE)
    endif()
    
    # Set gperftools libraries
    if(Gperftools_${_comp}_FOUND)
      if(";${Gperftools_FIND_COMPONENTS};" MATCHES ";${_comp};")
        list(APPEND Gperftools_LIBRARIES ${Gperftools_${_comp}_LIBRARY})
      endif()
    endif()
  endforeach()
  
  # Set gperftools libraries if not set based on component list
  if(NOT Gperftools_LIBRARIES)
    if(Gperftools_profiler_FOUND)
      set(Gperftools_LIBRARIES ${Gperftools_profiler_LIBRARY})
    endif()
    if(Gperftools_tcmalloc_FOUND)
      list(APPEND Gperftools_LIBRARIES ${Gperftools_tcmalloc_LIBRARY})
    elseif(Gperftools_tcmalloc_minimal_FOUND)
      list(APPEND Gperftools_LIBRARIES ${Gperftools_tcmalloc_minimal_LIBRARY})
    endif()
  endif()

  # handle the QUIETLY and REQUIRED arguments and set Gperftools_FOUND to TRUE
  # if all listed variables are TRUE
  find_package_handle_standard_args(Gperftools
      FOUND_VAR Gperftools_FOUND
      REQUIRED_VARS Gperftools_LIBRARIES Gperftools_INCLUDE_DIRS
      HANDLE_COMPONENTS)

  mark_as_advanced(GPERFTOOLS_INCLUDE_DIR GPERFTOOLS_LIBRARY 
      Gperftools_INCLUDE_DIRS Gperftools_LIBRARIES)

endif()
