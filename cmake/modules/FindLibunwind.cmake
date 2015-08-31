# - Try to find Libunwind
# Input variables:
#  LIBUNWIND_ROOT_DIR     - The libunwind install directory
#  LIBUNWIND_INCLUDE_DIR  - The libunwind include directory
#  LIBUNWIND_LIBRARY      - The libunwind library directory
# Output variables:
#  Libunwind_FOUND        - System has libunwind
#  Libunwind_INCLUDE_DIRS - The libunwind include directories
#  Libunwind_LIBRARIES    - The libraries needed to use libunwind
#  Libunwind_VERSION      - The version string for libunwind

include(FindPackageHandleStandardArgs)
  
if(NOT Libunwind_FOUND)

  # Set default sarch paths for libunwind
  if(LIBUNWIND_ROOT_DIR)
    set(LIBUNWIND_INCLUDE_DIR ${LIBUNWIND_ROOT_DIR}/include CACHE PATH "The include directory for libunwind")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
      set(LIBUNWIND_LIBRARY ${LIBUNWIND_ROOT_DIR}/lib64;${LIBUNWIND_ROOT_DIR}/lib CACHE PATH "The library directory for libunwind")
    else()
      set(LIBUNWIND_LIBRARY ${LIBUNWIND_ROOT_DIR}/lib CACHE PATH "The library directory for libunwind")
    endif()
  endif()
  
  find_path(Libunwind_INCLUDE_DIRS NAMES libunwind.h
      HINTS ${LIBUNWIND_INCLUDE_DIR})
  
  find_library(Libunwind_LIBRARIES unwind 
      HINTS ${LIBUNWIND_LIBRARY})
  
  # Get libunwind version
  if(Libunwind_INCLUDE_DIRS)
    file(READ "${Libunwind_INCLUDE_DIRS}/libunwind-common.h" _libunwind_version_header)
    string(REGEX MATCH "define[ \t]+UNW_VERSION_MAJOR[ \t]+([0-9]+)" LIBUNWIND_MAJOR_VERSION "${_libunwind_version_header}")
    string(REGEX MATCH "([0-9]+)" LIBUNWIND_MAJOR_VERSION "${LIBUNWIND_MAJOR_VERSION}")
    string(REGEX MATCH "define[ \t]+UNW_VERSION_MINOR[ \t]+([0-9]+)" LIBUNWIND_MINOR_VERSION "${_libunwind_version_header}")
    string(REGEX MATCH "([0-9]+)" LIBUNWIND_MINOR_VERSION "${LIBUNWIND_MINOR_VERSION}")
    string(REGEX MATCH "define[ \t]+UNW_VERSION_EXTRA[ \t]+([0-9]+)" LIBUNWIND_MICRO_VERSION "${_libunwind_version_header}")
    string(REGEX MATCH "([0-9]+)" LIBUNWIND_MICRO_VERSION "${LIBUNWIND_MICRO_VERSION}")
    set(Libunwind_VERSION "${LIBUNWIND_MAJOR_VERSION}.${LIBUNWIND_MINOR_VERSION}.${LIBUNWIND_MICRO_VERSION}")
    unset(_libunwind_version_header)
  endif()

  # handle the QUIETLY and REQUIRED arguments and set Libunwind_FOUND to TRUE
  # if all listed variables are TRUE
  find_package_handle_standard_args(Libunwind
      FOUND_VAR Libunwind_FOUND
      VERSION_VAR Libunwind_VERSION 
      REQUIRED_VARS Libunwind_LIBRARIES Libunwind_INCLUDE_DIRS)

  mark_as_advanced(LIBUNWIND_INCLUDE_DIR LIBUNWIND_LIBRARY 
      Libunwind_INCLUDE_DIRS Libunwind_LIBRARIES)

endif()