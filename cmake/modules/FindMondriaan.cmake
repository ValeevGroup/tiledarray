# - Try to find Mondriaan
# Input variables:
#  MONDRIAAN_ROOT_DIR     - The Mondriaan install directory
#  MONDRIAAN_INCLUDE_DIR  - The Mondriaan include directory
#  MONDRIAAN_LIBRARY      - The Mondriaan library directory
#  PATOH_INCLUDE_DIR      - The PaToH include directory
#  PATOH_LIBRARY          - The PaToH library directory
# Output variables:
#  Mondriaan_FOUND        - System has libunwind
#  Mondriaan_INCLUDE_DIRS - The libunwind include directories
#  Mondriaan_LIBRARIES    - The libraries needed to use libunwind

include(FindPackageHandleStandardArgs)
  
if(NOT Mondriaan_FOUND)

  # Set default sarch paths for Mondriaan
  if(MONDRIAAN_ROOT_DIR)
    set(MONDRIAAN_INCLUDE_DIR ${MONDRIAAN_ROOT_DIR}/include CACHE PATH "The include directory for libunwind")
    if(CMAKE_SIZEOF_VOID_P EQUAL 8 AND CMAKE_SYSTEM_NAME STREQUAL "Linux")
      set(MONDRIAAN_LIBRARY ${MONDRIAAN_ROOT_DIR}/lib64;${MONDRIAAN_ROOT_DIR}/lib CACHE PATH "The library directory for libunwind")
    else()
      set(MONDRIAAN_LIBRARY ${MONDRIAAN_ROOT_DIR}/lib CACHE PATH "The library directory for libunwind")
    endif()
  endif()
  
  find_path(Mondriaan_INCLUDE_DIRS NAMES Mondriaan.h
      HINTS ${MONDRIAAN_INCLUDE_DIR})
  
  find_library(Mondriaan_LIBRARIES Mondriaan4 
      HINTS ${MONDRIAAN_LIBRARY})

  # Find Component PaToH library      
  if(Mondriaan_FIND_COMPONENTS STREQUAL "PaToH")
    find_path(Mondriaan_PaToH_INCLUDE_DIRS NAMES patoh.h
        HINTS ${PATOH_INCLUDE_DIR})
    
    find_library(Mondriaan_PaToH_LIBRARIES patoh 
        HINTS ${PATOH_LIBRARY})

     message("Mondriaan_PaToH_INCLUDE_DIRS = '${Mondriaan_PaToH_INCLUDE_DIRS}'")
     message("Mondriaan_PaToH_LIBRARIES    = '${Mondriaan_PaToH_LIBRARIES}'")
        
    if(Mondriaan_PaToH_INCLUDE_DIRS AND Mondriaan_PaToH_LIBRARIES)
      set(Mondriaan_PaToH_FOUND TRUE)
      list(APPEND Mondriaan_INCLUDE_DIRS "${Mondriaan_PaToH_INCLUDE_DIRS}")
      list(APPEND Mondriaan_LIBRARIES "${Mondriaan_PaToH_LIBRARIES}")
    else()
      set(Mondriaan_PaToH_FOUND FALSE)
    endif()
  endif()
  
  if(Mondriaan_LIBRARIES)
    list(APPEND Mondriaan_LIBRARIES "-lm")
  endif()

  # handle the QUIETLY and REQUIRED arguments and set Mondriaan_FOUND to TRUE
  # if all listed variables are TRUE
  find_package_handle_standard_args(Mondriaan
      FOUND_VAR Mondriaan_FOUND
      REQUIRED_VARS Mondriaan_LIBRARIES Mondriaan_INCLUDE_DIRS
      HANDLE_COMPONENTS)

  mark_as_advanced(MONDRIAAN_INCLUDE_DIR MONDRIAAN_LIBRARY 
      Mondriaan_INCLUDE_DIRS Mondriaan_LIBRARIES)

endif()