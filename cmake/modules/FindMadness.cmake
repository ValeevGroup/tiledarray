# - Try to find MADNESS lib
#
# This module will search for MADNESS components in Madness_DIR and the default
# search path. It will also check for MADNESS dependencies: LAPACK and MPI. This
# module will define:
#
#   Madness_FOUND                - System has MADNESS lib with correct version
#   Madness_INCLUDE_DIR          - The MADNESS include directory
#   Madness_INCLUDE_DIRS         - The MADNESS include directory
#   Madness_LIBRARY_DIR          - The MADNESS library directory
#   Madness_LIBRARY              - The MADNESS library
#   Madness_LIBRARIES            - The MADNESS libraries and their dependencies
#   Madness_<COMPONENT>_FOUND    - System has the specified MADNESS COMPONENT
#   Madness_<COMPONENT>_LIBRARY  - The MADNESS COMPONENT library
#
# Dependicy variables are also defined by the FindMPI and FindLAPACK modules.
# See the CMake documentation for details.
#
# Valid COMPONENTS are:
#   MADmra 
#   MADtinyxml 
#   MADmuparser
#   MADlinalg
#   MADtensor
#   MADmisc
#   MADworld
#

include(LibFindMacros)
include(LibAddDep)

#Create a cache variable for the MADNESS install path
set(Madness_DIR "" CACHE PATH "MADNESS install path")

# Dependencies
libfind_package(MADNESS LAPACK)
if(NOT DISABLE_MPI)
  libfind_package(MADNESS MPI)
endif()

# Find the MADNESS include dir
if (NOT Madness_FIND_QUIETLY)
  message(STATUS "Looking for madness_config.h")
endif()
find_path(Madness_INCLUDE_DIR
    NAMES madness_config.h 
    PATHS ${Madness_DIR}/include ${CMAKE_PREFIX_PATH}
    DOC "MADNESS include path"
    NO_CMAKE_SYSTEM_PATH)
if(NOT Madness_FIND_QUIETLY)
  if(Madness_INCLUDE_DIR)
    message(STATUS "Looking for madness_config.h - found")
  else()
    message(STATUS "Looking for madness_config.h - not found")
  endif()
endif()

# Add any missing dependencies to the commponent list
lib_add_dep(Madness_FIND_COMPONENTS MADmra "MADtinyxml;MADmuparser;MADlinalg")
lib_add_dep(Madness_FIND_COMPONENTS MADlinalg "MADtensor")
lib_add_dep(Madness_FIND_COMPONENTS MADtensor "MADmisc")
lib_add_dep(Madness_FIND_COMPONENTS MADmisc "MADworld")

# Set the first library test path
if(Madness_LIBRARY_DIR)
  set(test_lib_dir ${Madness_LIBRARY_DIR})
else(Madness_LIBRARY_DIR)
  if(Madness_DIR)
    set(test_lib_dir ${Madness_DIR}/lib)
  else()
    set(test_lib_dir "")
  endif(Madness_DIR)
endif(Madness_LIBRARY_DIR)

# Finally the library itself
foreach (COMPONENT ${Madness_FIND_COMPONENTS})

  if(NOT DEFINED Madness_${COMPONENT}_FOUND)
    # Find the component library.
    if(NOT Madness_FIND_QUIETLY)
      message(STATUS "Looking for ${COMPONENT}")
    endif(NOT Madness_FIND_QUIETLY)
#    set(Madness_${COMPONENT}_LIBRARY Madness_${COMPONENT}-NOTFOUND)
    find_library(Madness_${COMPONENT}_LIBRARY 
        NAMES ${COMPONENT}
        PATHS ${test_lib_dir} ${CMAKE_PREFIX_PATH}
        DOC "MADNESS ${COMPONENT} library"
        NO_CMAKE_SYSTEM_PATH)

    # Set result status variables
    if(Madness_${COMPONENT}_LIBRARY)
      if(NOT Madness_LIBRARY_DIR)
        get_filename_component(Madness_LIBRARY_DIR ${Madness_${COMPONENT}_LIBRARY} PATH CACHE)
      endif(NOT Madness_LIBRARY_DIR)
      set(Madness_${COMPONENT}_FOUND TRUE)
    else(Madness_${COMPONENT}_LIBRARY)
      set(Madness_${COMPONENT}_FOUND FALSE)
    endif(Madness_${COMPONENT}_LIBRARY)

    # Print the result of the search
    if(NOT Madness_FIND_QUIETLY)
      if(Madness_${COMPONENT}_FOUND)
        message(STATUS "Looking for ${COMPONENT} - found")
      else(Madness_${COMPONENT}_FOUND)
        message(STATUS "Looking for ${COMPONENT} - not found")
      endif(Madness_${COMPONENT}_FOUND)
    endif(NOT Madness_FIND_QUIETLY)
  endif(NOT DEFINED Madness_${COMPONENT}_FOUND)
endforeach()

# Add the found libraries to Madness_PROCESS_LIBS so that they are in the proper order
foreach(COMPONENT MADmra MADtinyxml MADmuparser MADlinalg MADtensor MADmisc MADworld)
  if(Madness_${COMPONENT}_FOUND)
    list(APPEND Madness_LIBRARY ${Madness_${COMPONENT}_LIBRARY})
  endif()
endforeach()


if(DISABLE_MPI)
  set(Madness_PROCESS_INCLUDES Madness_INCLUDE_DIR)
  set(Madness_PROCESS_LIBS Madness_LIBRARY LAPACK_LIBRARIES)  
  set(Madness_COMPILE_FLAGS "")
  set(Madness_LINK_FLAGS "${LAPACK_LINKER_FLAGS}")
else()
  foreach(lang _C_ _CXX_ _)

    if(MPI${lang}FOUND)
      set(Madness_PROCESS_INCLUDES Madness_INCLUDE_DIR MPI${lang}INCLUDE_PATH)
      set(Madness_PROCESS_LIBS Madness_LIBRARY LAPACK_LIBRARIES MPI${lang}LIBRARIES)
  
      set(Madness_COMPILE_FLAGS "${MPI${lang}COMPILE_FLAGS}")
      set(Madness_LINK_FLAGS "${Madness_LINK_FLAGS} ${MPI${lang}LINK_FLAGS}")
    
      break()
    endif()

  endforeach()
endif()

if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND NOT ${CMAKE_SYSTEM_VERSION} VERSION_LESS 11.0)
  # Building on OS X 10.7 or later, so add "-Wl,-no_pie" linker flags.
  set(Madness_LINK_FLAGS "${Madness_LINK_FLAGS} -Wl,-no_pie")
endif()

# Set the include dir variables and the libraries and let libfind_process do the rest.
libfind_process(Madness)

mark_as_advanced(Madness_INCLUDE_DIR Madness_LIBRARY_DIR Madness_LIBRARY)
