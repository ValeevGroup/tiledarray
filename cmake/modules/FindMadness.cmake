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

################
# Module macros
################

include(LibFindMacros)

macro(append_flags _flags _append_flag)

  string(REGEX REPLACE "^[ ]+(.*)$" "\\1" _temp_flags "${_append_flag}")
  string(REGEX REPLACE "^(.*)[ ]+$" "\\1" _temp_flags "${_temp_flags}")
  
  set(${_flags} "${${_flags}} ${_temp_flags}")

  string(REGEX REPLACE "^[ ]+(.*)$" "\\1" ${_flags} "${${_flags}}")
  string(REGEX REPLACE "^(.*)[ ]+$" "\\1" ${_flags} "${${_flags}}")

endmacro()

macro(lib_add_dep _lib_list _lib _deps)
  
  list(FIND ${_lib_list} ${_lib} _lib_find)
  if(NOT _lib_find EQUAL -1)
    foreach(_dep ${_deps})
      list(FIND ${_lib_list} ${_dep} _dep_find)
      if(_dep_find EQUAL -1)
        list(APPEND ${_lib_list} ${_dep})
      endif()
    endforeach()
  endif()

endmacro()

# Assume MADNESS found until proven otherwise
set(Madness_FOUND TRUE)


####################
# Find Dependencies
####################

# Find Threads
set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
libfind_package(Madness Threads)
if(NOT CMAKE_USE_PTHREADS_INIT AND NOT CMAKE_USE_HP_PTHREAD_INIT)
  set(Madness_FOUND FALSE)
  if(Madness_FIND_REQUIRED)
    message(FATAL_ERROR "MADNESS requires Pthreads.")
  endif()
endif()

# Find LAPACK
libfind_package(Madness LAPACK)
if(NOT LAPACK_FOUND)
  set(Madness_FOUND FALSE)
  if(Madness_FIND_REQUIRED)
    message(FATAL_ERROR "MADNESS requires LaPACK.")
  endif()
endif()

# Find MPI
if(NOT DISABLE_MPI)
  libfind_package(Madness MPI)
  if(NOT MPI_FOUND)
    set(Madness_FOUND FALSE)
    if(Madness_FIND_REQUIRED) 
      message(FATAL_ERROR "MADNESS requires MPI.")
    endif()
  endif()
endif()

###############################
# Find the MADNESS include dir
###############################

if(Madness_FOUND AND NOT Madness_INCLUDE_DIR)

  if(NOT Madness_FIND_QUIETLY)
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
      set(Madness_FOUND FALSE)
      message(STATUS "Looking for madness_config.h - not found")
    endif()
  endif()

endif()

if(Madness_FOUND AND NOT Madness_INCLUDE_DIRS)

  set(Madness_INCLUDE_DIRS ${Madness_INCLUDE_DIR})
  if(NOT DISABLE_MPI)
    foreach(lang _C_ _CXX_ _)

      if(MPI${lang}FOUND)
        if(MPI${lang}INCLUDE_PATH)
          list(APPEND Madness_INCLUDE_DIRS ${MPI${lang}INCLUDE_PATH})
        endif()
        break()
      endif()

    endforeach()
  endif()
endif()


###################################
# Find MADNESS components
###################################

if(Madness_FOUND AND NOT Madness_LIBRARY)

  # Add any missing commponent dependencies to the commponent list
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

  # Find each component library
  foreach (COMPONENT ${Madness_FIND_COMPONENTS})

    if(NOT DEFINED Madness_${COMPONENT}_FOUND)
      # Find the component library.
      if(NOT Madness_FIND_QUIETLY)
        message(STATUS "Looking for ${COMPONENT}")
      endif(NOT Madness_FIND_QUIETLY)
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
        set(Madness_FOUND FALSE)
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

endif()


########################
# Set MADNESS libraries
########################

if(NOT Madness_LIBRARIES)

  # Add MADNESS libraries
  set(Madness_LIBRARIES ${Madness_LIBRARY})
  
  # Add MPI libraries
  if(NOT DISABLE_MPI)
    foreach(lang _C_ _CXX_ _)

      if(MPI${lang}FOUND)
        if(MPI${lang}LINK_FLAGS)
          list(APPEND Madness_LIBRARIES ${MPI${lang}LIBRARIES})
        endif()
        break()
      endif()

    endforeach()
  endif()

  # Add LAPACK libraries
  if(LAPACK_LIBRARIES)
    list(APPEND Madness_LIBRARIES ${LAPACK_LIBRARIES})
  endif()
  
  # Add thread libraries
  if(CMAKE_THREAD_LIBS_INIT)
    list(APPEND Madness_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
  endif()
endif()


############################
# Set MADNESS compile flags
############################

if(NOT Madness_COMPILE_FLAGS)

  # Add MADNESS libraries
  set(Madness_COMPILE_FLAGS "")
  
  # Add MPI libraries
  if(NOT DISABLE_MPI)
    foreach(lang _C_ _CXX_ _)

      if(MPI${lang}FOUND)
        append_flags(Madness_COMPILE_FLAGS "-DOMPI_SKIP_MPICXX=1 -DMPICH_SKIP_MPICXX=1")
        append_flags(Madness_COMPILE_FLAGS "${MPI${lang}COMPILE_FLAGS}")
        break()
      endif()

    endforeach()
  endif()

endif()


#########################
# Set MADNESS link flags
#########################

if(NOT Madness_LINK_FLAGS)

  # Add MADNESS libraries
  set(Madness_LINK_FLAGS "")
  # Add MPI libraries
  if(NOT DISABLE_MPI)
    foreach(lang _C_ _CXX_ _)

      if(MPI${lang}FOUND)
        append_flags(Madness_LINK_FLAGS "${MPI${lang}LINK_FLAGS}")
        break()
      endif()

    endforeach()
  endif()

  # Add LAPACK libraries
  append_flags(Madness_LINK_FLAGS "${LAPACK_LINKER_FLAGS}")
  
  if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND NOT ${CMAKE_SYSTEM_VERSION} VERSION_LESS 11.0)
    # Building on OS X 10.7 or later, so add "-Wl,-no_pie" linker flags.
    append_flags(Madness_LINK_FLAGS "-Wl,-no_pie")
  endif()
endif()

#########################
# Create cache variables
#########################

mark_as_advanced(Madness_INCLUDE_DIR Madness_INCLUDE_DIRS Madness_COMPILE_FLAGS
    Madness_LIBRARIES Madness_LINK_FLAGS Madness_LIBRARY)
set(Madness_DIR ${Madness_DIR} CACHE PATH "MADNESS install path")
set(Madness_LINK_FLAGS "${Madness_LINK_FLAGS}" CACHE STRING "The MADNESS link flags")
set(Madness_LIBRARIES "${Madness_LIBRARIES}" CACHE STRING "A comma seperated list of MADNESS libraries and their dependencies")
set(Madness_COMPILE_FLAGS "${Madness_COMPILE_FLAGS}" CACHE STRING "The MADNESS compile flags")
set(Madness_INCLUDE_DIRS "${Madness_INCLUDE_DIRS}" CACHE STRING "A comma sperated list of MADNESS include paths and its dependencies")

