#
#  This file is a part of TiledArray.
#  Copyright (C) 2014  Virginia Tech
#
#  This program is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  This program is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
#  Drew Lewis 
#  Department of Chemistry, Virginia Tech
#
#  FindElemental.cmake
#  Mar 26, 2014
#

# - Try to find Elemental 
# This module will look for elemental in Elemental_DIR
# Elemental_FOUND - Elemental found
# ELEMENTAL_INCLUDE_DIR
# ELEMENTAL_LIBRARY_DIR
# ELEMENTAL_LIBRARY
#
#######################################
# Module macros
#######################################
include(LibFindMacros)
include(AppendFlags)

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

# Assume elemental found until proven otherwise
set(Elemental_FOUND TRUE)

########################################
# Find Dependencies
########################################

# Find MPI
if(NOT DISABLE_MPI)
  libfind_package(Elemental MPI)
  if(NOT MPI_FOUND)
    set(Elemental_FOUND FALSE)
  endif()
else()
  message(FATAL_ERROR "Elemental needs MPI")
endif()


########################################
# Find Elemental include dir
########################################

if(Elemental_FOUND AND NOT Elemental_INCLUDE_DIR)
    find_path(ELEMENTAL_INCLUDE_DIR 
        NAMES elemental.hpp
        PATHS ${Elemental_DIR}/include ${CMAKE_PREFIX_PATH}
        DOC "Elemental include path"
        NO_CMAKE_SYSTEM_PATH)
    if(NOT ELEMENTAL_INCLUDE_DIR)
        message(FATAL_ERROR "ELEMENTAL include directory could not be found")
        SET(Elemental_FOUND false)
    endif()
endif()

if(NOT Elemental_INCLUDE_DIRS)

  if(Elemental_INCLUDE_DIR)
    set(Elemental_INCLUDE_DIRS ${Elemental_INCLUDE_DIR})
  endif()

  foreach(lang _C_ _CXX_ _)

    if(MPI${lang}FOUND)
      if(MPI${lang}INCLUDE_PATH)
        list(APPEND Elemental_INCLUDE_DIRS ${MPI${lang}INCLUDE_PATH})
      endif()
      break()
    endif()

  endforeach()

endif()

########################################
# Find Elemental Components
########################################

if(Elemental_FOUND AND NOT ELEMENTAL_LIBRARY)

    lib_add_dep(Elemental_FIND_COMPONENTS elemental "pmrrr")
    lib_add_dep(Elemental_FIND_COMPONENTS elemental "elemental")
    
    if(ELEMENTAL_LIBRARY_DIR)
        set(test_lib_dir ${ELEMENTAL_LIBRARY_DIR})
    else(ELEMENTAL_LIBRARY_DIR)
        if(Elemental_DIR)
            set(test_lib_dir ${Elemental_DIR}/lib)
        else(Elemental_DIR)
            set(test_lib_dir "")
        endif(Elemental_DIR)
   endif(ELEMENTAL_LIBRARY_DIR)
   
   #Find components
   MESSAGE(STATUS "Elemental_Components = ${Elemental_FIND_COMPONENTS}")
   foreach(COMPONENT ${Elemental_FIND_COMPONENTS})
    if(NOT DEFINED Elemental_${COMPONENT}_FOUND)
        MESSAGE(STATUS "Looking for Elemental Library ${COMPONENT} in ${test_lib_dir} or ${CMAKE_PREFIX_PATH}")
        find_library(Elemental_${COMPONENT}_LIBRARY
            NAMES ${COMPONENT}
            PATHS ${test_lib_dir} ${CMAKE_PREFIX_PATH}
            DOC "ELEMENTAL ${COMPONENT} library"
            NO_CMAKE_SYSTEM_PATH)
            
        if(Elemental_${COMPONENT}_LIBRARY)
            MESSAGE(STATUS "Library ${COMPONENT} found")
            if(NOT ELEMENTAL_LIBRARY_DIR)
                get_filename_component(ELEMENTAL_LIBRARY_DIR ${Elemental_${COMPONENT}_LIBRARY} PATH CACHE)
            endif(NOT ELEMENTAL_LIBRARY_DIR)
                set(Elemental_${COMPONENT}_FOUND TRUE)
        else(Elemental_${COMPONENT}_LIBRARY)
            set(Elemental_FOUND FALSE)
            set(Elemental_${COMPONENT}_FOUND FALSE)
        endif(Elemental_${COMPONENT}_LIBRARY)
     endif()
    endforeach()
                
    foreach(COMPONENT elemental pmrrr lapack-addons)
        if(Elemental_${COMPONENT}_FOUND)
            list(APPEND ELEMENTAL_LIBRARY ${Elemental_${COMPONENT}_LIBRARY})               
        endif()
    endforeach()

endif() 

########################################
# Find Elemental Components
########################################

if(NOT ELEMENTAL_LIBRARIES)
    if(ELEMENTAL_LIBRARY)
        set(ELEMENTAL_LIBRARIES ${ELEMENTAL_LIBRARY})
    endif()

    #add MPI libraries
    foreach(lang _C_ _CXX_ _)
      if(MPI${lang}FOUND)
        if(MPI${lang}LIBRARIES)
          list(APPEND ELEMENTAL_LIBRARIES ${MPI${lang}LIBRARIES})
        endif()
        break()
      endif()
    endforeach()

    #Add LAPACK_LIBRARIES
    if(LAPACK_LIBRARIES)
      list(APPEND ELEMENTAL_LIBRARIES ${LAPACK_LIBRARIES})
    endif()

endif()

########################################
# Set ELEM LINK
########################################
if(NOT ELEMENTAL_LINK_FLAGS)

  # ADD ELEMENTAL LIBRARIES
  set(ELEMENTAL_LINK_FLAGS "")

  # ADD MPI 
  foreach(lang _C_ _CXX_ _)
    if(MPI${lang}FOUND)
      append_flags(ELEMENTAL_LINK_FLAGS "${MPI${lang}LINK_FLAGS}")
      break()
    endif()
  endforeach()

  # Add LAPACK libraries
  append_flags(ELEMENTAL_LINK_FLAGS "${LAPACK_LINKER_FLAGS}")

endif()

########################################
# Set ELEM cache variables
########################################

mark_as_advanced(ELEMENTAL_INCLUDE_DIR ELEMENTAL_INCLUDE_DIRS ELEMENTAL_LIBRARIES
  ELEMENTAL_LINK_FLAGS ELEMENTAL_LIBRARY)

set(Elemental_DIR ${Elemental_DIR} CACHE PATH "Elemental install path")
set(ELEMENTAL_LIBRARIES "${ELEMENTAL_LIBRARIES}" CACHE STRING "A comma seperated list of ELEMENTAL
  libraries and their dependencies")
set(ELEMENTAL_INCLUDE_DIRS "${ELEMENTAL_INCLUDE_DIR}" CACHE STRING "A comma 
  seperated list of Elemental include paths and its dependencies")
  set(ELEMENTAL_LINK_FLAGS "${ELEMENTAL_LINK_FLAGS}" CACHE STRING "ELemental link flags")

























