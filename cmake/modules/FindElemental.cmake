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
# This module will look for elemental in the Elemental_DIR
# Elemental_FOUND -System has elemental
# ELEMENTAL_INCLUDE_DIR
# ELEMENTAL_LIBRARY_DIR
# ELEMENTAL_LIBRARY
#
#######################################
# Module macros
#######################################
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

# Assume elemental found until proven otherwise
set(Elemental_FOUND TRUE)

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

if(Elemental_FOUND AND NOT ELEMENTAL_LIBRARY)

    lib_add_dep(Elemental_FIND_COMPONENTS elemental "pmrrr")
    
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
                
    foreach(COMPONENT elemental pmrrr)
        if(Elemental_${COMPONENT}_FOUND)
            list(APPEND ELEMENTAL_LIBRARY ${Elemental_${COMPONENT}_LIBRARY})               
        endif()
    endforeach()

endif() 

if(NOT ELEMENTAL_LIBRARIES)
    if(ELEMENTAL_LIBRARY)
        set(ELEMENTAL_LIBRARIES ${ELEMENTAL_LIBRARY})
    endif()
endif()

set(Elemental_DIR ${Elemental_DIR} "Elemental install path")
set(ELEMENTAL_LIBRARIES "${ELEMENTAL_LIBRARIES}")
set(ELEMENTAL_INCLUDE_DIRS "${ELEMENTAL_INCLUDE_DIR}")


































