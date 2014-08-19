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
#  Drew Lewis & Justus Calvin
#  Department of Chemistry, Virginia Tech
#
#  FindElemental.cmake
#  Mar 26, 2014
#

# - Try to find Elemental 
# This module will look for elemental in Elemental_ROOT_DIR
# ELEMENTAL_FOUND - Elemental found
# Elemental_INCLUDE_DIR
# Elemental_LIBRARY_DIR
# ElementalL_LIBRARY
#

include(LibFindMacros)
include(FindPackageHandleStandardArgs)

set(Elemental_DEFAULT_COMPONENT_LIST pmrrr;lapack-addons)

# Check for valid component list
foreach(_component ${Elemental_FIND_COMPONENTS})
  list(FIND Elemental_DEFAULT_COMPONENT_LIST ${_component} _comp_found)
  if(_comp_found EQUAL -1)
    message(FATAL_ERROR "Invalid Elemental component: ${_component}")
  endif()
endforeach()

list(APPEND Elemental_FIND_COMPONENTS elemental)

if(NOT ELEMENTAL_FOUND)

  ###############################
  # Find the Elemental include dir
  ###############################

  libfind_header(Elemental Elemental_INCLUDE_DIRS elemental.hpp)
  
  ###################################
  # Find Elemental components
  ###################################

  foreach(_component ${Elemental_FIND_COMPONENTS})
    libfind_library(Elemental ${_component})
  endforeach()
  
  # Set the library variable if not provided by the user
  if(Elemental_LIBRARY)
    set(Elemental_LIBRARIES ${Elemental_LIBRARY})
  else()
    foreach(_component ${Elemental_DEFAULT_COMPONENT_LIST} elemental)
      if(Elemental_${_component}_FOUND)
        list(APPEND Elemental_LIBRARIES ${Elemental_${_component}_LIBRARY})
      endif()
    endforeach()
  endif()
  
  find_package_handle_standard_args(Elemental 
      REQUIRED_VARS Elemental_INCLUDE_DIRS Elemental_LIBRARIES
      HANDLE_COMPONENTS)
  
  mark_as_advanced(Elemental_INCLUDE_DIRS Elemental_LIBRARIES)

endif()
