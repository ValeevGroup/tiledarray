#
#  This file is a part of TiledArray.
#  Copyright (C) 2013  Virginia Tech
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
#  Justus Calvin
#  Department of Chemistry, Virginia Tech
#
#  FindMadness.cmake
#  Jul 25, 2013
#

# - Try to find MADNESS lib
#
# This module will search for MADNESS components in Madness_DIR and the default
# search path. This module will define the following variables:
#
#  Madness_FOUND                - System has MADNESS lib with correct version
#  Madness_INCLUDE_DIRS         - The MADNESS include directory
#  Madness_LIBRARIES            - The MADNESS libraries and their dependencies
#  Madness_<COMPONENT>_FOUND    - System has the specified MADNESS COMPONENT
#  Madness_<COMPONENT>_LIBRARY  - The MADNESS COMPONENT library
#
# The user may provide hints or override this find module with the follow
# variables
#
#  Madness_ROOT_DIR             - The MADNESS install director
#  Madness_INCLUDE_DIR          - The MADNESS include directory
#  Madness_LIBRARY              - The MADNESS library
#  Madness_<COMPONENT>_LIBRARY  - The MADNESS COMPONENT library
#
# Valid COMPONENTS are: MADchem, MADmra, MADtinyxml, MADmuparser, MADlinalg,
# MADtensor, MADmisc, and MADworld
#

include(LibFindMacros)
include(FindPackageHandleStandardArgs)

set(Madness_DEFAULT_COMPONENT_LIST MADchem;MADmra;MADtinyxml;MADmuparser;MADlinalg;MADtensor;MADmisc;MADworld)

# Check for valid component list
foreach(_component ${Madness_FIND_COMPONENTS})
  list(FIND Madness_DEFAULT_COMPONENT_LIST ${_component} _comp_found)
  if(_comp_found EQUAL -1)
    message(FATAL_ERROR "Invalid MADNESS component: ${_component}")
  endif()
endforeach()

# Set the find components
if(Madness_FIND_COMPONENTS)
  # Add any missing commponent dependencies to the commponent list
  libfind_add_dep(Madness Madness_FIND_COMPONENTS MADchem "MADmra")
  libfind_add_dep(Madness Madness_FIND_COMPONENTS MADmra "MADtinyxml;MADmuparser;MADlinalg")
  libfind_add_dep(Madness Madness_FIND_COMPONENTS MADlinalg "MADtensor")
  libfind_add_dep(Madness Madness_FIND_COMPONENTS MADtensor "MADmisc")
  libfind_add_dep(Madness Madness_FIND_COMPONENTS MADmisc "MADworld")
else()
  # Find all libraries by default
  set(Madness_FIND_COMPONENTS ${Madness_DEFAULT_COMPONENT_LIST})
  if(Madness_FIND_REQUIRED)
    foreach(_comp ${Madness_DEFAULT_COMPONENT_LIST})
      set(Madness_FIND_REQUIRED_${_comp} TRUE)
    else()
      set(Madness_FIND_REQUIRED_${_comp} FALSE)
    endforeach()
  endif()
endif()


if(NOT MADNESS_FOUND)

  ###############################
  # Find the Madness include dir
  ###############################

  libfind_header(Madness Madness_INCLUDE_DIRS madness_config.h)
  
  ###################################
  # Find Madness components
  ###################################

  foreach(_component ${Madness_FIND_COMPONENTS})
    libfind_library(Madness ${_component})
  endforeach()
  
  # Set the library variable if not provided by the user
  if(Madness_LIBRARY)
    set(Madness_LIBRARIES ${Madness_LIBRARY})
  else()
    foreach(_component ${Madness_DEFAULT_COMPONENT_LIST})
      if(Madness_${_component}_FOUND)
        list(APPEND Madness_LIBRARIES ${Madness_${_component}_LIBRARY})
      endif()
    endforeach()
  endif()

  find_package_handle_standard_args(Madness 
      REQUIRED_VARS Madness_INCLUDE_DIRS Madness_LIBRARIES
      HANDLE_COMPONENTS)
  
  mark_as_advanced(Madness_INCLUDE_DIRS Madness_LIBRARIES)

endif()