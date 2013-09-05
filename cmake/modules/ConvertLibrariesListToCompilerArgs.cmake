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
#  ConvertLibrariesListToCompilerArgs.cmake
#  Jul 19, 2013
#

#
# Converts a list of libraries (second argument, don't forget to enclose the 
# list in quotes) into a list of command-line parameters to the compiler/linker.
#

macro(convert_libs_to_compargs _args _libs )
  # transform library list into compiler args
  foreach (_lib ${_libs})
    get_filename_component(_ext ${_lib} EXT)
    get_filename_component(_libname ${_lib} NAME_WE)
    
    if(APPLE AND "${_ext}" STREQUAL ".framework")

      # Handle Apple Frameworks
      get_filename_component(_path ${_lib} PATH)
      if(${_path} STREQUAL "/System/Library/Frameworks")
        set(MAD_LIBS "${${_args}} -F${_path} -framework ${_libname}")
      else()
        set(MAD_LIBS "${${_args}} -framework ${_libname}")
      endif()

    else()
      
      # Handle the general case
      set(MAD_LIBS "${${_args}} ${_lib}")
    endif()

  endforeach()
endmacro()
