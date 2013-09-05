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
#  ConvertIncludesListToCompileArgs.cmake
#  Sep 4, 2013
#

#
# converts a list of include paths (second argument, don't forget to enclose the
# list in quotes) into a list of command-line parameters to the compiler/.
#

macro(convert_incs_to_compargs _args _inc_paths )
  # transform library list into compiler args

  # Add include paths to _args
  foreach(_inc_path ${_inc_paths})
    set(${_args} "${${_args}} -I${_inc_path}")
  endforeach()
endmacro()
