#
#  This file is a part of TiledArray.
#  Copyright (C) 2016  Virginia Tech
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
#  CMakeLists.txt
#  Dec 6th, 2016
#

# Create example executable

if(ENABLE_SCALAPACK)

foreach(_exec conversion evp)

  # Add executable
  add_executable(scalapack-${_exec} EXCLUDE_FROM_ALL ${_exec}.cpp)
  target_link_libraries(scalapack-${_exec} PRIVATE tiledarray)
  add_dependencies(examples-tiledarray scalapack-${_exec})

endforeach()

endif(ENABLE_SCALAPACK)
