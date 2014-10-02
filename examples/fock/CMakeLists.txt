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
#  CMakeLists.txt
#  Feb 4, 2014
#

# Create the ta_fock_build executable

# Add include directories
include_directories(${TiledArray_INCLUDE_DIRS})

# Add the ta_fock_build executable
add_executable(ta_fock_build EXCLUDE_FROM_ALL ta_fock_build.cpp)
target_link_libraries(ta_fock_build ${TiledArray_LIBRARIES})
add_dependencies(ta_fock_build External)
add_dependencies(example ta_fock_build)

