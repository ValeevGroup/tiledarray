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
#  Sep 4, 2013
#

# Create the ccd and ccsd executable

# Add include directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${TiledArray_INCLUDE_DIRS})

add_library(inputlib OBJECT input_data.cpp)
add_dependencies(inputlib External)

# Add the ccd executable
add_executable(ccd EXCLUDE_FROM_ALL ccd.cpp $<TARGET_OBJECTS:inputlib>)
target_link_libraries(ccd ${TiledArray_LIBRARIES})
add_dependencies(ccd External)
add_dependencies(example ccd)

# Add the ccsd executable
add_executable(ccsd EXCLUDE_FROM_ALL ccsd.cpp $<TARGET_OBJECTS:inputlib>)
target_link_libraries(ccsd ${TiledArray_LIBRARIES})
add_dependencies(ccsd External)
add_dependencies(example ccsd)
