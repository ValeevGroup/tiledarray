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
include_directories(${CMAKE_CURRENT_SOURCE_DIR})

add_library(inputlib EXCLUDE_FROM_ALL OBJECT input_data.cpp)
target_compile_definitions(inputlib PRIVATE 
   $<TARGET_PROPERTY:tiledarray,INTERFACE_COMPILE_DEFINITIONS>)
target_include_directories(inputlib PRIVATE 
   $<TARGET_PROPERTY:tiledarray,INTERFACE_INCLUDE_DIRECTORIES> $<TARGET_PROPERTY:TiledArray_Eigen,INTERFACE_INCLUDE_DIRECTORIES>)
target_compile_options(inputlib PRIVATE 
   $<TARGET_PROPERTY:tiledarray,INTERFACE_COMPILE_OPTIONS>)
add_dependencies(inputlib External)

# Add the ccd executable
add_executable(ccd EXCLUDE_FROM_ALL ccd.cpp $<TARGET_OBJECTS:inputlib>)
target_link_libraries(ccd PRIVATE tiledarray)
add_dependencies(ccd External)
add_dependencies(examples ccd)

# Add the ccsd executable
add_executable(ccsd EXCLUDE_FROM_ALL ccsd.cpp $<TARGET_OBJECTS:inputlib>)
target_link_libraries(ccsd PRIVATE tiledarray)
add_dependencies(ccsd External)
add_dependencies(examples ccsd)
