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
#  Jul 19, 2013
#

CONFIGURE_FILE(
  ${CMAKE_CURRENT_SOURCE_DIR}/unit_test_config_h.in
  ${PROJECT_BINARY_DIR}/tests/unit_test_config.h
)

# Create the ta_test executable
set(executable ta_test)
file(GLOB ta_test_src_files *.cpp)
if(NOT TILEDARRAY_HAS_ELEMENTAL)
    file(GLOB elem_file *elemental.cpp)
    list(REMOVE_ITEM ta_test_src_files ${elem_file})
endif(NOT TILEDARRAY_HAS_ELEMENTAL)
add_executable(${executable} EXCLUDE_FROM_ALL ${ta_test_src_files})

# Add include directories and compiler flags for ta_test
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${PROJECT_BINARY_DIR}/tests
    ${TiledArray_INCLUDE_DIRS}
    ${Boost_INCLUDE_DIRS})
set_target_properties(${executable} PROPERTIES
    COMPILE_FLAGS "${CMAKE_CPP_FLAGS} ${TiledArray_COMPILE_FLAGS}"
    COMPILE_DEFINITIONS "TILEDARRAY_NO_USER_ERROR_MESSAGES=1"
    LINK_FLAGS "${TiledArray_LINK_FLAGS}")
target_link_libraries(${executable} ${TiledArray_LIBRARIES})

# Add targets
add_dependencies(${executable} External)
add_dependencies(check ${executable})

# Add a test
add_test(${executable} ${executable})
