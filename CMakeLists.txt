#; -*-CMake-*-

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

cmake_minimum_required (VERSION 2.8.8)
project (TiledArray)

# Configure options
option(ENABLE_MPI "Enable MPI" ON)
option(ENABLE_ELEMENTAL "Enable use of MADNESS provided Elemental" OFF)
option(ENABLE_TBB "Enable use of TBB with MADNESS" OFF)
option(TA_BUILD_UNITTEST "Causes building TiledArray unit tests" OFF)
option(TA_EXPERT "TiledArray Expert mode: disables automatically downloading or building dependencies" OFF)

# Set TiledArray version
set(TILEDARRAY_MAJOR_VERSION 0)
set(TILEDARRAY_MINOR_VERSION 4)
set(TILEDARRAY_MICRO_VERSION 3)
set(TILEDARRAY_BUILDID alpha)
set(TILEDARRAY_VERSION "${TILEDARRAY_MAJOR_VERSION}.${TILEDARRAY_MINOR_VERSION}.${TILEDARRAY_MICRO_VERSION}-${TILEDARRAY_BUILDID}")

# extra cmake files are shipped with TiledArray
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules/)

include(CMakePushCheckState)
include(GNUInstallDirs)
include(AppendFlags)

##########################
# Standard build variables
##########################

# Get standard build variables from the environment if they have not already been set
if(NOT CMAKE_C_FLAGS OR NOT DEFINED CMAKE_C_FLAGS)
  set(CMAKE_C_FLAGS "$ENV{CPPFLAGS}")
  append_flags(CMAKE_C_FLAGS "$ENV{CFLAGS}")
endif()
if(NOT CMAKE_CXX_FLAGS OR NOT DEFINED CMAKE_CXX_FLAGS)
  set(CMAKE_CXX_FLAGS "$ENV{CPPFLAGS}")
  append_flags(CMAKE_CXX_FLAGS "$ENV{CXXFLAGS}")
endif()
if(NOT CMAKE_EXE_LINKER_FLAGS OR NOT DEFINED CMAKE_EXE_LINKER_FLAGS)
  set(CMAKE_EXE_LINKER_FLAGS "$ENV{LDFLAGS}")
endif()

enable_language (CXX)
if (NOT CMAKE_CXX_COMPILER)
  message(FATAL_ERROR "C++ compiler not found")
endif()

# Set the default fortran integer type. This is only used by MADNESS.
set(INTEGER4 "FALSE" CACHE BOOL "Set the default Fortran integer type to integer*4")
mark_as_advanced(INTEGER4)

# Set the CPU L1 cache line size.
set(VECTOR_ALIGNMENT "16" CACHE STRING "Set the vector alignment in memory (DO NOT CHANGE THIS VALUE UNLESS YOU KNOW WHAT YOU ARE DOING)")
mark_as_advanced(VECTOR_ALIGNMENT)
set(TILEDARRAY_ALIGNMENT ${VECTOR_ALIGNMENT})

# Set the vectory.
set(CACHE_LINE_SIZE "64" CACHE STRING "Set the CPU L1 cache line size in bytes (DO NOT CHANGE THIS VALUE UNLESS YOU KNOW WHAT YOU ARE DOING)")
mark_as_advanced(CACHE_LINE_SIZE)
set(TILEDARRAY_CACHELINE_SIZE ${CACHE_LINE_SIZE})

set(CMAKE_SKIP_RPATH FALSE)

set(BUILD_TESTING FALSE CACHE BOOLEAN "BUILD_TESTING")
set(BUILD_TESTING_STATIC FALSE CACHE BOOLEAN "BUILD_TESTING_STATIC")
set(BUILD_TESTING_SHARED FALSE CACHE BOOLEAN "BUILD_TESTING_SHARED")


##########################
# Get the git revision tag information
##########################

find_package(Git REQUIRED)
execute_process(
    COMMAND ${GIT_EXECUTABLE} rev-parse -q HEAD
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    OUTPUT_VARIABLE TILEDARRAY_REVISION )
string(REGEX MATCH "[0-9a-f]*"
    TILEDARRAY_REVISION "${TILEDARRAY_REVISION}")

##########################
# Check C++11 features
##########################


include(CheckCXX11Support)
check_cxx11_support(TILEDARRAY_HAS_CXX11)
if (NOT TILEDARRAY_HAS_CXX11)
  message(FATAL_ERROR "${CMAKE_CXX_COMPILER} does not support C++11.")
endif()

# Check type support
include(CheckTypeSize)
check_type_size("long double" TILEDARRAY_HAS_LONG_DOUBLE)
check_type_size("long long" TILEDARRAY_HAS_LONG_LONG)


##########################
# Include source dirctories
##########################
include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_BINARY_DIR}/src)

##########################
# external dependencies
##########################
add_custom_target(External)

include(external/pthread.cmake)
include(external/mpi.cmake)
include(external/lapack.cmake)
include(external/eigen.cmake)
include(external/madness.cmake)
if (TA_BUILD_UNITTEST)
  include(external/boost.cmake)
endif()

##########################
# sources
##########################

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(doc)

##########################
# checking/testing
##########################
enable_testing()
if (TA_BUILD_UNITTEST)
  add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})
  add_subdirectory(tests)
else()
  add_custom_target(check COMMAND echo "WARNING: unit testing disabled. To enable, add --unittest to configure, or give -DTA_BUILD_UNITTEST=TRUE to cmake")
endif()

##########################
# convert string values of TA_ERROR to numerical values expected by TA_DEFAULT_ERROR
##########################
set (TA_DEFAULT_ERROR 1)
if (TA_ERROR STREQUAL none)
  set (TA_DEFAULT_ERROR 0)
elseif (TA_ERROR STREQUAL throw)
  set (TA_DEFAULT_ERROR 1)
elseif (TA_ERROR STREQUAL assert)
  set (TA_DEFAULT_ERROR 2)
endif()

##########################
# QT CREATOR file grabber
##########################
if(USING_QT_CREATOR_AS_IDE)
  include_directories(tests)
  include_directories(examples)
  file(GLOB_RECURSE QT_CREATOR_SRC "*.h" "*.cpp" "*.c" "*.cc" "*.hpp" "*.txt" ".in")
  add_library(qt_creator_get_sources EXCLUDE_FROM_ALL ${QT_CREATOR_SRC})
  set_target_properties(qt_creator_get_sources PROPERTIES LINKER_LANGUAGE CXX)
endif(USING_QT_CREATOR_AS_IDE)

##########################
# pkg-config variables
##########################
foreach(_inc ${TiledArray_CONFIG_INCLUDE_DIRS})
  append_flags(TiledArray_PC_CFLAGS "-I${_inc}")
endforeach()
foreach(_lib ${TiledArray_CONFIG_LIBRARIES})
  append_flags(TiledArray_PC_LIBS "${_lib}")
endforeach()

##########################
# wrap up
##########################

# Force cache refresh for compile flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}" CACHE STRING "C compile flags" FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}" CACHE STRING "C++ compile flags" FORCE)

CONFIGURE_FILE(
  ${PROJECT_SOURCE_DIR}/src/TiledArray/config.h.in
  ${PROJECT_BINARY_DIR}/src/TiledArray/config.h
)

CONFIGURE_FILE(
  ${PROJECT_SOURCE_DIR}/tiledarray.pc.in
  ${PROJECT_BINARY_DIR}/tiledarray.pc
)

CONFIGURE_FILE(
  ${PROJECT_SOURCE_DIR}/tiledarray-config.cmake.in
  ${PROJECT_BINARY_DIR}/tiledarray-config.cmake
  @ONLY)

# install config files
install(FILES ${PROJECT_BINARY_DIR}/tiledarray.pc
    DESTINATION lib/pkgconfig)
install(FILES ${PROJECT_BINARY_DIR}/tiledarray-config.cmake
    DESTINATION lib/CMake/TiledArray)


# Add target to allow on-the-fly switching of build type

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

