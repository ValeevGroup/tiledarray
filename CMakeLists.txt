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

option(TA_BUILD_UNITTEST "Causes building TiledArray unit tests" OFF)
option(TA_EXPERT "TiledArray Expert mode: disables automatically downloading or building dependencies" OFF)
option(DISABLE_MPI "Disable the use of MPI" OFF)

enable_language (CXX)
if (NOT CMAKE_CXX_COMPILER)
  message(FATAL_ERROR "C++ compiler not found")
endif()

# Set set the default fortran integer type. This is only used by MADNESS.
set(INTEGER4 "FALSE" CACHE BOOL "Set the default Fortran integer type to integer*4")
mark_as_advanced(INTEGER4)

# extra cmake files are shipped with TiledArray
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules/)

set(CMAKE_SKIP_RPATH FALSE)

set(BUILD_TESTING FALSE CACHE BOOLEAN "BUILD_TESTING")
set(BUILD_TESTING_STATIC FALSE CACHE BOOLEAN "BUILD_TESTING_STATIC")
set(BUILD_TESTING_SHARED FALSE CACHE BOOLEAN "BUILD_TESTING_SHARED")

##########################
# Check C++11 features
##########################
#include(TestCXXAcceptsFlag)
include(CheckCXXSourceCompiles)

# static_assert is desired, but an implementation is included
set(CMAKE_REQUIRED_FLAGS_CURRENT ${CMAKE_REQUIRED_FLAGS})
list(APPEND CMAKE_REQUIRED_FLAGS ${CMAKE_CXX_FLAGS})
CHECK_CXX_SOURCE_COMPILES("
  int main(int argc, char* argv[]) {
    static_assert(true, \"don't worry, be happy\");
  }"
  TILEDARRAY_HAVE_STATIC_ASSERT)  
set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_CURRENT})

# Set TiledArray version
set(TARGET_ARCH "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(TILEDARRAY_MAJOR_VERSION 1)
set(TILEDARRAY_MINOR_VERSION 0)
set(TILEDARRAY_MICRO_VERSION 0)
set(TILEDARRAY_BUILDID alpha)

set(TILEDARRAY_VERSION "${TILEDARRAY_MAJOR_VERSION}.${TILEDARRAY_MINOR_VERSION}.${TILEDARRAY_MICRO_VERSION}-${TILEDARRAY_BUILDID}")

# Check type support
include(CheckTypeSize)
check_type_size("long double" LONG_DOUBLE)
check_type_size("long long" LONG_LONG)
if (HAVE_LONG_DOUBLE)
  set (TILEDARRAY_HAS_LONG_DOUBLE TRUE)
endif()
if (HAVE_LONG_LONG)
  set (TILEDARRAY_HAS_LONG_LONG TRUE)
endif()

include(GNUInstallDirs)

##########################
# Set the MPI compliers
##########################

# Set the MPICC compiler
if(NOT DEFINED MPI_C_COMPILER AND NOT DISABLE_MPI)
  set(MPI_C_COMPILER ${CMAKE_C_COMPILER})
endif()

# Set the MPICXX complier
if(NOT DEFINED MPI_CXX_COMPILER AND NOT DISABLE_MPI)
  set(MPI_CXX_COMPILER ${CMAKE_CXX_COMPILER})
endif()

##########################
# external dependencies
##########################
add_custom_target(External)
include(external/MADNESS)
include(external/Eigen)
if (TA_BUILD_UNITTEST)
  include(external/Boost)
endif()

##########################
# TiledArray compile options
##########################
list(APPEND TiledArray_INCLUDE_DIRS 
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR}/src
    ${Madness_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIR})
list(APPEND TiledArray_LIBRARIES
    ${Madness_LIBRARIES})
#list(APPEND TiledArray_LIBRARIES
#    tiledarray)
set(TiledArray_LINK_FLAGS ${Madness_LINK_FLAGS})
set(TiledArray_COMPILE_FLAGS ${Madness_COMPILE_FLAGS})

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
# wrap up
##########################
CONFIGURE_FILE(
  ${PROJECT_SOURCE_DIR}/src/TiledArray/config.h.in
  ${PROJECT_BINARY_DIR}/src/TiledArray/config.h
)

