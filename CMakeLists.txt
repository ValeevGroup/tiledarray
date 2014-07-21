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

# Set TiledArray version
set(TILEDARRAY_MAJOR_VERSION 1)
set(TILEDARRAY_MINOR_VERSION 0)
set(TILEDARRAY_MICRO_VERSION 0)
set(TILEDARRAY_BUILDID alpha)
set(TILEDARRAY_VERSION "${TILEDARRAY_MAJOR_VERSION}.${TILEDARRAY_MINOR_VERSION}.${TILEDARRAY_MICRO_VERSION}-${TILEDARRAY_BUILDID}")
set(TARGET_ARCH "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

option(TA_BUILD_UNITTEST "Causes building TiledArray unit tests" OFF)
option(TA_EXPERT "TiledArray Expert mode: disables automatically downloading or building dependencies" OFF)
option(DISABLE_MPI "Disable the use of MPI" OFF)

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

cmake_push_check_state()
list(APPEND CMAKE_REQUIRED_FLAGS ${CMAKE_CXX_FLAGS})

# static_assert is desired, but an implementation is included
CHECK_CXX_SOURCE_COMPILES("
  int main(int argc, char* argv[]) {
    static_assert(true, \"don't worry, be happy\");
    return 0;
  }"
  TILEDARRAY_HAVE_STATIC_ASSERT)  

# check for variadic templates
CHECK_CXX_SOURCE_COMPILES("
  #include <iostream>
  #include <iterator>
  #include <algorithm>
  template <typename... args> void print_numbers(const args&... numbers) {
    const int n = sizeof...(args);
    int nums[sizeof...(args)] = { static_cast<int>(numbers)... };
    std::copy(nums, nums+n, std::ostream_iterator<int>(std::cout, \" \"));
  }
  int main(int argc, char* argv[]) {
    print_numbers(-1);
    print_numbers(0, 1, 2, 3, 4);
    return 0;
  }"
  TILEDARRAY_HAVE_VARIADIC_TEMPLATES)  

cmake_pop_check_state()

# Check type support
include(CheckTypeSize)
check_type_size("long double" TILEDARRAY_HAS_LONG_DOUBLE)
check_type_size("long long" TILEDARRAY_HAS_LONG_LONG)

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
include(external/eigen.cmake)
if (TA_BUILD_UNITTEST)
  include(external/boost.cmake)
endif()

include(external/madness.cmake)

# look for Elemental if MADNESS has it, or Elemental_DIR was given explicitly
if (Elemental_DIR OR MADNESS_HAS_ELEMENTAL)
  # if MADNESS found but does not support elemental, too much potential for error, skip
  if (Madness_FOUND AND NOT MADNESS_HAS_ELEMENTAL)
    message(WARNING "Found MADNESS but it does not know about Elemental ... will punt on Elemental")
  else()
    include(external/elemental.cmake)
  endif()
endif()
  
if (TA_BUILD_UNITTEST)
  include(external/boost.cmake)
endif()

##########################
# TiledArray compile options
##########################
list(APPEND TiledArray_INCLUDE_DIRS 
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR}/src
    ${Madness_INCLUDE_DIRS}
    ${EIGEN_INCLUDE_DIR})
if(TILEDARRAY_HAS_ELEMENTAL)
  list(APPEND TiledArray_INCLUDE_DIRS
      ${ELEMENTAL_INCLUDE_DIR})
endif()

list(APPEND TiledArray_LIBRARIES ${Madness_LIBRARIES})
if(TILEDARRAY_HAS_ELEMENTAL)
  list(APPEND TiledArray_LIBRARIES ${ELEMENTAL_LIBRARIES})
endif()
#list(APPEND TiledArray_LIBRARIES tiledarray)
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
# wrap up
##########################
CONFIGURE_FILE(
  ${PROJECT_SOURCE_DIR}/src/TiledArray/config.h.in
  ${PROJECT_BINARY_DIR}/src/TiledArray/config.h
)

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

