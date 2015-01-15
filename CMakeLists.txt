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
option(TA_BUILD_UNITTEST "Causes building TiledArray unit tests" OFF)
option(TA_EXPERT "TiledArray Expert mode: disables automatically downloading or building dependencies" OFF)

# Set TiledArray version
set(TILEDARRAY_MAJOR_VERSION 1)
set(TILEDARRAY_MINOR_VERSION 0)
set(TILEDARRAY_MICRO_VERSION 0)
set(TILEDARRAY_BUILDID alpha)
set(TILEDARRAY_VERSION "${TILEDARRAY_MAJOR_VERSION}.${TILEDARRAY_MINOR_VERSION}.${TILEDARRAY_MICRO_VERSION}-${TILEDARRAY_BUILDID}")
set(TARGET_ARCH "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")

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
#include(TestCXXAcceptsFlag)
include(CheckCXXSourceCompiles)

# Check for gerneral C++11 support
CHECK_CXX_SOURCE_COMPILES(
    "
    #if (__cplusplus < 201103L) && ! defined(__INTEL_CXX11_MODE__)
    #error C++11 support not enabled.
    #endif
    int main(){ return 0; }
    "
    TILEDARRAY_HAS_CXX11)

if(TILEDARRAY_HAS_CXX11)
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
 
endif(TILEDARRAY_HAS_CXX11)

# Check type support
include(CheckTypeSize)
check_type_size("long double" TILEDARRAY_HAS_LONG_DOUBLE)
check_type_size("long long" TILEDARRAY_HAS_LONG_LONG)

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


# Set the  build variables for this project; not to be used by the *users* of TiledArray
include_directories(${Madness_INCLUDE_DIRS} ${EIGEN_INCLUDE_DIR} ${MPI_INCLUDE_PATH})
include_directories(${Elemental_INCLUDE_DIRS})
  include_directories()
include_directories( 
    ${PROJECT_SOURCE_DIR}/src
    ${PROJECT_BINARY_DIR}/src
    ${MPI_INCLUDE_PATH})
set(TiledArray_LIBRARIES ${Madness_LIBRARIES} ${TiledArray_LIBRARIES})

append_flags(CMAKE_EXE_LINKER_FLAGS "${BLAS_LINKER_FLAGS}")
append_flags(CMAKE_EXE_LINKER_FLAGS "${LAPACK_LINKER_FLAGS}")
set(TiledArray_LIBRARIES "${LAPACK_LIBRARIES}" "${BLAS_LIBRARIES}" ${TiledArray_LIBRARIES})


##########################
# sources
##########################

include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_BINARY_DIR}/src)

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

foreach(_lib ${TiledArray_LIBRARIES})
  append_flags(TiledArray_LIBRARIES_DEMANGLED "${_lib}")
endforeach()
foreach(_path ${TiledArray_INCLUDE_DIRS})
  append_flags(TiledArray_INCLUDE_DIRS_DEMANGLED "-I${_path}")
endforeach()
CONFIGURE_FILE(
  ${PROJECT_SOURCE_DIR}/tiledarray.pc.in
  ${PROJECT_BINARY_DIR}/tiledarray.pc
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

