#; -*-CMake-*-

cmake_minimum_required (VERSION 2.8.8)
project (TiledArray)

option(TA_BUILD_UNITTEST "Causes building TiledArray unit tests" OFF)
option(TA_EXPERT "TiledArray Expert mode: disables automatically downloading or building dependencies" OFF)

enable_language (CXX)
if (NOT CMAKE_CXX_COMPILER)
  message(FATAL_ERROR "C++ compiler not found")
endif()

# Set set the default fortran integer type. This is only used by MADNESS.
set(INTEGER4 "FALSE" CACHE BOOL "Set the default Fortran integer type to integer*4")
mark_as_advanced(INTEGER4)
if(INTEGER4) 
  set(MAD_F77_INT32 yes)
else()
  set (MAD_F77_INT32 no)
endif()

# extra cmake files are shipped with TiledArray
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${PROJECT_SOURCE_DIR}/cmake/modules/")

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release")
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-g ${CMAKE_CXX_FLAGS_RELEASE}")

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
# TiledArray linker flags
##########################
if(${CMAKE_SYSTEM_NAME} MATCHES "Darwin" AND NOT ${CMAKE_SYSTEM_VERSION} VERSION_LESS 11.0)
  # Building on OS X 10.7 or later, so add "-Wl,-no_pie" linker flags if the user ommitted it. 
  string(REGEX MATCH "-Wl,-no_pie" no_pie_find ${CMAKE_EXE_LINKER_FLAGS})
  if(NOT no_pie_find)
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -Wl,-no_pie")
    #set(CMAKE_MODULE_LINKER_FLAGS "${CMAKE_MODULE_LINKER_FLAGS} -Wl,-no_pie")
    #set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,-no_pie")
    #set(STATIC_LIBRARY_FLAGS "${STATIC_LIBRARY_FLAGS} -Wl,-no_pie")
  endif()
endif()

##########################
# external dependencies
##########################
add_custom_target(External)
include(external/MPI)
include(external/Lapack)
include(external/MADNESS)
include(external/Eigen)
if (TA_BUILD_UNITTEST)
  include(external/Boost)
endif()

##########################
# headers
##########################
include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_BINARY_DIR}/src")

##########################
# libraries
##########################
set(LINK_LIBRARIES)
list(APPEND LINK_LIBRARIES ${LIBRARIES})
list(APPEND LINK_LIBRARIES ${CMAKE_DL_LIBS})

##########################
# sources
##########################
add_subdirectory(src)
#add_subdirectory(examples)
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
# Export include dirs, library list
##########################
set(TILEDARRAY_CONFIG_LIBRARIES "")

# transform library list into compiler args
include(ConvertLibrariesListToCompilerArgs)
convert_libs_to_compargs(TILEDARRAY_CONFIG_LIBRARIES "${LINK_LIBRARIES}")
#message(STATUS "TILEDARRAY_CONFIG_LIBRARIES = ${TILEDARRAY_CONFIG_LIBRARIES}")

##########################
# TiledArray CPPFLAGS
##########################
set(TILEDARRAY_CPP_FLAGS "-I${CMAKE_INSTALL_PREFIX}/include -I${MADNESS_INCLUDE_DIR} ${CMAKE_CPP_FLAGS}")
#message(STATUS "TILEDARRAY_CPP_FLAGS = ${TILEDARRAY_CPP_FLAGS}")

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

