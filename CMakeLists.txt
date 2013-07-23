#; -*-CMake-*-

cmake_minimum_required (VERSION 2.8.8)
project (TiledArray)

option(TA_BUILD_UNITTEST "Causes building TiledArray unit tests" OFF)
option(TA_EXPERT "TiledArray Expert mode: disables automatically downloading or building dependencies" OFF)

enable_language (CXX)
if (NOT CMAKE_CXX_COMPILER)
  message(FATAL_ERROR "C++ compiler not found")
endif()

if(INTEGER8) 
  set(F77_INTEGER_WIDTH 8)
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -DINTEGER77='INTEGER*8' ")
else()
  set(F77_INTEGER_WIDTH 4)
  set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} -DINTEGER77='INTEGER*4' ")
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
include(TestCXXAcceptsFlag)
include(CheckCXXSourceCompiles)

# some C++11 support is needed, figure out how to turn on the support
CHECK_CXX_ACCEPTS_FLAG("-std=c++11" CXX_FLAG_CXX11)
if(CXX_FLAG_CXX11)
  set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
  message(STATUS "C++11 flag: -std=c++11")
else(CXX_FLAG_CXX11)
  CHECK_CXX_ACCEPTS_FLAG("-std=c++0x" CXX_FLAG_CXX0X)
  if (CXX_FLAG_CXX0X)
    set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
    message(STATUS "C++11 flag: -std=c++0x")
  endif()
endif()

# static_assert is desired, but an implementation is included
set(CMAKE_REQUIRED_FLAGS_CURRENT ${CMAKE_REQUIRED_FLAGS})
list(APPEND CMAKE_REQUIRED_FLAGS ${CMAKE_CXX_FLAGS})
CHECK_CXX_SOURCE_COMPILES("
  int main(int argc, char* argv[]) {
    static_assert(true, \"don't worry, be happy\");
  }"
  TILEDARRAY_HAVE_STATIC_ASSERT)  
set(CMAKE_REQUIRED_FLAGS ${CMAKE_REQUIRED_FLAGS_CURRENT})

set(TARGET_ARCH "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(TILEDARRAY_MAJOR_VERSION 1)
set(TILEDARRAY_MINOR_VERSION 0)
set(TILEDARRAY_MICRO_VERSION 0)
set(TILEDARRAY_BUILDID alpha)

set(TILEDARRAY_VERSION "${TILEDARRAY_MAJOR_VERSION}.${TILEDARRAY_MINOR_VERSION}.${TILEDARRAY_MICRO_VERSION}")
set(TILEDARRAY_VERSION "${TILEDARRAY_VERSION}-${TILEDARRAY_BUILDID}")

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

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_BINARY_DIR}/src")

if (TA_BUILD_UNITTEST)
  set (TILEDARRAY_NEEDS_BOOST TRUE)
endif()

# external dependencies
include(external/External)

# libraries
set(LINK_LIBRARIES)
list(APPEND LINK_LIBRARIES ${MADNESS_LIBRARIES})
list(APPEND LINK_LIBRARIES ${Boost_LIBRARIES})
list(APPEND LINK_LIBRARIES ${LAPACK_LIBRARIES})
list(APPEND LINK_LIBRARIES ${BLAS_LIBRARIES})
list(APPEND LINK_LIBRARIES ${MPI_LIBRARIES})
list(APPEND LINK_LIBRARIES ${LIBRARIES})
list(APPEND LINK_LIBRARIES ${CMAKE_DL_LIBS})

# sources
add_subdirectory(src)
add_subdirectory(doc)

# checking/testing
enable_testing()
if (TA_BUILD_UNITTEST)
  add_custom_target(check COMMAND ${CMAKE_CTEST_COMMAND})
  add_subdirectory(tests)
else()
  add_custom_target(check COMMAND echo "WARNING: unit testing disabled. To enable, add --unittest to configure, or give -DTA_BUILD_UNITTEST=TRUE to cmake")
endif()

# Export include dirs, library list
set(TILEDARRAY_CONFIG_LIBRARIES "")

# transform library list into compiler args
include(ConvertLibrariesListToCompilerArgs)
convert_libs_to_compargs(TILEDARRAY_CONFIG_LIBRARIES "${LINK_LIBRARIES}")
#message(STATUS "TILEDARRAY_CONFIG_LIBRARIES = ${TILEDARRAY_CONFIG_LIBRARIES}")

# TiledArray CPPFLAGS
set(TILEDARRAY_CPP_FLAGS "-I${CMAKE_INSTALL_PREFIX}/include -I${MADNESS_INCLUDE_DIR} ${CMAKE_CPP_FLAGS}")
#message(STATUS "TILEDARRAY_CPP_FLAGS = ${TILEDARRAY_CPP_FLAGS}")

# convert string values of TA_ERROR to numerical values expected by TA_DEFAULT_ERROR
set (TA_DEFAULT_ERROR 1)
if (TA_ERROR STREQUAL none)
  set (TA_DEFAULT_ERROR 0)
elseif (TA_ERROR STREQUAL throw)
  set (TA_DEFAULT_ERROR 1)
elseif (TA_ERROR STREQUAL assert)
  set (TA_DEFAULT_ERROR 2)
endif()

# wrap up
CONFIGURE_FILE(
  ${PROJECT_SOURCE_DIR}/src/TiledArray/config.h.in
  ${PROJECT_BINARY_DIR}/src/TiledArray/config.h
)

