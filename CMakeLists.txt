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

include(TestCXXAcceptsFlag)
include(CheckCXXSourceCompiles)

if (CMAKE_CXX_COMPILER_ID MATCHES Clang)
  message("* C++11: assuming Clang doesn't need -std=c++0x flag.")
else()
  # C++11 flag is required
  CHECK_CXX_ACCEPTS_FLAG("-std=c++0x" CXX_FLAG_CXX0X)
  CHECK_CXX_ACCEPTS_FLAG("-std=c++11" CXX_FLAG_CXX11)
  set(HAVE_CXX11 FALSE)
  if(CXX_FLAG_CXX11)
      set(CMAKE_CXX_FLAGS "-std=c++11 ${CMAKE_CXX_FLAGS}")
      message(STATUS "C++11 flag: -std=c++11")
      set(HAVE_CXX11 TRUE)
  elseif (CXX_FLAG_CXX0X)
    set(CMAKE_CXX_FLAGS "-std=c++0x ${CMAKE_CXX_FLAGS}")
    message(STATUS "C++11 flag: -std=c++0x")
    set(HAVE_CXX11 TRUE)
  endif()
endif()

set(TARGET_ARCH "${CMAKE_SYSTEM_NAME}-${CMAKE_SYSTEM_PROCESSOR}")
set(TILEDARRAY_MAJOR_VERSION 3)
set(TILEDARRAY_MINOR_VERSION 0)
set(TILEDARRAY_MICRO_VERSION 0)
set(TILEDARRAY_BUILDID alpha)

set(TILEDARRAY_VERSION "${TILEDARRAY_MAJOR_VERSION}.${TILEDARRAY_MINOR_VERSION}.${TILEDARRAY_MICRO_VERSION}")
set(TILEDARRAY_VERSION "${TILEDARRAY_VERSION}-${TILEDARRAY_BUILDID}")

include(CheckTypeSize)
check_type_size(int64_t C_TYPE_INT64_T)
check_type_size(int32_t C_TYPE_INT32_T)
check_type_size(long C_TYPE_LONG)
check_type_size(int C_TYPE_INT)

include(GNUInstallDirs)           

include_directories("${PROJECT_SOURCE_DIR}/src")
include_directories("${PROJECT_BINARY_DIR}/src")

if (CMAKE_BUILD_UNITTESTS)
  set (TILEDARRAY_NEEDS_BOOST TRUE)
endif()

# external dependencies
include(external/External)

CONFIGURE_FILE(
  ${PROJECT_SOURCE_DIR}/src/TiledArray/config.h.in
  ${PROJECT_BINARY_DIR}/src/TiledArray/config.h
)

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
if (TA_BUILD_UNITTEST)
  add_subdirectory(tests)
  set (TA_NEEDS_BOOST TRUE)
endif()

# Export include dirs, library list
set(TILEDARRAY_CONFIG_LIBRARIES "")

# transform library list into compiler args
include(ConvertLibrariesListToCompilerArgs)
convert_libs_to_compargs(TILEDARRAY_CONFIG_LIBRARIES "${LINK_LIBRARIES}")
message(STATUS "TILEDARRAY_CONFIG_LIBRARIES = ${TILEDARRAY_CONFIG_LIBRARIES}")

# include dirs
#get_directory_property(TILEDARRAY_INCLUDE_DIRECTORIES DIRECTORY src/bin/mpqc INCLUDE_DIRECTORIES)
# remove internal source dirs
#list(REMOVE_ITEM
#  TILEDARRAY_INCLUDE_DIRECTORIES
#  ${PROJECT_BINARY_DIR}/src/lib
#  ${PROJECT_SOURCE_DIR}/src/lib
#  ${PROJECT_SOURCE_DIR}/include
#) 

set(TILEDARRAY_CPP_FLAGS "${CMAKE_CPP_FLAGS}")
foreach (_dir ${TILEDARRAY_INCLUDE_DIRECTORIES})
  set(TILEDARRAY_CPP_FLAGS "${TILEDARRAY_CPP_FLAGS} -I${_dir}")
endforeach()

# TiledArray includes
set(TILEDARRAY_CPP_FLAGS "-I${CMAKE_INSTALL_PREFIX}/include -I${MADNESS_INCLUDE_DIR} ${TILEDARRAY_CPP_FLAGS}")
message(STATUS "TILEDARRAY_CPP_FLAGS = ${TILEDARRAY_CPP_FLAGS}")
