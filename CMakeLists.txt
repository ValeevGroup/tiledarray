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

cmake_minimum_required (VERSION 3.15.0) # need list(PREPEND for toolchains

# Set TiledArray version =======================================================

# see https://semver.org/
set(TILEDARRAY_MAJOR_VERSION 1)
set(TILEDARRAY_MINOR_VERSION 0)
set(TILEDARRAY_MICRO_VERSION 0)
set(TILEDARRAY_PRERELEASE_ID )

set(TILEDARRAY_VERSION "${TILEDARRAY_MAJOR_VERSION}.${TILEDARRAY_MINOR_VERSION}.${TILEDARRAY_MICRO_VERSION}")
if (TILEDARRAY_PRERELEASE_ID)
  set(TILEDARRAY_EXT_VERSION "${TILEDARRAY_VERSION}-${TILEDARRAY_PRERELEASE_ID}")
else(TILEDARRAY_PRERELEASE_ID)
  set(TILEDARRAY_EXT_VERSION "${TILEDARRAY_VERSION}")
endif(TILEDARRAY_PRERELEASE_ID)

# Preload versions/tags of all dependencies ====================================
include(external/versions.cmake)

# Safety net for dev workflow: accidental install will not affect FindOrFetch*
if (NOT DEFINED CACHE{CMAKE_FIND_NO_INSTALL_PREFIX})
  set(CMAKE_FIND_NO_INSTALL_PREFIX ON CACHE BOOL "Whether find_* commands will search CMAKE_INSTALL_PREFIX and CMAKE_STAGING_PREFIX; see https://cmake.org/cmake/help/latest/variable/CMAKE_FIND_NO_INSTALL_PREFIX.html#variable:CMAKE_FIND_NO_INSTALL_PREFIX")
endif()

# Bring ValeevGroup cmake toolkit ==============================================
include(FetchContent)
if (DEFINED PROJECT_BINARY_DIR)
  set(VG_CMAKE_KIT_PREFIX_DIR PROJECT_BINARY_DIR)
else ()
  set(VG_CMAKE_KIT_PREFIX_DIR CMAKE_CURRENT_BINARY_DIR)
endif()
FetchContent_Declare(
    vg_cmake_kit
    QUIET
    GIT_REPOSITORY      https://github.com/ValeevGroup/kit-cmake.git
    GIT_TAG             ${TA_TRACKED_VGCMAKEKIT_TAG}
    SOURCE_DIR ${${VG_CMAKE_KIT_PREFIX_DIR}}/cmake/vg
    BINARY_DIR ${${VG_CMAKE_KIT_PREFIX_DIR}}/cmake/vg-build
    SUBBUILD_DIR ${${VG_CMAKE_KIT_PREFIX_DIR}}/cmake/vg-subbuild
)
FetchContent_MakeAvailable(vg_cmake_kit)
list(APPEND CMAKE_MODULE_PATH "${vg_cmake_kit_SOURCE_DIR}/modules")

# Declare ourselves ============================================================
project(TiledArray
        VERSION ${TILEDARRAY_VERSION}
        DESCRIPTION "TiledArray: block-sparse tensor framework for modern (distributed-memory and heterogeneous) computing"
        LANGUAGES CXX
        HOMEPAGE_URL "https://valeevgroup.github.io/tiledarray/")
enable_language(C)  # C needed even for basic platform introspection

# Set install paths ============================================================

include(GNUInstallDirs)
set(TILEDARRAY_INSTALL_BINDIR "${CMAKE_INSTALL_BINDIR}"
        CACHE PATH "TiledArray binary install directory")
set(TILEDARRAY_INSTALL_INCLUDEDIR "${CMAKE_INSTALL_INCLUDEDIR}"
        CACHE PATH "TiledArray INCLUDE install directory")
set(TILEDARRAY_INSTALL_LIBDIR "${CMAKE_INSTALL_LIBDIR}"
        CACHE PATH "TiledArray LIB install directory")
set(TILEDARRAY_INSTALL_DATADIR "${CMAKE_INSTALL_DATAROOTDIR}/tiledarray/${TILEDARRAY_EXT_VERSION}"
        CACHE PATH "TiledArray DATA install directory")
set(TILEDARRAY_INSTALL_DOCDIR "${TILEDARRAY_INSTALL_DATADIR}/doc"
        CACHE PATH "TiledArray DOC install directory")
set(TILEDARRAY_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/tiledarray"
        CACHE PATH "TiledArray CMAKE install directory")

# Add module directory and modules =============================================
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/modules/)
include(CMakePushCheckState)
include(AppendFlags)
include(RedefaultableOption)
include(DetectMADNESSConfig)
include(AddCustomTargetSubproject)
include(AddTAExecutable)
include(FindPackageRegimport)
init_package_regimport()
include(LoadFetchContent)

# Load extra CMake features ====================================================

include(CMakeDependentOption)
include(CMakePackageConfigHelpers)
include(FeatureSummary)
include(CTest)  # testing, defined BUILD_TESTING

set(MPI_CXX_SKIP_MPICXX TRUE CACHE BOOL "MPI_CXX_SKIP_MPICXX")

# Configure options =======================================================
option(ENABLE_MPI "Enable MPI" ON)
add_feature_info(MPI ENABLE_MPI "Message-Passing Interface supports distributed-memory parallel programs")

option(ENABLE_SCALAPACK "Enable ScaLAPACK Bindings in TiledArray" OFF)
add_feature_info(ScaLAPACK ENABLE_SCALAPACK "ScaLAPACK provides distributed linear algebra")

option(ENABLE_WFN91_LINALG_DISCOVERY_KIT "Use linear algebra discovery kit from github.com/wavefunction91 [recommended]" ON)
add_feature_info(WFN91LinearAlgebraDiscoveryKit ENABLE_WFN91_LINALG_DISCOVERY_KIT "Linear algebra discovery kit from github.com/wavefunction91 supports many more corner cases than the default CMake modules and/or ICL's BLAS++/LAPACK++ modules")

redefaultable_option(ENABLE_TBB "Enable use of TBB with MADNESS" OFF)
add_feature_info(TBB ENABLE_TBB "Intel Thread-Building Blocks (TBB) supports programming shared-memory systems")

option(ENABLE_CUDA "Enable use of CUDA with TiledArray" OFF)
add_feature_info(CUDA ENABLE_CUDA "NVIDIA CUDA support for GPU")

option(ENABLE_HIP "Enable use of HIP with TiledArray" OFF)
add_feature_info(HIP ENABLE_HIP "AMD HIP/ROCm support for GPU")

option(ENABLE_GPERFTOOLS "Enable linking with Gperftools" OFF)
add_feature_info(GPERFTOOLS ENABLE_GPERFTOOLS "Google Performance Tools provide fast memory allocation and performance profiling")
option(ENABLE_TCMALLOC_MINIMAL "Enable linking with tcmalloc_minimal" OFF)

if((ENABLE_GPERFTOOLS OR ENABLE_TCMALLOC_MINIMAL) AND CMAKE_SYSTEM_NAME MATCHES "Linux")
  set(ENABLE_LIBUNWIND ON)
  add_feature_info(Libunwind ENABLE_LIBUNWIND "Libunwind provides stack unwinding")
endif()

option(TA_TENSOR_MEM_TRACE "Turn on instrumented tracing of TA::Tensor memory use" OFF)
add_feature_info(TENSOR_MEM_TRACE TA_TENSOR_MEM_TRACE "instrumented tracing of TA::Tensor memory use")

option(TA_TENSOR_MEM_PROFILE "Turn on instrumented profiling of TA::Tensor memory use" ${TA_TENSOR_MEM_TRACE})
add_feature_info(TENSOR_MEM_PROFILE TA_TENSOR_MEM_PROFILE "instrumented profiling of TA::Tensor memory use")

option(TA_EXPERT "TiledArray Expert mode: disables automatically downloading or building dependencies" OFF)

option(TA_SIGNED_1INDEX_TYPE "Enables the use of signed 1-index coordinate type (OFF in 1.0.0-alpha.2 and older)" ON)
add_feature_info(SIGNED_1INDEX_TYPE TA_SIGNED_1INDEX_TYPE "Use of signed 1-index coordinate type in TiledArray")

# define this as needed
set(TA_MAX_SOO_RANK_METADATA 8 CACHE STRING "Specifies the max rank for which small object optimization will be used (hence, heap use avoided) for metadata objects")

option(TA_TRACE_TASKS "Enable debug tracing of MADNESS tasks in (some components of) TiledArray" OFF)
add_feature_info(TASK_TRACE_DEBUG TA_TRACE_TASKS "Debug tracing of MADNESS tasks in (some components of) TiledArray")
set(TILEDARRAY_ENABLE_TASK_DEBUG_TRACE ${TA_TRACE_TASKS})

option(TA_ENABLE_TILE_OPS_LOGGING "Enable logging of (some) TiledArray tile ops" OFF)
add_feature_info(TILE_OPS_LOGGING TA_ENABLE_TILE_OPS_LOGGING "Debug logging of TiledArray tile ops")
if(TA_ENABLE_TILE_OPS_LOGGING AND NOT DEFINED TA_TILE_OPS_LOG_LEVEL)
  set(TA_TILE_OPS_LOG_LEVEL 1)
endif(TA_ENABLE_TILE_OPS_LOGGING AND NOT DEFINED TA_TILE_OPS_LOG_LEVEL)

option(TA_RANGEV3 "Enable Range-V3 library" OFF)
add_feature_info(TA_RANGEV3 TA_RANGEV3 "Range-V3 ranges library")

option(TA_TTG "Enable search/build of TTG library" OFF)
add_feature_info(TA_TTG TA_TTG "TTG library")

# Enable shared library support options
redefaultable_option(TA_ASSUMES_ASLR_DISABLED "TiledArray assumes the Address Space Layout Randomization (ASLR) to be disabled" OFF)
add_feature_info(ASSUMES_ASLR_DISABLED TA_ASSUMES_ASLR_DISABLED
    "TiledArray assumes the Address Space Layout Randomization (ASLR) to be disabled")
get_property(SUPPORTS_SHARED GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS)
if (SUPPORTS_SHARED AND TA_ASSUMES_ASLR_DISABLED)
  set(default_BUILD_SHARED_LIBS ON)
else ()
  set(default_BUILD_SHARED_LIBS OFF)
endif()
redefaultable_option(BUILD_SHARED_LIBS "Enable shared libraries" ${default_BUILD_SHARED_LIBS})
if (BUILD_SHARED_LIBS OR (NOT ${TA_ASSUMES_ASLR_DISABLED}))
  set(default_CMAKE_POSITION_INDEPENDENT_CODE ON)
else ()
  set(default_CMAKE_POSITION_INDEPENDENT_CODE OFF)
endif()
redefaultable_option(CMAKE_POSITION_INDEPENDENT_CODE "Default value for POSITION_INDEPENDENT_CODE of targets" ${default_CMAKE_POSITION_INDEPENDENT_CODE})

set(BLA_STATIC FALSE CACHE BOOL "Whether to use static linkage for BLAS, LAPACK, and related libraries")
if(BUILD_SHARED_LIBS)
  set(CMAKE_MACOSX_RPATH TRUE)
else()
  set(CMAKE_MACOSX_RPATH FALSE)
endif()

# miscellaneous cmake platform-neutral and platform-specific configuration =============================
set(CMAKE_FIND_NO_INSTALL_PREFIX TRUE)  # do not search in CMAKE_INSTALL_PREFIX
set(CMAKE_SKIP_RPATH FALSE)
set(CMAKE_SKIP_BUILD_RPATH  FALSE)
set(CMAKE_SKIP_INSTALL_RPATH FALSE)
set(CMAKE_NO_SYSTEM_FROM_IMPORTED TRUE)  # do not use -isystem by default to avoid include dir ordering issues as well as other issues like https://gcc.gnu.org/onlinedocs/cpp/System-Headers.html

if(CMAKE_SYSTEM_NAME MATCHES "Darwin")
  # look for frameworks and appbundles last
  set(CMAKE_FIND_FRAMEWORK LAST)
  set(CMAKE_FIND_APPBUNDLE LAST)
endif()



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
if (NOT CMAKE_CXX_COMPILER)
  message(FATAL_ERROR "C++ compiler not found")
endif()

# Set the default fortran integer type. This is only used by MADNESS.
set(INTEGER4 TRUE CACHE BOOL "If TRUE, use integer*4 Fortran integers in BLAS calls. Otherwise use integer*8.")
mark_as_advanced(INTEGER4)

# Set the align size
include(DetectAlignSize)
if (NOT DEFINED CACHE{TA_ALIGN_SIZE})
  set(TA_ALIGN_SIZE "${TA_ALIGN_SIZE_DETECTED}" CACHE STRING "Set the default alignment of data buffers used by array tiles (DO NOT CHANGE THIS VALUE UNLESS YOU KNOW WHAT YOU ARE DOING)")
endif()
mark_as_advanced(TA_ALIGN_SIZE)
set(TILEDARRAY_ALIGN_SIZE ${TA_ALIGN_SIZE})

# Set the CPU L1 cache line size.
set(CACHE_LINE_SIZE "64" CACHE STRING "Set the CPU L1 cache line size in bytes (DO NOT CHANGE THIS VALUE UNLESS YOU KNOW WHAT YOU ARE DOING)")
mark_as_advanced(CACHE_LINE_SIZE)
set(TILEDARRAY_CACHELINE_SIZE ${CACHE_LINE_SIZE})

##########################
# Get the git metadata
##########################
include(GetGitMetadata)
vgkit_cmake_git_metadata()

##########################
# Check compiler features
##########################
# need C++17, insist on strict standard
set(CMAKE_CXX_STANDARD 17 CACHE STRING "C++ ISO Standard version")
if (NOT(CMAKE_CXX_STANDARD EQUAL 17 OR CMAKE_CXX_STANDARD EQUAL 20))
  message(FATAL_ERROR "C++ 2017 ISO Standard or higher is required to compile TiledArray")
endif()
# C++20 is only configurable via compile features with cmake 3.12 and older
if (CMAKE_CXX_STANDARD EQUAL 20 AND CMAKE_VERSION VERSION_LESS 3.12.0)
    cmake_minimum_required (VERSION 3.12.0)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF CACHE BOOL  "Whether to use extensions of C++  ISO Standard version")
# Check type support
include(CheckTypeSize)
check_type_size("long double" TILEDARRAY_HAS_LONG_DOUBLE LANGUAGE CXX)
check_type_size("long long" TILEDARRAY_HAS_LONG_LONG LANGUAGE CXX)

# TA_ASSERT
if (CMAKE_BUILD_TYPE STREQUAL Release OR
    CMAKE_BUILD_TYPE STREQUAL MinSizeRel)
  set (TA_ASSERT_POLICY_DEFAULT TA_ASSERT_IGNORE)
else ()
  set (TA_ASSERT_POLICY_DEFAULT TA_ASSERT_THROW)
endif()
set (TA_ASSERT_POLICY ${TA_ASSERT_POLICY_DEFAULT} CACHE STRING "Controls the behavior of TA_ASSERT; TA_ASSERT_THROW causes TA_ASSERT to throw, TA_ASSERT_ABORT causes TA_ASSERT to abort, TA_ASSERT_IGNORE makes TA_ASSERT a no-op")
set_property(
  CACHE TA_ASSERT_POLICY PROPERTY
  STRINGS TA_ASSERT_THROW TA_ASSERT_ABORT TA_ASSERT_IGNORE)

##########################
# Include source dirctories
##########################
include_directories(${PROJECT_SOURCE_DIR}/src ${PROJECT_BINARY_DIR}/src)

##########################
# external dependencies
##########################
add_custom_target(External-tiledarray)

# required deps:
# 1. derive runtime (CUDA/HIP/...) first since others may depend on it
if(ENABLE_CUDA)
  include(external/cuda.cmake)
endif()
if(ENABLE_HIP)
  include(external/hip.cmake)
endif()
include(${PROJECT_SOURCE_DIR}/cmake/modules/FindOrFetchMADWorld.cmake)
if (TA_TTG)
  include(${PROJECT_SOURCE_DIR}/cmake/modules/FindOrFetchTTG.cmake)
endif(TA_TTG)
detect_MADNESS_configuration()
include(external/eigen.cmake)
# the FetchContent-based version will not work due to BLT target name conflicts
# include(${PROJECT_SOURCE_DIR}/cmake/modules/FindOrFetchUmpire.cmake)
# use the ExternalProject-based version
include(external/umpire.cmake)

###### discover linear algebra

# use NWChemEx/David's linear algebra discovery modules?
# - yes => Invoke first to configure the correct libraries config and run modules to find BLAS/LAPACK/ScaLAPACK(if needed)
# - no  => BLAS/LAPACK will be discovered by BLAS++/LAPACK++ (loaded by BTAS) which use standard CMake modules or
#          their custom modules; if needed, ScaLAPACK will be discovered by BLACS++
if (ENABLE_WFN91_LINALG_DISCOVERY_KIT)
  include(FetchWfn91LinAlgModules)
  include(FindLinalg)
endif(ENABLE_WFN91_LINALG_DISCOVERY_KIT)
# BTAS does a better job of building and checking Boost since it uses Boost::serialization
# it also memorized the location of its config for use from install tree
include(${PROJECT_SOURCE_DIR}/cmake/modules/FindOrFetchBTAS.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/modules/FindOrFetchBoost.cmake)
if(ENABLE_SCALAPACK)
  include(external/scalapackpp.cmake)
endif()

# optional deps:
# 1. ccache
find_program(CCACHE ccache)
if(CCACHE)
    mark_as_advanced(CCACHE)
    message (STATUS "Found ccache: ${CCACHE}")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE}" CACHE STRING "Compiler launcher to use for compiling C++")
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE}" CACHE STRING "Compiler launcher to use for compiling C")
endif(CCACHE)
# 2. range-v3
if (TA_RANGEV3)
  include(${PROJECT_SOURCE_DIR}/cmake/modules/FindOrFetchRangeV3.cmake)
endif(TA_RANGEV3)
# 3. TTG
# N.B. make sure TA configures MADNESS correctly
#if (TA_TTG)
#  include(${PROJECT_SOURCE_DIR}/cmake/modules/FindOrFetchTTG.cmake)
#endif(TA_TTG)

##########################
# sources
##########################

add_subdirectory(src)
add_subdirectory(examples)
add_subdirectory(doc)

##########################
# checking/testing
##########################
# N.B. CTest was included above
if (BUILD_TESTING)
  set(_ctest_args -V -R "tiledarray/unit/run-np.*")
  set(_ctest_args_serial -V -R "tiledarray/unit/run-np-1")
  if (DEFINED TA_UT_CTEST_TIMEOUT)
    list(APPEND _ctest_args --timeout ${TA_UT_CTEST_TIMEOUT})
    list(APPEND _ctest_args_serial --timeout ${TA_UT_CTEST_TIMEOUT})
  endif(DEFINED TA_UT_CTEST_TIMEOUT)

  # if building unit tests need to configure with TA_ASSERT_POLICY=TA_ASSERT_THROW to be able to test TA_ASSERT statements
  if (NOT (TA_ASSERT_POLICY STREQUAL TA_ASSERT_THROW))
    message(WARNING "BUILD_TESTING=ON requires configuring with TA_ASSERT_POLICY=TA_ASSERT_THROW to engage REQUIRE_THROWS() tests; will skip these tests")
  endif()

  add_custom_target_subproject(tiledarray check USES_TERMINAL COMMAND ${CMAKE_CTEST_COMMAND} ${_ctest_args})
  add_custom_target_subproject(tiledarray check_serial USES_TERMINAL COMMAND ${CMAKE_CTEST_COMMAND} ${_ctest_args_serial})
  add_subdirectory(tests)
else()
  add_custom_target_subproject(tiledarray check USES_TERMINAL COMMAND echo "WARNING: unit testing disabled. To enable, give -DBUILD_TESTING=ON to cmake")
  add_custom_target_subproject(tiledarray check_serial USES_TERMINAL COMMAND echo "WARNING: unit testing disabled. To enable, give -DBUILD_TESTING=ON to cmake")
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
  ${PROJECT_SOURCE_DIR}/src/TiledArray/version.h.in
  ${PROJECT_BINARY_DIR}/src/TiledArray/version.h
)

CONFIGURE_FILE(
  ${PROJECT_SOURCE_DIR}/tiledarray.pc.in
  ${PROJECT_BINARY_DIR}/tiledarray.pc
)

# install config files
install(FILES ${PROJECT_BINARY_DIR}/tiledarray.pc
    DESTINATION "${CMAKE_INSTALL_LIBDIR}/pkgconfig")

# include extra cmake files
install(FILES
    "${PROJECT_SOURCE_DIR}/cmake/modules/SanitizeCUDAImplicitDirectories.cmake"
    DESTINATION "${TILEDARRAY_INSTALL_CMAKEDIR}")

# Create the version file
write_basic_package_version_file(tiledarray-config-version.cmake
  VERSION ${TILEDARRAY_VERSION} COMPATIBILITY AnyNewerVersion)

# Create the targets file
export(EXPORT tiledarray
  FILE "${PROJECT_BINARY_DIR}/tiledarray-targets.cmake")

# Create the configure file
configure_package_config_file(cmake/tiledarray-config.cmake.in
    "${PROJECT_BINARY_DIR}/tiledarray-config.cmake"
  INSTALL_DESTINATION "${TILEDARRAY_INSTALL_CMAKEDIR}"
  PATH_VARS CMAKE_INSTALL_PREFIX TILEDARRAY_INSTALL_BINDIR
            TILEDARRAY_INSTALL_INCLUDEDIR TILEDARRAY_INSTALL_LIBDIR
            TILEDARRAY_INSTALL_DOCDIR TILEDARRAY_INSTALL_CMAKEDIR)

# Install config, version, and target files
install(EXPORT tiledarray
    FILE "tiledarray-targets.cmake"
    DESTINATION "${TILEDARRAY_INSTALL_CMAKEDIR}"
    COMPONENT tiledarray)
install(FILES
    "${PROJECT_BINARY_DIR}/tiledarray-config.cmake"
    "${PROJECT_BINARY_DIR}/tiledarray-config-version.cmake"
    DESTINATION "${TILEDARRAY_INSTALL_CMAKEDIR}"
    COMPONENT tiledarray)


# Add target to allow on-the-fly switching of build type

ADD_CUSTOM_TARGET(debug
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Debug ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Debug"
  )

ADD_CUSTOM_TARGET(release
  COMMAND ${CMAKE_COMMAND} -DCMAKE_BUILD_TYPE=Release ${CMAKE_CURRENT_SOURCE_DIR}
  COMMAND ${CMAKE_COMMAND} --build ${CMAKE_CURRENT_BINARY_DIR} --target all
  COMMENT "Switch CMAKE_BUILD_TYPE to Release"
  )

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  feature_summary(WHAT ALL
                  DESCRIPTION "=== TiledArray Package/Feature Info ===")
  feature_summary(FILENAME ${CMAKE_CURRENT_BINARY_DIR}/features.log WHAT ALL)
endif()

option(TA_PYTHON "Build TA python module" OFF)
if (TA_PYTHON)
  if (NOT CMAKE_POSITION_INDEPENDENT_CODE)
    message(FATAL_ERROR "Python module requires CMAKE_POSITION_INDEPENDENT_CODE=ON")
  endif()
  add_subdirectory(python)
endif()
