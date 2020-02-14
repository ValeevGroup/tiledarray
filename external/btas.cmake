# -*- mode: cmake -*-

include(CMakePushCheckState)
include(CheckCXXSourceCompiles)
include(AppendFlags)

find_path(_BTAS_INCLUDE_DIR btas/btas.h PATHS ${BTAS_INCLUDE_DIR})

if (_BTAS_INCLUDE_DIR)

  file(STRINGS ${_BTAS_INCLUDE_DIR}/btas/version.h BTAS_REVISION_LINE REGEX "define BTAS_REVISION")
  if (BTAS_REVISION_LINE) # BTAS_REVISION found? make sure it matches the required tag exactly
    string(REGEX REPLACE ".*define[ \t]+BTAS_REVISION[ \t]+\"([a-z0-9]+)\"" "\\1" BTAS_REVISION "${BTAS_REVISION_LINE}")
    if (BTAS_TAG) # user-defined BTAS_TAG overrides TA_TRACKED_BTAS_TAG
      set(BTAS_REQUIRED_TAG "${BTAS_TAG}")
    else (BTAS_TAG)
      set(BTAS_REQUIRED_TAG "${TA_TRACKED_BTAS_TAG}")
    endif (BTAS_TAG)
    if ("${BTAS_REVISION}" STREQUAL "${BTAS_REQUIRED_TAG}")
      message(STATUS "Found BTAS with required revision ${BTAS_REQUIRED_TAG}")
    else()
      message(FATAL_ERROR "Found BTAS with revision ${BTAS_REVISION}, but ${BTAS_REQUIRED_TAG} is required; if BTAS was built by TiledArray, remove the TiledArray install directory, else build the required revision of BTAS")
    endif()
  else (BTAS_REVISION_LINE) # BTAS_REVISION not found? BTAS is not recent enough, reinstall
    set(_msg "Found BTAS, but it is not recent enough OR was not configured and installed with CMake; either provide BTAS with revision ${TA_TRACKED_BTAS_TAG} or let TiledArray download it")
    if (TA_EXPERT)
      message(WARNING "${_msg}")
    else (TA_EXPERT)
      message(FATAL_ERROR "${_msg}")
    endif (TA_EXPERT)
  endif(BTAS_REVISION_LINE)

  # Perform a compile check for BTAS
  cmake_push_check_state()
  
  list(APPEND CMAKE_REQUIRED_INCLUDES ${_BTAS_INCLUDE_DIR})
  list(APPEND CMAKE_REQUIRED_INCLUDES ${Boost_INCLUDE_DIRS})
  list(APPEND CMAKE_REQUIRED_DEFINITIONS -DHAVE_BOOST_CONTAINER)
  list(APPEND CMAKE_REQUIRED_FLAGS -std=c++14)
  CHECK_CXX_SOURCE_COMPILES("
    #include <btas/btas.h>
    #include <iostream>
    #include <cstdlib>
    int main(int argc, char* argv[]){
      auto t = btas::Tensor<int>(2, 3, 4);
      t.generate([]() { return rand() % 100; });
      std::cout << t << std::endl;
    }"
    BTAS_COMPILES)
  
  cmake_pop_check_state()

  if (NOT BTAS_COMPILES)
    message(STATUS "BTAS found at ${_BTAS_INCLUDE_DIR}, but failed to compile test program")
  endif()

  add_library(TiledArray_BTAS INTERFACE)
  set_property(TARGET TiledArray_BTAS PROPERTY
          INTERFACE_INCLUDE_DIRECTORIES ${_BTAS_INCLUDE_DIR})
  install(TARGETS TiledArray_BTAS EXPORT tiledarray COMPONENT tiledarray)

elseif(TA_EXPERT)

  message("** BTAS was not found")
  message(STATUS "** Downloading and building BTAS is explicitly disabled in EXPERT mode")

else()

  include(ExternalProject)

  # Set source and build path for BTAS in the TiledArray Project
  set(EXTERNAL_SOURCE_DIR   ${PROJECT_BINARY_DIR}/external/source/btas)
  set(EXTERNAL_BUILD_DIR  ${PROJECT_BINARY_DIR}/external/build/btas)
  if (NOT BTAS_URL)
    set(BTAS_URL https://github.com/BTAS/BTAS)
  endif (NOT BTAS_URL)
  if (NOT BTAS_TAG)
    set(BTAS_TAG ${TA_TRACKED_BTAS_TAG})
  endif (NOT BTAS_TAG)

  message("** Will clone BTAS from ${BTAS_URL}")

  # non-cmake-configured BTAS has btas/version.h.in, make a version.h to be able to track revision
  file(WRITE ${EXTERNAL_BUILD_DIR}/btas_version.h "#ifndef BTAS_VERSION_H__INCLUDED\n#define BTAS_VERSION_H__INCLUDED\n#define BTAS_REVISION \"${BTAS_TAG}\"\n#endif // BTAS_VERSION_H__INCLUDED\n")

  ExternalProject_Add(btas
    PREFIX ${CMAKE_INSTALL_PREFIX}
    STAMP_DIR ${EXTERNAL_BUILD_DIR}/stamp
    TMP_DIR ${EXTERNAL_BUILD_DIR}/tmp
   #--Download step--------------
    DOWNLOAD_DIR ${EXTERNAL_SOURCE_DIR}
    GIT_REPOSITORY ${BTAS_URL}
    GIT_TAG ${BTAS_TAG}
   #--Configure step-------------
    SOURCE_DIR ${EXTERNAL_SOURCE_DIR}
    CONFIGURE_COMMAND ""
   #--Build step-----------------
    BINARY_DIR ${EXTERNAL_BUILD_DIR}
    BUILD_COMMAND ${CMAKE_COMMAND} -E copy ${EXTERNAL_BUILD_DIR}/btas_version.h ${EXTERNAL_SOURCE_DIR}/btas/version.h
   #--Install step---------------
    INSTALL_COMMAND ""
   #--Custom targets-------------
    STEP_TARGETS download
    )

  # Add BTAS dependency to External
  add_dependencies(External-tiledarraybtas)

  # create an exportable interface target for BTAS
  add_library(TiledArray_BTAS INTERFACE)
  set_property(TARGET TiledArray_BTAS PROPERTY
          INTERFACE_INCLUDE_DIRECTORIES
          $<BUILD_INTERFACE:${EXTERNAL_SOURCE_DIR}>
          $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/btas>)
  install(TARGETS TiledArray_BTAS EXPORT tiledarray COMPONENT tiledarray)

  # how to install BTAS
  install(
    DIRECTORY
        ${EXTERNAL_SOURCE_DIR}/btas
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/btas
    COMPONENT btas
    )

endif()
