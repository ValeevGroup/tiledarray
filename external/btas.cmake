# -*- mode: cmake -*-

include(CMakePushCheckState)
include(CheckCXXSourceCompiles)
include(AppendFlags)

find_path(_BTAS_INCLUDE_DIR btas/btas.h PATHS ${BTAS_INCLUDE_DIR})

if (_BTAS_INCLUDE_DIR)

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
    set(BTAS_TAG 7a4fae1259da91f988a56f6bfa54ab18c1481393)
  endif (NOT BTAS_TAG)

  message("** Will clone BTAS from ${BTAS_URL}")

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
    BUILD_COMMAND ""
   #--Install step---------------
    INSTALL_COMMAND ""
   #--Custom targets-------------
    STEP_TARGETS download
    )

  # Add BTAS dependency to External
  add_dependencies(External btas)

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
