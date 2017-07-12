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
  
  ##### uncomment, if ready to bundle with TiledArray #####
#  list(APPEND TiledArray_CONFIG_INCLUDE_DIRS ${_BTAS_INCLUDE_DIR})
  
elseif(TA_EXPERT)

  message("** BTAS was not found")
  message(STATUS "** Downloading and building BTAS is explicitly disabled in EXPERT mode")

else()

  include(ExternalProject)

  # Set source and build path for Eigen3 in the TiledArray Project
  set(EXTERNAL_SOURCE_DIR   ${PROJECT_BINARY_DIR}/external/source/btas)
  set(EXTERNAL_BUILD_DIR  ${PROJECT_BINARY_DIR}/external/build/btas)
  if (NOT BTAS_URL)
    set(BTAS_URL https://github.com/BTAS/BTAS)
  endif (NOT BTAS_URL)
  set(BTAS_TAG master)

  message("** Will clone BTAS from ${BTAS_URL}")

  ExternalProject_Add(btas
    PREFIX ${CMAKE_INSTALL_PREFIX}
    STAMP_DIR ${EXTERNAL_BUILD_DIR}/stamp
   #--Download step--------------
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

  # Add btas dependency
  add_dependencies(External btas)
  
  # Set the BTAS included directory
  set(BTAS_INCLUDE_DIR ${EXTERNAL_SOURCE_DIR})
  
  ##### uncomment, if ready to bundle with TiledArray #####
#  # Install BTAS
#  install(
#    DIRECTORY
#        ${EXTERNAL_SOURCE_DIR}/btas
#    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/btas
#    COMPONENT btas
#    )

endif()

# Set the  build variables
include_directories(${BTAS_INCLUDE_DIR})
