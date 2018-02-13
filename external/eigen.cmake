# -*- mode: cmake -*-

include(CMakePushCheckState)
include(CheckCXXSourceCompiles)
include(AppendFlags)

# Check for existing Eigen
# prefer CMake-configured-and-installed instance
find_package(Eigen3 3.1 NO_MODULE)
if (TARGET Eigen3::Eigen)
  # import alias into TiledArray "namespace"
  # TODO bump CMake requirement to 3.11 when available, uncomment this and remove the rest of this clause
  # add_library(TiledArray_Eigen ALIAS Eigen3::Eigen)
  add_library(TiledArray_Eigen INTERFACE)
  foreach(prop INTERFACE_INCLUDE_DIRECTORIES INTERFACE_COMPILE_DEFINITIONS INTERFACE_COMPILE_OPTIONS INTERFACE_LINK_LIBRARIES INTERFACE_POSITION_INDEPENDENT_CODE)
    get_property(EIGEN3_${prop} TARGET Eigen3::Eigen PROPERTY ${prop})
    set_property(TARGET TiledArray_Eigen PROPERTY
            ${prop} ${EIGEN3_${prop}})
  endforeach()
  install(TARGETS TiledArray_Eigen EXPORT tiledarray COMPONENT tiledarray)
else (TARGET Eigen3::Eigen)
  # otherwise use bundled FindEigen3.cmake module controlled by EIGEN3_INCLUDE_DIR
  find_package(Eigen3 3.1)

  if (EIGEN3_FOUND)
    add_library(TiledArray_Eigen INTERFACE)
    set_property(TARGET TiledArray_Eigen PROPERTY
            INTERFACE_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIR})
    install(TARGETS TiledArray_Eigen EXPORT tiledarray COMPONENT tiledarray)
  endif (EIGEN3_FOUND)
endif (TARGET Eigen3::Eigen)

# validate found
if (TARGET TiledArray_Eigen)

  # Perform a compile check with Eigen
  cmake_push_check_state()

  get_property(EIGEN3_INCLUDE_DIRS TARGET TiledArray_Eigen PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  list(APPEND CMAKE_REQUIRED_INCLUDES ${EIGEN3_INCLUDE_DIRS})
  CHECK_CXX_SOURCE_COMPILES("
    #include <Eigen/Core>
    #include <Eigen/Dense>
    #include <Eigen/SparseCore>
    #include <iostream>
    int main(int argc, char* argv[]){
      Eigen::MatrixXd m = Eigen::MatrixXd::Random(5, 5);
      m = m.transpose() + m;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(m);
      Eigen::MatrixXd m_invsqrt = eig.operatorInverseSqrt();
      std::cout << m_invsqrt << std::endl;
    }"
    EIGEN3_COMPILES)
    
  cmake_pop_check_state()

  if (NOT EIGEN3_COMPILES)
    message(FATAL_ERROR "Eigen3 found, but failed to compile test program")
  endif()

elseif(TA_EXPERT)

  message("** Eigen3 was not found")
  message(FATAL_ERROR "** Downloading and building Eigen3 is explicitly disabled in EXPERT mode")

else()

  include(ExternalProject)

  # Set source and build path for Eigen3 in the TiledArray Project
  set(EXTERNAL_SOURCE_DIR   ${PROJECT_BINARY_DIR}/external/source/eigen)
  set(EXTERNAL_BUILD_DIR  ${PROJECT_BINARY_DIR}/external/build/eigen)
  set(EIGEN3_URL https://bitbucket.org/eigen/eigen)
  set(EIGEN3_TAG 3.2.4)

  message("** Will build Eigen from ${EIGEN3_URL}")

  ExternalProject_Add(eigen3
    PREFIX ${CMAKE_INSTALL_PREFIX}
    STAMP_DIR ${EXTERNAL_BUILD_DIR}/stamp
    TMP_DIR ${EXTERNAL_BUILD_DIR}/tmp
   #--Download step--------------
    DOWNLOAD_DIR ${EXTERNAL_SOURCE_DIR}
    HG_REPOSITORY ${EIGEN3_URL}
    HG_TAG ${EIGEN3_TAG}
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

  # Add eigen3 dependency to External
  add_dependencies(External eigen3)

  # create an exportable interface target for eigen3
  add_library(TiledArray_Eigen INTERFACE)
  set_property(TARGET TiledArray_Eigen PROPERTY
          INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${EXTERNAL_SOURCE_DIR}>
          $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/eigen3>)
  install(TARGETS TiledArray_Eigen EXPORT tiledarray COMPONENT tiledarray)

  # Install Eigen 3
  install(
    DIRECTORY
        ${EXTERNAL_SOURCE_DIR}/Eigen
        ${EXTERNAL_SOURCE_DIR}/unsupported
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/eigen3
    COMPONENT eigen3
    )
  install(
    FILES ${EXTERNAL_SOURCE_DIR}/signature_of_eigen3_matrix_library
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/eigen3
    COMPONENT eigen3
    )

endif()
