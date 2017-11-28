# -*- mode: cmake -*-

include(CMakePushCheckState)
include(CheckCXXSourceCompiles)
include(AppendFlags)

# Check for Eigen
find_package(Eigen3 3.1)

if (EIGEN3_FOUND)

  # Perform a compile check with Eigen
  cmake_push_check_state()
  
  list(APPEND CMAKE_REQUIRED_INCLUDES ${EIGEN3_INCLUDE_DIR})
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
    message(FATAL_ERROR "Eigen3 found at ${EIGEN3_INCLUDE_DIR}, but failed to compile test program")
  endif()

  list(APPEND TiledArray_CONFIG_INCLUDE_DIRS ${EIGEN3_INCLUDE_DIR})
  
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
   #--Download step--------------
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

  # Add eigen3 dependency
  add_dependencies(External eigen3)
  
  # Set the Eigen 3 included directory
  set(EIGEN3_INCLUDE_DIR ${EXTERNAL_SOURCE_DIR})
  
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

# Set the  build variables
include_directories(${EIGEN3_INCLUDE_DIR})
