# -*- mode: cmake -*-

# Check for Eigen
find_package(Eigen 3.0)

if (EIGEN_FOUND)

  # Perform a compile check with Eigen
  list(APPEND CMAKE_REQUIRED_INCLUDES ${EIGEN_INCLUDE_DIR})
  CHECK_CXX_SOURCE_COMPILES("
    #include <Eigen/Core>
    #include <Eigen/Dense>
    #include <iostream>
    int main(int argc, char* argv[]){
      Eigen::MatrixXd m = Eigen::MatrixXd::Random(5, 5);
      m = m.transpose() + m;
      Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eig(m);
      Eigen::MatrixXd m_invsqrt = eig.operatorInverseSqrt();
      std::cout << m_invsqrt << std::endl;
    }"
    EIGEN_COMPILES)

  if (NOT EIGEN_COMPILES)
    message(FATAL_ERROR "Eigen found at ${Eigen_INCLUDE_DIR}, but failed to compile test program")
  endif()
  
elseif(TA_EXPERT)

  message("** Eigen was not found")
  message(FATAL_ERROR "** Downloading and building Eigen is explicitly disabled in EXPERT mode")

else()

  include(ExternalProject)

  # Set source and build path for Eigen in the TiledArray Project
  set(EXTERNAL_DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/external/src)
  set(EXTERNAL_SOURCE_DIR   ${PROJECT_SOURCE_DIR}/external/src/eigen)

  # Set the external source
  if (EXISTS ${EXTERNAL_DOWNLOAD_DIR}/eigen.tar.gz)
    # Use local file
    set(EIGEN_URL ${EXTERNAL_DOWNLOAD_DIR}/eigen.tar.gz)
    set(EIGEN_URL_HASH "")
  else()
    # Download remote file
    set(EIGEN_URL https://bitbucket.org/eigen/eigen/get/3.2.0.tar.gz)
    set(EIGEN_URL_HASH MD5=d41d8cd98f00b204e9800998ecf8427e)
  endif()

  message("** Will build Eigen from ${EIGEN_URL}")

  ExternalProject_Add(eigen3
    PREFIX ${CMAKE_INSTALL_PREFIX}
    STAMP_DIR ${EXTERNAL_BUILD_DIR}/stamp
   #--Download step--------------
    URL ${EIGEN_URL}
    URL_HASH ${EIGEN_URL_HASH}
    DOWNLOAD_DIR ${EXTERNAL_DOWNLOAD_DIR}
   #--Configure step-------------
    SOURCE_DIR ${EXTERNAL_SOURCE_DIR}
    CONFIGURE_COMMAND ""
   #--Build step-----------------
    BUILD_COMMAND ""
   #--Install step---------------
    INSTALL_COMMAND ""
   #--Custom targets-------------
    STEP_TARGETS download
    )

  # Add eigen3 dependency
  add_dependencies(External eigen3)
  
  # Set the Eigen 3 included directory
  set(EIGEN_INCLUDE_DIR ${EXTERNAL_SOURCE_DIR})
  
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

include_directories(${EIGEN_INCLUDE_DIR})
message(STATUS "\tEigen include dir: ${EIGEN_INCLUDE_DIR}")
