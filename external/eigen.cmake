# -*- mode: cmake -*-

include(CMakePushCheckState)
include(CheckCXXSourceCompiles)
include(AppendFlags)

# if CUDA is enabled (assuming CUDA version is 9 or 10) need Eigen 3.3.7
# see https://gitlab.com/libeigen/eigen/issues/1491
if (ENABLE_CUDA)
  set(_tiledarray_required_eigen_version 3.3.7)
else(ENABLE_CUDA)
  set(_tiledarray_required_eigen_version ${TA_TRACKED_EIGEN_VERSION})
endif(ENABLE_CUDA)

# Check for existing Eigen
# prefer CMake-configured-and-installed instance
# re:NO_CMAKE_PACKAGE_REGISTRY: eigen3 registers its *build* tree with the user package registry ...
#                               to avoid issues with wiped build directory look for installed eigen
find_package(Eigen3 ${_tiledarray_required_eigen_version} NO_MODULE QUIET NO_CMAKE_PACKAGE_REGISTRY)
if (TARGET Eigen3::Eigen)
  # import alias into TiledArray "namespace"
  add_library(TiledArray_Eigen INTERFACE)
  foreach(prop INTERFACE_INCLUDE_DIRECTORIES INTERFACE_COMPILE_DEFINITIONS INTERFACE_COMPILE_OPTIONS INTERFACE_LINK_LIBRARIES INTERFACE_POSITION_INDEPENDENT_CODE)
    get_property(EIGEN3_${prop} TARGET Eigen3::Eigen PROPERTY ${prop})
    set_property(TARGET TiledArray_Eigen PROPERTY
            ${prop} ${EIGEN3_${prop}})
  endforeach()
else (TARGET Eigen3::Eigen)
  # otherwise use bundled FindEigen3.cmake module controlled by EIGEN3_INCLUDE_DIR
  # but make sure EIGEN3_INCLUDE_DIR exists!
  find_package(Eigen3 ${_tiledarray_required_eigen_version})

  if (EIGEN3_FOUND)
    if (NOT EXISTS "${EIGEN3_INCLUDE_DIR}")
      message(WARNING "Eigen3 is \"found\", but the reported EIGEN3_INCLUDE_DIR=${EIGEN3_INCLUDE_DIR} does not exist; likely corrupt Eigen3 build registered in user or system package registry; specify EIGEN3_INCLUDE_DIR manually or (better) configure (with CMake) and install Eigen3 package")
    else(NOT EXISTS "${EIGEN3_INCLUDE_DIR}")
      add_library(TiledArray_Eigen INTERFACE)
      set_property(TARGET TiledArray_Eigen PROPERTY
              INTERFACE_INCLUDE_DIRECTORIES ${EIGEN3_INCLUDE_DIR})
    endif(NOT EXISTS "${EIGEN3_INCLUDE_DIR}")
  endif (EIGEN3_FOUND)
endif (TARGET Eigen3::Eigen)

# validate found
if (TARGET TiledArray_Eigen)

  # Perform a compile check with Eigen
  cmake_push_check_state()

  # INTERFACE libraries cannot be used as CMAKE_REQUIRED_LIBRARIES, so must manually transfer deps info
  get_property(EIGEN3_INCLUDE_DIRS TARGET TiledArray_Eigen PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
  list(APPEND CMAKE_REQUIRED_INCLUDES ${EIGEN3_INCLUDE_DIRS} ${PROJECT_BINARY_DIR}/src ${PROJECT_SOURCE_DIR}/src
       ${LAPACK_INCLUDE_DIRS})
  list(APPEND CMAKE_REQUIRED_LIBRARIES ${LAPACK_LIBRARIES})
  foreach(_def ${LAPACK_COMPILE_DEFINITIONS})
    list(APPEND CMAKE_REQUIRED_DEFINITIONS "-D${_def}")
  endforeach()
  list(APPEND CMAKE_REQUIRED_FLAGS ${LAPACK_COMPILE_OPTIONS})

  CHECK_CXX_SOURCE_COMPILES("
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

  # last resort: unpack and copy to install dir
  # N.B. NOT building via FetchContent since Eigen3 is not subprojectable due to polluting global namespace of targets (e.g. lapack, check, etc.)

  set(Eigen3_VERSION ${TA_INSTALL_EIGEN_VERSION})
  set(EIGEN3_URL_HASH ${TA_INSTALL_EIGEN_URL_HASH})
  set(EIGEN3_URL https://gitlab.com/libeigen/eigen/-/archive/${Eigen3_VERSION}/eigen-${Eigen3_VERSION}.tar.bz2)

  include(ExternalProject)

  # Set source and build path for Eigen3 in the TiledArray Project
  set(EXTERNAL_SOURCE_DIR   ${FETCHCONTENT_BASE_DIR}/eigen-src)
  set(EXTERNAL_BUILD_DIR  ${FETCHCONTENT_BASE_DIR}/eigen-build)

  message("** Will build Eigen from ${EIGEN3_URL}")

  ExternalProject_Add(eigen3
    PREFIX ${CMAKE_INSTALL_PREFIX}
   #--Download step--------------
    DOWNLOAD_DIR ${EXTERNAL_SOURCE_DIR}
    URL ${EIGEN3_URL}
    URL_HASH ${EIGEN3_URL_HASH}
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
  add_dependencies(External-tiledarray eigen3)

  # create an exportable interface target for eigen3
  add_library(TiledArray_Eigen INTERFACE)
  set_property(TARGET TiledArray_Eigen PROPERTY
          INTERFACE_INCLUDE_DIRECTORIES $<BUILD_INTERFACE:${EXTERNAL_SOURCE_DIR}>
          $<INSTALL_INTERFACE:${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}/eigen3>)

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

# finish configuring TiledArray_Eigen and install
if (TARGET TiledArray_Eigen)
  set(TiledArray_Eigen_VERSION "${Eigen3_VERSION}" CACHE STRING "Eigen3_VERSION of the library interfaced by TiledArray_Eigen target")
  install(TARGETS TiledArray_Eigen EXPORT tiledarray COMPONENT tiledarray)
endif(TARGET TiledArray_Eigen)
