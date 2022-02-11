cmake_minimum_required(VERSION 3.12)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY      https://github.com/ValeevGroup/pybind11.git
  GIT_TAG             80d452484c5409444b0ec19383faa84bb7a4d351 # v2.4.3
  )
FetchContent_MakeAvailable(pybind11)

project(python-tiledarray)

set(CMAKE_CXX_STANDARD 17)
add_compile_options(-Wall)

find_package(Eigen3 3.3 REQUIRED)

# find_package(MPI REQUIRED)
# include_directories(${MPI_INCLUDE_PATH})

find_package(Boost REQUIRED) # spirit absent from cmake
# include_directories(${BOOST_INCLUDE_DIRS})

#find_package (Python COMPONENTS Interpreter Development)
#include_directories(${PROJECT_SOURCE_DIR}/pybind11/include)
#python_add_library(python-tiledarray MODULE src/tiledarray.cc WITH_SOABI)

#add_subdirectory(pybind11)
pybind11_add_module(python-tiledarray MODULE src/tiledarray.cc)

# mpi c libs need come from tiledarray
target_link_libraries(python-tiledarray PRIVATE tiledarray)

# pybind uses sized deallocations when using C++17, clang requires -fsized-deallocation compiler flag to enable them
# see https://reviews.llvm.org/D8467 and https://github.com/pybind/pybind11/issues/1818
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  target_compile_options(python-tiledarray PRIVATE -fsized-deallocation)
endif()

set_target_properties(
  python-tiledarray
  PROPERTIES
  #PREFIX ""
  OUTPUT_NAME tiledarray
  )

if (Eigen3::Eigen)
  target_link_libraries(python-tiledarray INTERFACE Eigen3::Eigen)
else()
  include_directories(${EIGEN3_INCLUDE_DIR})
endif()

target_link_libraries(python-tiledarray INTERFACE Boost::boost)

# tests
if (BUILD_TESTING)
  # build step
  add_test(tiledarray/unit/python/build "${CMAKE_COMMAND}" --build ${CMAKE_BINARY_DIR} --target python-tiledarray)
  set_tests_properties(tiledarray/unit/python/build PROPERTIES FIXTURES_SETUP TA_UNIT_TESTS_PYTHON_EXEC)

  # check for presence of prerequisite modules
  foreach(_mod pytest numpy)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "import ${_mod}"
        OUTPUT_QUIET ERROR_QUIET
        RESULTS_VARIABLE check_for_${_mod})
    if (check_for_${_mod})
      message(FATAL_ERROR "Python module \"${_mod}\" is not installed; install via \"${PYTHON_EXECUTABLE} -m pip install ${_mod}\" and rerun cmake")
    endif(check_for_${_mod})
  endforeach(_mod)

  # test step
  add_test(
    NAME tiledarray/unit/python/run
    # need to use pytest to find tiledarray module properly
    COMMAND ${PYTHON_EXECUTABLE} -m pytest ${PROJECT_SOURCE_DIR}/test_tiledarray.py -v
    WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  )
  set_tests_properties(tiledarray/unit/python/run
          PROPERTIES FIXTURES_REQUIRED TA_UNIT_TESTS_PYTHON_EXEC
          ENVIRONMENT MAD_NUM_THREADS=2)

endif(BUILD_TESTING)
