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

enable_testing()

# add the executable
add_test(
  NAME test_tiledarray
  # need to use pytest to find tiledarray module properly
  COMMAND ${Python3_EXECUTABLE} -m pytest ${PROJECT_SOURCE_DIR}/test_tiledarray.py
  WORKING_DIRECTORY ${PROJECT_BINARY_DIR}
  )
