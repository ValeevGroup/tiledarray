# query EIGEN3_DIR and (deprecated) EIGEN_DIR envvars
if (DEFINED ENV{EIGEN3_DIR})
  set(ENV_EIGEN3_DIR "$ENV{EIGEN3_DIR}")
else()
  set(ENV_EIGEN3_DIR "$ENV{EIGEN_DIR}")
endif()
set(EIGEN3_INCLUDE_DIR "${ENV_EIGEN3_DIR}" CACHE PATH "Eigen3 library directory")
